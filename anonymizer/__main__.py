"""One command starts the API and both independently owned worker processes."""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time


def serve():
    import uvicorn
    uvicorn.run("anonymizer.api:app", host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8000")), workers=1)


def launch():
    from .config import SETTINGS
    from .database import Jobs
    from .runtime import RoleLock
    SETTINGS.prepare()
    Jobs(SETTINGS.database).initialize()
    with RoleLock(SETTINGS.data_dir / "workers", "launcher"):
        stopping = False
        children = {}
        failures = {role: [] for role in ("api", "redact", "check")}

        def stop(*_):
            nonlocal stopping
            stopping = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, stop)

        def start(role):
            return subprocess.Popen([sys.executable, "-m", "anonymizer", "--role", role])

        try:
            for role in failures:
                children[role] = start(role)
            while not stopping:
                for role, process in children.items():
                    code = process.poll()
                    if code is None:
                        continue
                    now = time.monotonic()
                    failures[role] = [t for t in failures[role] if now - t < 60] + [now]
                    if code == 2 or len(failures[role]) > 3:
                        raise RuntimeError(f"{role} could not stay running; see its error above")
                    logging.warning("Restarting %s (exit %s)", role, code)
                    children[role] = start(role)
                time.sleep(0.5)
        finally:
            for process in children.values():
                if process.poll() is None:
                    process.terminate()
            for process in children.values():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("all", "api", "redact", "check"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(levelname)s %(message)s")
    try:
        if args.role == "all":
            launch()
        elif args.role == "api":
            serve()
        else:
            from .worker import run_worker
            run_worker(args.role)
    except RuntimeError as error:
        logging.error("%s", error)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
