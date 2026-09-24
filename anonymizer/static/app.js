const $ = (id) => document.getElementById(id);
const video = $("video");
const labels = {queued: "Queued", processing: "Anonymizing", checking: "Checking", completed: "Completed", needs_review: "Needs review", failed: "Failed", passed: "Passed", flagged: "Flagged", waiting: "Waiting"};
let currentId = "", currentJob = null, selectedMedia = null, uploadId = crypto.randomUUID();
let selectedFinding = null, findings = [], visibleFlags = 50, visibleBriefFlags = 50, reportGeneration = 0, reportGeometry = null;
let reportSignature = "", jobsSignature = "", chunkSignature = "", polling = false;

function timeLabel(seconds) {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60).toString().padStart(2, "0")}:${(whole % 60).toString().padStart(2, "0")}`;
}
async function jsonRequest(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`);
  }
  return response.json();
}
function showError(error) { $("error").textContent = error.message; $("error").hidden = false; }
function badge(element, status, text = labels[status] || status) { element.className = `badge ${status}`; element.textContent = text; }
function progress(id, done, total, finished) {
  const element = $(id);
  if (finished) { element.max = 1; element.value = 1; }
  else if (total) { element.max = total; element.value = done; }
  else { element.max = 1; element.removeAttribute("value"); }
}

function renderJobs(jobs) {
  const signature = JSON.stringify(jobs.map(j => [j.upload_id, j.filename, j.status])) + currentId;
  if (signature === jobsSignature) return;
  jobsSignature = signature;
  const buttons = jobs.map(job => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "job-button";
    button.setAttribute("aria-current", job.upload_id === currentId ? "true" : "false");
    const title = document.createElement("strong"); title.textContent = job.filename || job.upload_id;
    const detail = document.createElement("small"); detail.textContent = `${labels[job.status]} · ${new Date(job.created_at).toLocaleDateString()}`;
    button.append(title, detail); button.addEventListener("click", () => openJob(job.upload_id));
    return button;
  });
  if (!buttons.length) { const p = document.createElement("p"); p.className = "muted small"; p.textContent = "No uploads yet."; buttons.push(p); }
  $("jobs").replaceChildren(...buttons);
}

function renderJob(job) {
  currentJob = job;
  $("empty").hidden = true; $("job").hidden = false;
  $("job-name").textContent = job.filename || "Processed video";
  $("job-id").textContent = job.upload_id;
  badge($("status"), job.status);
  $("job-error").hidden = !job.error; $("job-error").textContent = job.error || "";
  const legacy = !job.pipeline_version;
  const redacted = job.redaction_status === "completed" || legacy;
  $("redaction-state").textContent = legacy ? "Earlier pipeline" : labels[job.redaction_status];
  $("checking-state").textContent = labels[job.checking_status] || job.checking_status;
  progress("redaction-progress", job.frames_done, job.total_frames, redacted);
  progress("checking-progress", job.checking_frames, job.total_frames, job.checking_status === "completed");
  $("redaction-count").textContent = `${job.frames_done.toLocaleString()}${job.total_frames ? ` / ${job.total_frames.toLocaleString()}` : ""} frames · ${job.chunks.length} chunks ready`;
  $("checking-count").textContent = `${job.checking_frames.toLocaleString()} frames checked · ${job.checked_chunks} / ${job.chunks.length} chunks`;
  $("audio-label").textContent = legacy ? "Earlier output — audio may be present" : "Video only — audio removed";
  $("download").hidden = job.status !== "completed";
  $("download").href = `/jobs/${encodeURIComponent(currentId)}/video`;
  $("download-help").textContent = job.status === "completed" ? "" : job.status === "needs_review" ? "Download blocked: review flagged frames." : "Download unlocks after all checks pass.";
  if (selectedMedia === null) {
    if (job.chunks.length) selectedMedia = job.chunks[0].chunk_index;
    else if (job.full_preview_ready) selectedMedia = "full";
  }
  renderChunks();
  renderPreview();
}

function renderChunks() {
  const job = currentJob;
  const signature = JSON.stringify([currentId, selectedMedia, job.full_preview_ready, job.status,
    job.chunks.map(c => [c.chunk_index, c.status])]);
  if (signature === chunkSignature) return;
  chunkSignature = signature;
  const make = (key, title, status, disabled = false) => {
    const button = document.createElement("button"); button.type = "button"; button.className = "chunk"; button.disabled = disabled;
    button.setAttribute("aria-pressed", key === selectedMedia ? "true" : "false");
    const name = document.createElement("strong"); name.textContent = title;
    const state = document.createElement("span"); badge(state, status, status === "queued" ? "Unchecked" : labels[status] || status);
    button.append(name, state);
    button.addEventListener("click", () => {
      selectedMedia = key; selectedFinding = null; reportSignature = "";
      resetFindings(); reportGeneration++; $("report").hidden = true;
      clearBoxes(); renderChunks(); renderPreview();
    });
    return button;
  };
  const buttons = [make("full", "Full video", job.full_preview_ready ? job.status : "Not ready", !job.full_preview_ready)];
  for (const chunk of job.chunks) buttons.push(make(chunk.chunk_index, `${timeLabel(chunk.start_seconds)}–${timeLabel(chunk.start_seconds + chunk.duration_seconds)}`, chunk.status));
  $("chunks").replaceChildren(...buttons);
}

function renderPreview() {
  const job = currentJob, base = `/jobs/${encodeURIComponent(currentId)}`;
  const ready = selectedMedia !== null;
  $("player").hidden = !ready; $("player-empty").hidden = ready;
  if (!ready) {
    $("preview-state").textContent = "";
    $("flag-help").textContent = "Findings appear after each chunk has been checked.";
    return;
  }
  const chunk = job.chunks.find(c => c.chunk_index === selectedMedia);
  const source = selectedMedia === "full" ? `${base}/preview` : `${base}/chunks/${selectedMedia}/preview`;
  if (video.getAttribute("src") !== source) {
    video.pause(); selectedFinding = null; reportGeometry = null; clearBoxes();
    $("selected-flag").textContent = ""; $("video-error").hidden = true;
    video.src = source;
  }
  const status = chunk?.status || job.status;
  $("preview-state").textContent = ["passed", "completed"].includes(status) ? "Checked — no residual faces detected." :
    ["flagged", "needs_review"].includes(status) ? "Flagged — possible visible faces need operator review." :
    status === "failed" ? "Check failed — this preview has not passed quality checking." : "Unchecked preview — quality checking is still in progress.";
  loadFindings().catch(showError);
}

async function loadFindings() {
  const job = currentJob, id = currentId, media = selectedMedia;
  const selectedChunks = job.chunks.filter(c => (media === "full" || c.chunk_index === media) && ["passed", "flagged"].includes(c.status));
  // A later chunk can confirm detections at the end of an earlier chunk.
  const signature = JSON.stringify([id, media, job.status, job.chunks.map(c => [c.chunk_index, c.status])]);
  if (signature === reportSignature) return;
  const generation = ++reportGeneration;
  const base = `/jobs/${encodeURIComponent(id)}`;
  let loaded = [];
  let geometry = null;
  if (!job.pipeline_version && ["completed", "needs_review"].includes(job.status)) {
    const report = await jsonRequest(`${base}/report`);
    geometry = report.processing;
    loaded = report.findings.map(f => ({...f, displayTime: f.timestamp_seconds}));
  } else {
    for (const chunk of selectedChunks) {
      const report = await jsonRequest(`${base}/chunks/${chunk.chunk_index}/report`);
      geometry = report.processing;
      loaded.push(...report.findings.map(f => ({...f,
        frame: f.frame + chunk.start_frame,
        displayTime: f.timestamp_seconds + chunk.start_seconds,
        timestamp_seconds: f.timestamp_seconds + (media === "full" ? chunk.start_seconds : 0),
      })));
    }
  }
  if (generation !== reportGeneration || id !== currentId || media !== selectedMedia) return;
  reportSignature = signature; findings = loaded; reportGeometry = geometry;
  const fullReportReady = media === "full" && ["completed", "needs_review"].includes(job.status);
  $("report").hidden = !fullReportReady && !selectedChunks.length;
  $("report").href = fullReportReady ? `${base}/report` : `${base}/chunks/${selectedChunks[0]?.chunk_index}/report`;
  if (media === "full" && !fullReportReady) $("report").hidden = true;
  renderFlags();
}
function resetFindings() {
  findings = []; visibleFlags = visibleBriefFlags = 50;
  $("flags").replaceChildren(); $("brief-flags").replaceChildren();
  $("persistent-findings").hidden = $("brief-findings").hidden = true;
  $("brief-findings").open = false;
  badge($("flag-count"), "queued", "0 flagged frames");
}
function renderFindingList(id, moreId, rows, limit) {
  const items = rows.slice(0, limit).map(finding => {
    const item = document.createElement("li"), button = document.createElement("button"); button.type = "button";
    const milliseconds = Math.round(finding.displayTime * 1000);
    const time = document.createElement("time"); time.textContent = `${timeLabel(milliseconds / 1000)}.${(milliseconds % 1000).toString().padStart(3, "0")}`;
    const detail = document.createElement("span"); detail.textContent = `Frame ${finding.frame} · ${finding.faces.length} possible face${finding.faces.length === 1 ? "" : "s"} · ${Math.round(Math.max(...finding.faces.map(f => f.confidence)) * 100)}%`;
    button.append(time, detail); button.addEventListener("click", () => jumpToFlag(finding)); item.append(button); return item;
  });
  $(id).replaceChildren(...items); $(moreId).hidden = rows.length <= limit;
}
function renderFlags() {
  badge($("flag-count"), findings.length ? "flagged" : "queued", `${findings.length} flagged frames`);
  $("flag-help").textContent = findings.length ? "Persistent and unconfirmed findings both block download. Select a frame to inspect it; times refer to the original video." :
    currentJob.checking_status === "completed" || currentJob.chunks.find(c => c.chunk_index === selectedMedia)?.status === "passed" ? "No flags in this preview." : "No findings available yet; checking may still be in progress.";
  const group = persistent => findings.map(finding => ({...finding,
    faces: finding.faces.filter(face => (face.confirmation === "persistent") === persistent),
  })).filter(finding => finding.faces.length);
  const persistent = group(true), brief = group(false);
  $("persistent-findings").hidden = !persistent.length;
  $("brief-findings").hidden = !brief.length;
  $("persistent-count").textContent = `${persistent.length} frame${persistent.length === 1 ? "" : "s"}`;
  $("brief-count").textContent = `${brief.length} frame${brief.length === 1 ? "" : "s"}`;
  $("brief-help").textContent = findings.some(f => f.faces.some(face => !face.confirmation)) ?
    "Includes earlier results without temporal confirmation. These findings still need review." :
    "Too few matching detections to confirm persistence. Briefly visible faces can still be real. Counts may overlap when a frame contains both kinds.";
  renderFindingList("flags", "more-flags", persistent, visibleFlags);
  renderFindingList("brief-flags", "more-brief-flags", brief, visibleBriefFlags);
}
function clearBoxes() { $("boxes").replaceChildren(); }
function positionBoxes() {
  if (!video.videoWidth) return;
  const scale = Math.min(video.clientWidth / video.videoWidth, video.clientHeight / video.videoHeight);
  const width = video.videoWidth * scale, height = video.videoHeight * scale;
  const svg = $("boxes");
  // Reports use encoded pixels; the browser may display a different sample aspect ratio.
  svg.setAttribute("viewBox", `0 0 ${reportGeometry?.width || video.videoWidth} ${reportGeometry?.height || video.videoHeight}`);
  svg.setAttribute("preserveAspectRatio", "none");
  Object.assign(svg.style, {width: `${width}px`, height: `${height}px`, left: `${(video.clientWidth - width) / 2}px`, top: `${(video.clientHeight - height) / 2}px`});
}
function drawSelectedFlag() {
  clearBoxes(); positionBoxes();
  if (!selectedFinding || !video.paused || Math.abs(video.currentTime - selectedFinding.timestamp_seconds) > .01) return;
  for (const face of selectedFinding.faces) {
    const [x1, y1, x2, y2] = face.box;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    for (const [key, value] of Object.entries({x:x1, y:y1, width:x2-x1, height:y2-y1})) rect.setAttribute(key, value);
    $("boxes").append(rect);
  }
}
function jumpToFlag(finding) {
  selectedFinding = finding; video.pause(); video.currentTime = finding.timestamp_seconds + .000001;
  $("selected-flag").textContent = `Frame ${finding.frame}: ` + finding.faces.map(face => `box [${face.box.map(Math.round).join(", ")}], ${(face.confidence * 100).toFixed(1)}%`).join("; ");
  drawSelectedFlag();
}
function openJob(id) {
  currentId = id; currentJob = null; selectedMedia = null; selectedFinding = null;
  reportSignature = ""; reportGeneration++; resetFindings();
  history.replaceState(null, "", id ? `#${encodeURIComponent(id)}` : location.pathname);
  video.pause(); video.removeAttribute("src"); video.load(); clearBoxes();
  $("flags").replaceChildren(); $("selected-flag").textContent = ""; $("report").hidden = true;
  $("error").hidden = true; $("job").hidden = true; $("empty").hidden = false;
  refresh();
}
async function refresh() {
  if (polling) return;
  polling = true;
  const requestedId = currentId;
  try {
    const [jobs, queues, health] = await Promise.all([
      jsonRequest("/jobs"), jsonRequest("/queues"), fetch("/health").then(r => r.json()),
    ]);
    if (requestedId !== currentId) return;
    renderJobs(jobs);
    for (const [role, queue] of [["redact", "anonymization"], ["check", "checking"]]) {
      const state = health.workers[role].state;
      $(role + "-service").textContent = state === "stopped" ? "Offline" : state === "idle" ? "Ready" : state[0].toUpperCase() + state.slice(1);
      $(role + "-service").className = state === "stopped" ? "offline" : "";
      $(role + "-queue").textContent = `${queues[queue].queued} queued · ${queues[queue].active} active`;
    }
    if (!currentId && jobs.length) { polling = false; openJob(jobs[0].upload_id); return; }
    if (currentId) {
      const job = jobs.find(j => j.upload_id === currentId) || await jsonRequest(`/jobs/${encodeURIComponent(currentId)}`);
      if (requestedId === currentId) renderJob(job);
    }
    $("error").hidden = true;
  } catch (error) { if (requestedId === currentId) showError(error); }
  finally { polling = false; }
}

$("upload-form").addEventListener("submit", async event => {
  event.preventDefault(); const file = $("file").files[0]; if (!file) return;
  $("file").disabled = true; $("upload-button").disabled = true; $("upload-message").textContent = "Uploading…";
  try {
    const form = new FormData(); form.append("upload_id", uploadId); form.append("video", file);
    const job = await jsonRequest("/jobs", {method:"POST", body:form});
    $("upload-message").textContent = "Upload accepted. Preview chunks will appear as they finish.";
    $("file").value = ""; uploadId = crypto.randomUUID(); openJob(job.upload_id);
  } catch (error) { $("upload-message").textContent = error.message; }
  finally { $("file").disabled = false; $("upload-button").disabled = false; }
});
$("file").addEventListener("change", () => { uploadId = crypto.randomUUID(); });
$("refresh").addEventListener("click", refresh);
$("more-flags").addEventListener("click", () => { visibleFlags += 50; renderFlags(); });
$("more-brief-flags").addEventListener("click", () => { visibleBriefFlags += 50; renderFlags(); });
window.addEventListener("hashchange", () => { try { openJob(decodeURIComponent(location.hash.slice(1))); } catch (error) { showError(error); } });
video.addEventListener("loadedmetadata", () => { positionBoxes(); drawSelectedFlag(); });
video.addEventListener("seeked", drawSelectedFlag); video.addEventListener("seeking", clearBoxes); video.addEventListener("play", clearBoxes);
video.addEventListener("error", () => { if (video.getAttribute("src")) { $("video-error").textContent = "Preview unavailable. Use the local server; the file may be missing or unsupported by your browser."; $("video-error").hidden = false; } });
new ResizeObserver(positionBoxes).observe(video);
try { currentId = decodeURIComponent(location.hash.slice(1)); } catch (error) { showError(error); }
refresh(); setInterval(refresh, 2000);
