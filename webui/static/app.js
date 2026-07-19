// Thin client: polls /api/runs, /api/jobs, /api/artifacts and submits forms
// to the corresponding POST endpoints. No framework, no build step.

const POLL_MS = 3000;
let lastRuns = [];

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function fmt(v) {
  return v === null || v === undefined ? "-" : v;
}

function renderRuns(runs) {
  lastRuns = runs;
  const tbody = document.getElementById("runs-tbody");
  const empty = document.getElementById("runs-empty");
  tbody.innerHTML = "";
  empty.hidden = runs.length > 0;

  for (const r of runs) {
    const tr = document.createElement("tr");
    const statusBadge = r.running
      ? '<span class="badge badge-running">running</span>'
      : '<span class="badge badge-idle">idle</span>';
    tr.innerHTML = `
      <td><input type="checkbox" class="compare-check" data-name="${r.name}" data-kind="${r.kind}"></td>
      <td>${r.name}</td>
      <td>${r.kind === "ppo" ? "PPO baseline" : "Dreamer"}</td>
      <td>${fmt(r.step)}</td>
      <td>${fmt(r.best_x)}</td>
      <td>${fmt(r.flags)}</td>
      <td>${r.size_mb} MB</td>
      <td>${statusBadge}</td>
      <td>
        <button class="dashboard-btn" data-name="${r.name}" data-kind="${r.kind}">Dashboard</button>
        <button class="delete-btn" data-name="${r.name}" data-kind="${r.kind}">Delete</button>
      </td>`;
    tbody.appendChild(tr);
  }

  // Keep the Evaluate/Dream run pickers in sync with known Dreamer runs.
  const dreamerNames = runs.filter(r => r.kind === "dreamer").map(r => r.name);
  for (const sel of document.querySelectorAll(".dreamer-run-select")) {
    const current = sel.value;
    sel.innerHTML = dreamerNames.map(n => `<option value="${n}">${n}</option>`).join("");
    if (dreamerNames.includes(current)) sel.value = current;
  }

  for (const btn of tbody.querySelectorAll(".delete-btn")) {
    btn.addEventListener("click", onDelete);
  }
  for (const btn of tbody.querySelectorAll(".dashboard-btn")) {
    btn.addEventListener("click", () => compareRuns([{ name: btn.dataset.name, kind: btn.dataset.kind }]));
  }
}

async function onDelete(e) {
  const { name, kind } = e.target.dataset;
  if (!confirm(`Delete run "${name}" (${kind})? This cannot be undone.`)) return;
  const result = await fetchJSON("/api/runs/delete", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, kind }),
  });
  if (!result.ok) alert("Delete failed:\n" + result.output);
  refreshRuns();
}

async function compareRuns(entries) {
  const status = document.getElementById("compare-status");
  status.textContent = "launching TensorBoard…";
  try {
    const job = await fetchJSON("/api/compare", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ runs: entries }),
    });
    status.textContent = `opened ${job.url}`;
    window.open(job.url, "_blank");
    refreshDashboards();
  } catch (err) {
    status.textContent = "failed: " + err;
  }
}

document.getElementById("compare-btn").addEventListener("click", () => {
  const checked = [...document.querySelectorAll(".compare-check:checked")]
    .map(c => ({ name: c.dataset.name, kind: c.dataset.kind }));
  if (checked.length < 2) {
    alert("Pick at least 2 runs to compare (checkboxes in the Runs table).");
    return;
  }
  compareRuns(checked);
});

// ------------------------------------------------------- train/ppo jobs
// Scoped to train/ppo only -- evaluate/dream get their own inline history
// (below), and dashboards get their own inline list under Compare.
function renderJobs(jobs) {
  const container = document.getElementById("jobs-list");
  const trainJobs = jobs.filter(j => j.kind === "train" || j.kind === "ppo");
  container.innerHTML = "";
  if (trainJobs.length === 0) {
    container.innerHTML = '<p class="empty-note">No training jobs yet.</p>';
    return;
  }
  for (const j of trainJobs) {
    const div = document.createElement("div");
    div.className = "job";
    const statusBadge = j.alive
      ? '<span class="badge badge-running">running</span>'
      : '<span class="badge badge-idle">finished</span>';
    div.innerHTML = `
      <div class="job-head">
        <strong>${j.kind}</strong> &mdash; ${j.name} ${statusBadge}
        ${j.alive ? `<button class="stop-btn" data-id="${j.job_id}">Stop</button>` : ""}
        <button class="log-toggle" data-id="${j.job_id}">Toggle log</button>
      </div>
      <code class="job-cmd">${j.cmd.join(" ")}</code>
      <pre class="job-log" id="log-${j.job_id}" hidden></pre>`;
    container.appendChild(div);
  }
  for (const btn of container.querySelectorAll(".stop-btn")) {
    btn.addEventListener("click", async () => {
      await fetchJSON(`/api/jobs/${btn.dataset.id}/stop`, { method: "POST" });
      refreshJobs();
    });
  }
  for (const btn of container.querySelectorAll(".log-toggle")) {
    btn.addEventListener("click", async () => {
      const pre = document.getElementById(`log-${btn.dataset.id}`);
      pre.hidden = !pre.hidden;
      if (!pre.hidden) {
        const res = await fetch(`/api/jobs/${btn.dataset.id}/log?tail=200`);
        pre.textContent = await res.text();
      }
    });
  }
}

// ------------------------------------------------------- eval/dream history
// Flat across every run name (not just whichever run the form's dropdown
// currently points at) -- see docs/webui.md. Each entry is either backed by
// a job this GUI launched (alive/finished, has cmd + Stop) or a disk-only
// "orphan" (no matching job record -- produced by a plain terminal
// evaluate.py/dream.py invocation, or its registry entry was cleared).
function artifactStatusBadge(a) {
  if (a.alive) return '<span class="badge badge-running">running</span>';
  if (!a.job_id) return '<span class="badge badge-idle">finished (started outside the GUI)</span>';
  return '<span class="badge badge-idle">finished</span>';
}

function renderArtifactHistory(containerId, jobKind, entries) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  if (entries.length === 0) {
    container.innerHTML = '<p class="empty-note">No jobs yet.</p>';
    return;
  }
  for (const a of entries) {
    const div = document.createElement("div");
    div.className = "job";
    const paramsHtml = a.cmd
      ? `<code class="job-cmd">${a.cmd.join(" ")}</code>`
      : '<p class="hint">(started outside the GUI -- no recorded command line)</p>';
    const logId = `alog-${jobKind}-${a.name}-${a.filename || a.job_id}`.replace(/[^\w-]/g, "_");
    const videoOk = !a.alive && a.filename && a.size_mb !== null;
    const videoHtml = videoOk
      ? `<video controls preload="metadata" src="/files/dreamer/${a.name}/${a.filename}"></video>`
      : "";
    div.innerHTML = `
      <div class="job-head">
        <strong>${a.name}</strong> ${artifactStatusBadge(a)}
        ${a.alive && a.job_id ? `<button class="stop-btn" data-id="${a.job_id}">Stop</button>` : ""}
        ${!a.alive ? `<button class="delete-artifact-btn">Delete</button>` : ""}
        ${a.has_log ? `<button class="artifact-log-toggle" data-log-id="${logId}">Toggle log</button>` : ""}
      </div>
      ${paramsHtml}
      <pre class="job-log" id="${logId}" hidden></pre>
      ${videoHtml}`;
    container.appendChild(div);

    const stopBtn = div.querySelector(".stop-btn");
    if (stopBtn) {
      stopBtn.addEventListener("click", async () => {
        await fetchJSON(`/api/jobs/${a.job_id}/stop`, { method: "POST" });
        refreshArtifacts(jobKind);
      });
    }

    const deleteBtn = div.querySelector(".delete-artifact-btn");
    if (deleteBtn) {
      deleteBtn.addEventListener("click", async () => {
        if (!a.filename) {
          alert("Nothing to delete -- this job never wrote a video.");
          return;
        }
        if (!confirm(`Delete ${a.filename} (and its log, if any)? This cannot be undone.`)) return;
        await fetchJSON("/api/artifacts/delete", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: a.name, job_kind: jobKind, filename: a.filename }),
        });
        refreshArtifacts(jobKind);
      });
    }

    const logToggle = div.querySelector(".artifact-log-toggle");
    if (logToggle) {
      logToggle.addEventListener("click", async () => {
        const pre = document.getElementById(logId);
        pre.hidden = !pre.hidden;
        if (!pre.hidden) {
          // Job-tracked entries: read via the job-id log endpoint (works
          // whether or not a video was requested). Orphans have no job_id,
          // so fall back to reading the same-stem .log file straight off
          // disk through the existing /files/ route.
          const url = a.job_id
            ? `/api/jobs/${a.job_id}/log?tail=200`
            : `/files/dreamer/${a.name}/${a.filename.replace(/\.mp4$/, ".log")}`;
          const res = await fetch(url);
          pre.textContent = res.ok ? await res.text() : "(log unavailable)";
        }
      });
    }
  }
}

async function refreshArtifacts(jobKind) {
  const containerId = jobKind === "evaluate" ? "evaluate-history" : "dream-history";
  try {
    renderArtifactHistory(containerId, jobKind, await fetchJSON(`/api/artifacts?job_kind=${jobKind}`));
  } catch (e) { console.error(e); }
}

// ------------------------------------------------------------- dashboards
function renderDashboards(jobs) {
  const container = document.getElementById("dashboards-list");
  const alive = jobs.filter(j => j.kind === "dashboard" && j.alive);
  container.innerHTML = "";
  if (alive.length === 0) return;
  for (const j of alive) {
    const div = document.createElement("div");
    div.className = "job";
    div.innerHTML = `
      <div class="job-head">
        <strong>${j.name}</strong> <span class="badge badge-running">running</span>
        <a href="${j.url}" target="_blank">${j.url}</a>
        <button class="stop-btn" data-id="${j.job_id}">Stop</button>
      </div>`;
    container.appendChild(div);
  }
  for (const btn of container.querySelectorAll(".stop-btn")) {
    btn.addEventListener("click", async () => {
      await fetchJSON(`/api/jobs/${btn.dataset.id}/stop`, { method: "POST" });
      refreshJobs();
    });
  }
}

async function refreshRuns() {
  try { renderRuns(await fetchJSON("/api/runs")); } catch (e) { console.error(e); }
}
let lastJobs = [];
async function refreshJobs() {
  try {
    lastJobs = await fetchJSON("/api/jobs");
    renderJobs(lastJobs);
    renderDashboards(lastJobs);
  } catch (e) { console.error(e); }
}
function refreshDashboards() { renderDashboards(lastJobs); }

function bindStartForm(formId, endpoint, extraFields, onDone) {
  document.getElementById(formId).addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const data = { name: form.name.value.trim() };
    for (const f of extraFields) {
      const el = form.elements[f];
      if (!el) continue;
      data[f] = el.type === "checkbox" ? el.checked : el.value;
    }
    submitBtn.disabled = true;
    try {
      await fetchJSON(endpoint, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (onDone) onDone();
    } finally {
      submitBtn.disabled = false;
    }
  });
}

bindStartForm("train-form", "/api/jobs/train",
  ["total_steps", "entropy_coef", "sparse_reward", "device", "set_text"], refreshJobs);
bindStartForm("ppo-form", "/api/jobs/ppo",
  ["total_steps", "ent_coef", "sparse_reward", "device", "set_text"], refreshJobs);
bindStartForm("evaluate-form", "/api/jobs/evaluate",
  ["episodes", "video", "set_text"], () => refreshArtifacts("evaluate"));
bindStartForm("dream-form", "/api/jobs/dream",
  ["context", "horizon", "upscale", "fps", "set_text"], () => refreshArtifacts("dream"));

refreshRuns();
refreshJobs();
refreshArtifacts("evaluate");
refreshArtifacts("dream");
setInterval(refreshRuns, POLL_MS);
setInterval(refreshJobs, POLL_MS);
setInterval(() => refreshArtifacts("evaluate"), POLL_MS);
setInterval(() => refreshArtifacts("dream"), POLL_MS);
