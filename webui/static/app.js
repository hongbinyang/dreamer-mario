// Thin client: polls /api/runs and /api/jobs, submits forms to the
// corresponding POST endpoints. No framework, no build step.

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

function videoUrlForJob(j) {
  // Derived from the job's own cmd array (--video for evaluate, --out for
  // dream) rather than a separate stored field -- the cmd is already the
  // source of truth shown in the UI for transparency, so this stays in
  // sync automatically instead of needing its own persisted copy.
  if (j.kind !== "evaluate" && j.kind !== "dream") return null;
  const flag = j.kind === "evaluate" ? "--video" : "--out";
  const idx = j.cmd.indexOf(flag);
  if (idx === -1 || idx + 1 >= j.cmd.length) return null;
  const filename = j.cmd[idx + 1].split("/").pop(); // "runs/trial/eval_x.mp4" -> "eval_x.mp4"
  return `/files/dreamer/${j.name}/${filename}`;
}

function renderJobs(jobs) {
  const container = document.getElementById("jobs-list");
  container.innerHTML = "";
  if (jobs.length === 0) {
    container.innerHTML = '<p class="empty-note">No jobs yet.</p>';
    return;
  }
  for (const j of jobs) {
    const div = document.createElement("div");
    div.className = "job";
    const statusBadge = j.alive
      ? '<span class="badge badge-running">running</span>'
      : '<span class="badge badge-idle">finished</span>';
    const videoUrl = !j.alive ? videoUrlForJob(j) : null;
    div.innerHTML = `
      <div class="job-head">
        <strong>${j.kind}</strong> &mdash; ${j.name} ${statusBadge}
        ${j.alive ? `<button class="stop-btn" data-id="${j.job_id}">Stop</button>` : ""}
        <button class="log-toggle" data-id="${j.job_id}">Toggle log</button>
      </div>
      <code class="job-cmd">${j.cmd.join(" ")}</code>
      <pre class="job-log" id="log-${j.job_id}" hidden></pre>
      ${videoUrl ? `<video controls preload="metadata" src="${videoUrl}"></video>` : ""}`;
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

async function refreshRuns() {
  try { renderRuns(await fetchJSON("/api/runs")); } catch (e) { console.error(e); }
}
async function refreshJobs() {
  try { renderJobs(await fetchJSON("/api/jobs")); } catch (e) { console.error(e); }
}

function bindStartForm(formId, endpoint, extraFields) {
  document.getElementById(formId).addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const data = { name: form.name.value.trim() };
    for (const f of extraFields) {
      const el = form.elements[f];
      if (!el) continue;
      data[f] = el.type === "checkbox" ? el.checked : el.value;
    }
    await fetchJSON(endpoint, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    refreshJobs();
  });
}

bindStartForm("train-form", "/api/jobs/train",
  ["total_steps", "entropy_coef", "sparse_reward", "device", "set_text"]);
bindStartForm("ppo-form", "/api/jobs/ppo",
  ["total_steps", "ent_coef", "sparse_reward", "device", "set_text"]);
bindStartForm("evaluate-form", "/api/jobs/evaluate",
  ["episodes", "video", "set_text"]);
bindStartForm("dream-form", "/api/jobs/dream",
  ["context", "horizon", "upscale", "fps", "set_text"]);

refreshRuns();
refreshJobs();
setInterval(refreshRuns, POLL_MS);
setInterval(refreshJobs, POLL_MS);
