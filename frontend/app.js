const API = window.APP_CONFIG.API_BASE_URL.replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
const COLORS = { parasitized: "#b8325e", uninfected: "#217a68", violet: "#5a3e8e", grey: "#9b96ae" };
const charts = {};
let lastStatus = null;

async function api(path, options) {
  const res = await fetch(API + path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

function setMsg(id, text, kind = "") {
  const el = $(id);
  el.textContent = text;
  el.className = "msg " + kind;
}

function formatUptime(s) {
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : m ? `${m}m ${Math.floor(s % 60)}s` : `${Math.floor(s)}s`;
}

document.querySelectorAll("nav button").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button, .tab").forEach((el) => el.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "dashboard") loadDashboard();
  })
);

async function pollStatus() {
  try {
    lastStatus = await api("/status");
    $("pill").className = "pill";
    $("pill-text").textContent = `Online, model v${lastStatus.model_version ?? "none"}`;
    renderKpis(lastStatus);
    renderRetrain(lastStatus);
  } catch {
    lastStatus = null;
    $("pill").className = "pill off";
    $("pill-text").textContent = "API unreachable";
    $("k-api").textContent = "Down";
  }
}

function renderKpis(s) {
  $("k-api").textContent = "Up";
  $("k-uptime").textContent = formatUptime(s.uptime_seconds);
  $("k-version").textContent = s.model_version ? `v${s.model_version}` : "none";
  $("k-preds").textContent = s.counters.predictions;
  const l = s.latency_ms;
  $("k-lat").textContent = l.mean == null ? "-" : `${l.mean} / ${l.p95} ms`;
}

$("file").addEventListener("change", () => {
  const f = $("file").files[0];
  $("predict-btn").disabled = !f;
  $("result").hidden = true;
  $("lens").className = "lens";
  if (!f) return;
  $("preview").src = URL.createObjectURL(f);
  $("preview").style.display = "block";
});

$("predict-btn").addEventListener("click", async () => {
  const form = new FormData();
  form.append("file", $("file").files[0]);
  $("predict-btn").disabled = true;
  setMsg("predict-msg", "Classifying...");
  try {
    const r = await api("/predict", { method: "POST", body: form });
    const label = r.label;
    $("lens").className = "lens " + label;
    $("verdict").textContent = label === "parasitized" ? "Parasitized" : "Uninfected";
    $("verdict").className = "verdict " + label;
    $("detail").textContent = `${(r.confidence * 100).toFixed(1)}% confidence, ${r.latency_ms} ms, model v${r.model_version}`;
    $("bar-fill").style.width = `${r.probability_parasitized * 100}%`;
    $("result").hidden = false;
    setMsg("predict-msg", "");
  } catch (e) {
    setMsg("predict-msg", e.message, "error");
  }
  $("predict-btn").disabled = false;
});

$("files").addEventListener("change", () => {
  const n = $("files").files.length;
  $("files-text").textContent = n ? `${n} image${n > 1 ? "s" : ""} selected` : "Choose one or more images";
  $("upload-btn").disabled = !n;
});

$("upload-btn").addEventListener("click", async () => {
  const form = new FormData();
  form.append("label", document.querySelector('input[name="label"]:checked').value);
  [...$("files").files].forEach((f) => form.append("files", f));
  $("upload-btn").disabled = true;
  setMsg("upload-msg", "Uploading...");
  try {
    const r = await api("/upload", { method: "POST", body: form });
    const rejected = r.rejected.length ? `, ${r.rejected.length} rejected as unreadable` : "";
    setMsg("upload-msg", `${r.saved} images saved${rejected}.${r.auto_retrain_started ? " Retraining started automatically." : ""}`, "ok");
    $("files").value = "";
    $("files-text").textContent = "Choose one or more images";
    pollStatus();
  } catch (e) {
    setMsg("upload-msg", e.message, "error");
    $("upload-btn").disabled = false;
  }
});

$("retrain-btn").addEventListener("click", async () => {
  const body = {
    epochs: +$("p-epochs").value,
    lr: +$("p-lr").value,
    unfreeze_last: +$("p-unfreeze").value,
  };
  try {
    await api("/retrain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    setMsg("retrain-msg", "Retraining started.", "ok");
    pollStatus();
  } catch (e) {
    setMsg("retrain-msg", e.message, "error");
  }
});

function renderRetrain(s) {
  const pending = Object.values(s.pending_uploads).reduce((a, b) => a + b, 0);
  $("pending").textContent = pending;
  $("threshold").textContent = s.retrain_threshold;
  $("pending-fill").style.width = `${Math.min(100, (pending / s.retrain_threshold) * 100)}%`;
  const job = s.retrain;
  $("retrain-btn").disabled = job.status === "running";
  if (job.status === "running") {
    setMsg("retrain-msg", `Retraining (${job.trigger}) since ${job.started_at}. This can take several minutes.`);
  } else if (job.status === "done") {
    const r = job.result;
    const f1 = (m) => (m.f1 * 100).toFixed(2) + "%";
    setMsg(
      "retrain-msg",
      r.promoted
        ? `Version ${r.version} is now live. Test F1 ${f1(r.previous_metrics)} to ${f1(r.new_metrics)}, trained on ${r.n_new_images} new images.`
        : `Version ${r.version} scored worse on the test set (F1 ${f1(r.new_metrics)} vs ${f1(r.previous_metrics)}), so version ${r.previous_version} stays live.`,
      r.promoted ? "ok" : ""
    );
  } else if (job.status === "failed") {
    setMsg("retrain-msg", `Retraining failed: ${job.error}`, "error");
  }
}

function draw(id, config) {
  charts[id]?.destroy();
  charts[id] = new Chart($(id), config);
}

async function loadDashboard() {
  const [summary, features, metrics] = await Promise.all([
    api("/data/summary").catch(() => null),
    api("/data/features").catch(() => null),
    api("/metrics").catch(() => null),
  ]);
  if (summary) {
    const classes = Object.keys(summary.train);
    draw("c-classes", {
      type: "bar",
      data: {
        labels: ["train", "test", "uploads"],
        datasets: classes.map((c) => ({
          label: c,
          data: ["train", "test", "uploads"].map((k) => summary[k][c]),
          backgroundColor: COLORS[c],
        })),
      },
    });
  }
  if (features && Object.keys(features.classes).length) {
    const names = Object.keys(features.classes);
    draw("c-rgb", {
      type: "bar",
      data: {
        labels: ["Red", "Green", "Blue"],
        datasets: names.map((c) => ({ label: c, data: features.classes[c].mean_rgb, backgroundColor: COLORS[c] })),
      },
    });
    const bins = features.stain_bins.slice(0, -1).map((b, i) => `${Math.round(b * 100)}-${Math.round(features.stain_bins[i + 1] * 100)}%`);
    draw("c-stain", {
      type: "line",
      data: {
        labels: bins,
        datasets: names.map((c) => ({ label: c, data: features.classes[c].stain_hist, borderColor: COLORS[c], tension: 0.3 })),
      },
    });
  }
  if (metrics && metrics.current) renderModel(metrics);
}

function renderModel(metrics) {
  const cur = metrics.versions.find((v) => v.version === metrics.current);
  const m = cur.metrics;
  draw("c-metrics", {
    type: "bar",
    data: {
      labels: ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
      datasets: [{ data: [m.accuracy, m.precision, m.recall, m.f1, m.roc_auc], backgroundColor: COLORS.violet }],
    },
    options: { plugins: { legend: { display: false } }, scales: { y: { min: Math.max(0, Math.min(m.accuracy, m.precision, m.recall, m.f1) - 0.05), max: 1 } } },
  });
  const [[tn, fp], [fn, tp]] = m.confusion_matrix;
  const cell = (v, good) => `<div style="background:${good ? "#dcefe9" : "#f6dbe3"}">${v}</div>`;
  $("cm").innerHTML =
    `<div></div><div class="head">uninfected</div><div class="head">parasitized</div>` +
    `<div class="head">uninfected</div>${cell(tn, true)}${cell(fp, false)}` +
    `<div class="head">parasitized</div>${cell(fn, false)}${cell(tp, true)}`;
  $("versions").innerHTML =
    "<table><tr><th>Version</th><th>Trigger</th><th>F1</th><th>Recall</th><th>Status</th></tr>" +
    metrics.versions
      .slice()
      .reverse()
      .map((v) => `<tr><td>v${v.version}</td><td>${v.trigger}</td><td>${(v.metrics.f1 * 100).toFixed(2)}%</td><td>${(v.metrics.recall * 100).toFixed(2)}%</td><td>${v.version === metrics.current ? "live" : v.promoted ? "replaced" : "rejected"}</td></tr>`)
      .join("") +
    "</table>";
}

pollStatus();
setInterval(pollStatus, 5000);
