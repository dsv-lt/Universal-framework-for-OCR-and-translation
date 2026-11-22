const state = {
  packs: [],
  currentImageBlob: null,
  currentImageObj: null,
  regions: [],
  crops: [],
};

function $(id) {
  return document.getElementById(id);
}

function showBanner(el, msg, kind = "info") {
  el.textContent = msg;
  el.className = `status-banner ${kind}`;
  el.classList.remove("hidden");
}

function hideBanner(el) {
  el.classList.add("hidden");
}

// Tabs
document.querySelectorAll("#main-tabs .tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#main-tabs .tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// Detector parameter switcher (CRAFT vs DBNet++)
$("rec-detector").addEventListener("change", (e) => {
  const isCraft = e.target.value === "craft";
  $("craft-params").classList.toggle("hidden", !isCraft);
  $("dbnet-params").classList.toggle("hidden", isCraft);
});

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    $("health-pill").textContent = `Online v${data.version}`;
  } catch {
    $("health-pill").textContent = "Offline";
  }
}

async function refreshCapabilities() {
  const res = await fetch("/api/capabilities");
  const data = await res.json();
  $("capabilities-box").textContent = JSON.stringify(data, null, 2);
}

async function refreshPacks() {
  const res = await fetch("/api/packs");
  const data = await res.json();
  state.packs = data.packs || [];

  const recSelect = $("rec-pack");
  const synthSelect = $("synth-pack");
  const prevRec = recSelect.value;
  const prevSynth = synthSelect.value;
  recSelect.innerHTML = "";
  synthSelect.innerHTML = "";

  const listEl = $("packs-list");
  listEl.innerHTML = "";

  state.packs.forEach((p) => {
    const opt1 = document.createElement("option");
    opt1.value = p.id;
    opt1.textContent = `${p.name} (${p.id})`;
    recSelect.appendChild(opt1);

    const opt2 = opt1.cloneNode(true);
    synthSelect.appendChild(opt2);

    const card = document.createElement("div");
    card.className = "pack-card";
    card.innerHTML = `
      <div>
        <div>
          <span class="badge ${p.valid ? "valid" : "invalid"}">${p.valid ? "VALID" : "INVALID"}</span>
          <strong>${p.name}</strong> <code>${p.id}</code>
        </div>
        <div class="muted mt-12" style="margin-top:4px;">
          class: <code>${p.language_class || "?"}</code> ·
          renderer: <code>${p.renderer || "?"}</code> ·
          detector: <code>${p.default_detector || "?"}</code> ·
          postprocess: <code>${p.postprocess_enabled ? "on" : "off"}</code>
        </div>
      </div>
      <div class="row-actions" style="margin-top:0;">
        <button class="btn btn-secondary btn-inspect" data-id="${p.id}">Validate / Inspect</button>
      </div>
    `;
    listEl.appendChild(card);
  });

  if (prevRec && state.packs.some((p) => p.id === prevRec)) recSelect.value = prevRec;
  if (prevSynth && state.packs.some((p) => p.id === prevSynth)) synthSelect.value = prevSynth;

  document.querySelectorAll(".btn-inspect").forEach((btn) => {
    btn.addEventListener("click", () => inspectPack(btn.dataset.id));
  });

  syncPackControls();
}

function syncPackControls() {
  const selectedId = $("rec-pack").value;
  const pack = state.packs.find((p) => p.id === selectedId);
  const toggle = $("rec-postprocess");
  const row = $("postprocess-toggle-row");
  const hint = $("postprocess-hint");

  if (!pack) return;
  $("rec-detector").value = pack.default_detector || "craft";
  $("rec-detector").dispatchEvent(new Event("change"));
  $("rec-granularity").value = pack.granularity || "line";
  $("rec-backend").value = pack.default_backend || "onnx_ocr";

  if (pack.has_postprocess_module) {
    toggle.disabled = false;
    toggle.checked = Boolean(pack.postprocess_enabled);
    row.classList.remove("disabled");
    hint.textContent = "Промежуточный слой доступен (можно включать или отключать для отладки).";
  } else {
    toggle.disabled = true;
    toggle.checked = false;
    row.classList.add("disabled");
    hint.textContent = "В этом пакете нет postprocess.py — распознанный текст пойдёт напрямую в переводчик.";
  }
}

$("rec-pack").addEventListener("change", syncPackControls);

async function inspectPack(packId) {
  const res = await fetch(`/api/packs/${encodeURIComponent(packId)}`);
  const data = await res.json();
  const container = $("pack-validation-details");
  const val = data.validation;
  const rows = (val.issues || [])
    .map(
      (i) =>
        `<tr><td><code>${i.level.toUpperCase()}</code></td><td><code>${i.file}</code></td><td>${i.message}</td></tr>`
    )
    .join("");
  container.innerHTML = `
    <div>
      <span class="badge ${val.valid ? "valid" : "invalid"}">${val.valid ? "VALID" : "INVALID"}</span>
      <strong>${val.pack_id}</strong> · Glossary entries: <strong>${data.glossary_count ?? 0}</strong>
    </div>
    ${rows ? `<table class="issue-table"><thead><tr><th>Level</th><th>File</th><th>Message</th></tr></thead><tbody>${rows}</tbody></table>` : "<p class='muted'>Нет замечаний.</p>"}
    <pre class="code-box mt-12">${JSON.stringify(data.manifest, null, 2)}</pre>
  `;
}

// Image loading & interactive canvas
function loadImageBlobIntoCanvas(blob) {
  state.currentImageBlob = blob;
  state.regions = [];
  state.crops = [];
  $("crops-strip").innerHTML = "";
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    state.currentImageObj = img;
    drawCanvas();
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

$("rec-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) loadImageBlobIntoCanvas(file);
});

$("btn-load-demo-img").addEventListener("click", async () => {
  const packId = $("rec-pack").value || "_example_demo";
  const res = await fetch("/api/synth/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pack_id: packId,
      text: "uryyb jbeyq",
      augmentation: {
        background: "parchment",
        rotation_deg: 6,
        perspective_strength: 0.06,
        strikethrough_prob: 0.0,
        occlusion_prob: 0.0,
      },
    }),
  });
  if (!res.ok) {
    const err = await res.json();
    showBanner($("rec-status"), err.detail || "Failed to generate demo image", "error");
    return;
  }
  const blob = await res.blob();
  loadImageBlobIntoCanvas(blob);
  showBanner($("rec-status"), `Тестовое изображение для "${packId}" загружено.`, "ok");
});

function drawCanvas() {
  const canvas = $("detect-canvas");
  const ctx = canvas.getContext("2d");
  const img = state.currentImageObj;
  if (!img) {
    canvas.width = 480;
    canvas.height = 180;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  state.regions.forEach((reg) => {
    const pts = reg.polygon;
    if (!pts || pts.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = reg.enabled ? "#38bdf8" : "#64748b";
    ctx.fillStyle = reg.enabled ? "rgba(56, 189, 248, 0.14)" : "rgba(100, 116, 139, 0.12)";
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = reg.enabled ? "#38bdf8" : "#94a3b8";
    ctx.font = "bold 14px monospace";
    ctx.fillText(`#${reg.id}`, pts[0][0] + 4, Math.max(16, pts[0][1] - 4));
  });
}

function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-6) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

$("detect-canvas").addEventListener("click", (e) => {
  if (!state.regions.length || !state.currentImageObj) return;
  const rect = e.target.getBoundingClientRect();
  const scaleX = e.target.width / rect.width;
  const scaleY = e.target.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  let changed = false;
  state.regions.forEach((reg) => {
    if (pointInPoly(x, y, reg.polygon)) {
      reg.enabled = !reg.enabled;
      changed = true;
    }
  });
  if (changed) drawCanvas();
});

function renderCrops(crops) {
  const strip = $("crops-strip");
  strip.innerHTML = "";
  crops.forEach((b64, idx) => {
    const div = document.createElement("div");
    div.className = "crop-item";
    div.innerHTML = `<img src="data:image/png;base64,${b64}" alt="crop ${idx}" /><div>Region #${idx}</div>`;
    strip.appendChild(div);
  });
}

$("btn-detect").addEventListener("click", async () => {
  if (!state.currentImageBlob) {
    showBanner($("rec-status"), "Сначала выберите или сгенерируйте изображение.", "error");
    return;
  }
  hideBanner($("rec-status"));
  const fd = new FormData();
  fd.append("file", state.currentImageBlob, "input.png");
  fd.append("pack_id", $("rec-pack").value || "_example_demo");
  fd.append("detector", $("rec-detector").value);
  fd.append("granularity", $("rec-granularity").value);
  fd.append("text_threshold", $("det-text-thresh").value);
  fd.append("link_threshold", $("det-link-thresh").value);
  fd.append("low_text", $("det-low-text").value);
  fd.append("db_thresh", $("det-db-thresh").value);
  fd.append("db_box_thresh", $("det-db-box-thresh").value);
  fd.append("db_unclip_ratio", $("det-db-unclip").value);
  fd.append("demo_contour_fallback", $("rec-fallback").checked ? "true" : "false");

  const res = await fetch("/api/detect", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) {
    showBanner($("rec-status"), data.detail || "Detection failed", "error");
    return;
  }
  state.regions = data.regions || [];
  state.crops = data.crops || [];
  drawCanvas();
  renderCrops(state.crops);
  const modeNote = data.used_fallback ? " (использован contour fallback без ONNX-весов)" : "";
  showBanner($("rec-status"), `Найдено регионов: ${state.regions.length}${modeNote}.`, "ok");
});

$("btn-run-pipeline").addEventListener("click", async () => {
  if (!state.currentImageBlob) {
    showBanner($("rec-status"), "Сначала выберите или сгенерируйте изображение.", "error");
    return;
  }
  const fd = new FormData();
  fd.append("file", state.currentImageBlob, "input.png");
  fd.append("pack_id", $("rec-pack").value);
  fd.append("detector", $("rec-detector").value);
  fd.append("backend", $("rec-backend").value);
  fd.append("use_postprocess", $("rec-postprocess").checked ? "true" : "false");
  fd.append("target_lang", $("rec-target-lang").value);
  fd.append("skip_translation", $("rec-skip-trans").checked ? "true" : "false");
  fd.append("demo_contour_fallback", $("rec-fallback").checked ? "true" : "false");
  if (state.regions.length) {
    fd.append("regions_json", JSON.stringify(state.regions));
  }

  const res = await fetch("/api/pipeline/run", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) {
    showBanner($("rec-status"), data.detail || "Pipeline error", "error");
    return;
  }
  $("out-raw").value = data.raw_full_text || "";
  $("out-post").value = data.postprocessed_full_text || "";
  $("out-en").value = data.en_text || "";
  $("out-target").value = data.target_text || "";
  const hits = (data.glossary_hits || []).map((g) => g.term).join(", ");
  const warns = (data.warnings || []).join(" | ");
  $("out-meta").textContent = `Postprocess applied: ${data.postprocess_applied} | Glossary hits: [${hits}] ${warns ? " | Warnings: " + warns : ""}`;
  showBanner($("rec-status"), "Пайплайн выполнен.", "ok");
});

// Pack creation
$("create-pack-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    pack_id: $("new-pack-id").value.trim(),
    name: $("new-pack-name").value.trim(),
    language_class: $("new-pack-class").value,
    renderer: $("new-pack-renderer").value,
    direction: $("new-pack-direction").value,
    enable_postprocess: $("new-pack-postprocess").checked,
  };
  const res = await fetch("/api/packs/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    showBanner($("create-pack-result"), data.detail || "Error creating pack", "error");
    return;
  }
  showBanner($("create-pack-result"), `Пакет создан в ${data.pack_dir}`, "ok");
  await refreshPacks();
  inspectPack(payload.pack_id);
});

$("btn-refresh-packs").addEventListener("click", refreshPacks);

// Synth preview & sliders
[
  ["aug-rot", "val-rot"],
  ["aug-persp", "val-persp"],
  ["aug-shear", "val-shear"],
  ["aug-elastic", "val-elastic"],
  ["aug-blur", "val-blur"],
  ["aug-strike", "val-strike"],
  ["aug-occ", "val-occ"],
  ["aug-light", "val-light"],
].forEach(([sliderId, labelId]) => {
  $(sliderId).addEventListener("input", () => {
    $(labelId).textContent = $(sliderId).value;
  });
});

async function updateSynthPreview() {
  hideBanner($("synth-error"));
  const rendererVal = $("synth-renderer").value || null;
  const payload = {
    pack_id: $("synth-pack").value || "_example_demo",
    text: $("synth-text").value || "hello",
    renderer_override: rendererVal,
    seed: Number($("synth-seed").value || 42),
    augmentation: {
      background: $("synth-bg").value,
      rotation_deg: Number($("aug-rot").value),
      perspective_strength: Number($("aug-persp").value),
      shear_deg: Number($("aug-shear").value),
      elastic_alpha: Number($("aug-elastic").value),
      blur_sigma: Number($("aug-blur").value),
      strikethrough_prob: Number($("aug-strike").value),
      occlusion_prob: Number($("aug-occ").value),
      lighting_variation: Number($("aug-light").value),
    },
  };
  const res = await fetch("/api/synth/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    showBanner($("synth-error"), err.detail || "Render error", "error");
    return;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  $("synth-preview-img").src = url;
}

$("btn-synth-preview").addEventListener("click", updateSynthPreview);

$("btn-train-plan").addEventListener("click", async () => {
  const payload = {
    mode: $("train-mode").value,
    pack_id: $("synth-pack").value || "_example_demo",
    dataset_path: $("train-dataset-path").value,
    output_dir: $("train-output-dir").value,
    epochs: Number($("train-epochs").value || 10),
  };
  const res = await fetch("/api/train/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  $("train-plan-out").textContent = JSON.stringify(data, null, 2);
});

// Settings
async function loadSettingsUI() {
  const res = await fetch("/api/settings");
  const s = await res.json();
  $("set-packs-dir").value = s.packs_dir;
  $("set-models-dir").value = s.models_dir;
  $("set-datasets-dir").value = s.datasets_dir;
  $("set-device").value = s.device;
  $("set-target-lang").value = s.default_target_lang;
  $("set-llm-url").value = s.llm_base_url;
  $("set-llm-model").value = s.llm_model;
  $("set-llm-key").value = s.llm_api_key;
}

$("settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    packs_dir: $("set-packs-dir").value,
    models_dir: $("set-models-dir").value,
    datasets_dir: $("set-datasets-dir").value,
    device: $("set-device").value,
    default_target_lang: $("set-target-lang").value,
    llm_base_url: $("set-llm-url").value,
    llm_model: $("set-llm-model").value,
    llm_api_key: $("set-llm-key").value,
  };
  const res = await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  showBanner($("settings-status"), `Сохранено в ${data.saved_to}`, "ok");
  refreshPacks();
  refreshCapabilities();
});

$("btn-test-llm").addEventListener("click", async () => {
  const res = await fetch("/api/settings/test-llm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      base_url: $("set-llm-url").value,
      api_key: $("set-llm-key").value,
      model: $("set-llm-model").value,
    }),
  });
  const data = await res.json();
  if (data.ok) {
    showBanner($("settings-status"), `Connected! Models: ${(data.models || []).join(", ") || "ready"}`, "ok");
  } else {
    showBanner($("settings-status"), `LLM unreachable (${data.url}): ${data.error}`, "error");
  }
});

// Init
(async () => {
  await refreshHealth();
  await refreshPacks();
  await loadSettingsUI();
  await refreshCapabilities();
  if (state.packs.length) {
    updateSynthPreview();
  }
})();
