import { createTokenizer, loadBundle } from "./format.js";
import { NanoGPTTracer, sliceTensor, tensorStats } from "./runtime.js";

const state = {
  bundle: null,
  tokenizer: null,
  tracer: null,
  trace: null,
  selectedNodeId: null,
  modalPlot: null,
  headTensorVisibility: null,
  headTensorOptions: [],
};

const STORAGE_KEY = "nanogpt-signal-lab-ui";

const elements = {
  bundleInput: document.querySelector("#bundle-input"),
  dropzone: document.querySelector("#dropzone"),
  modelSummary: document.querySelector("#model-summary"),
  promptInput: document.querySelector("#prompt-input"),
  temperatureInput: document.querySelector("#temperature-input"),
  topkInput: document.querySelector("#topk-input"),
  strategyInput: document.querySelector("#strategy-input"),
  seedInput: document.querySelector("#seed-input"),
  stepButton: document.querySelector("#step-button"),
  resetButton: document.querySelector("#reset-button"),
  runSummary: document.querySelector("#run-summary"),
  diagram: document.querySelector("#diagram"),
  inspectorContent: document.querySelector("#inspector-content"),
  inspectorHint: document.querySelector("#inspector-hint"),
  statusPill: document.querySelector("#status-pill"),
  plotModal: document.querySelector("#plot-modal"),
  plotBackdrop: document.querySelector("#plot-backdrop"),
  plotClose: document.querySelector("#plot-close"),
  plotTitle: document.querySelector("#plot-title"),
  plotMeta: document.querySelector("#plot-meta"),
  plotBody: document.querySelector("#plot-body"),
};

wireEvents();
restoreUiState();
renderInitialState();

function wireEvents() {
  elements.bundleInput.addEventListener("change", async (event) => {
    const [file] = event.target.files ?? [];
    if (file) {
      await handleBundleFile(file);
    }
  });

  ["dragenter", "dragover"].forEach((type) => {
    elements.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      elements.dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((type) => {
    elements.dropzone.addEventListener(type, () => {
      elements.dropzone.classList.remove("dragover");
    });
  });

  elements.dropzone.addEventListener("drop", async (event) => {
    event.preventDefault();
    const [file] = event.dataTransfer?.files ?? [];
    if (file) {
      await handleBundleFile(file);
    }
  });

  elements.stepButton.addEventListener("click", () => runStep());
  elements.resetButton.addEventListener("click", resetTrace);
  [
    elements.promptInput,
    elements.temperatureInput,
    elements.topkInput,
    elements.strategyInput,
    elements.seedInput,
  ].forEach((element) => {
    element.addEventListener("input", persistUiState);
    element.addEventListener("change", persistUiState);
  });
  elements.plotBackdrop.addEventListener("click", closePlotModal);
  elements.plotClose.addEventListener("click", closePlotModal);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !elements.plotModal.classList.contains("hidden")) {
      closePlotModal();
    }
  });
}

async function handleBundleFile(file) {
  setStatus("Loading bundle");
  try {
    state.bundle = await loadBundle(file);
    state.tokenizer = createTokenizer(state.bundle.manifest.tokenizer);
    state.tracer = new NanoGPTTracer(state.bundle, state.tokenizer);
    state.trace = null;
    elements.promptInput.placeholder = state.tokenizer.promptHelp;
    renderModelSummary();
    renderRunSummary(`Loaded ${file.name}. Ready to step.`);
    renderDiagram();
    renderInspector();
    setStatus("Bundle ready");
  } catch (error) {
    setStatus("Load failed");
    renderRunSummary(error.message, true);
  }
}

function runStep() {
  if (!state.tracer) {
    renderRunSummary("Load a converted bundle first.", true);
    return;
  }

  setStatus("Running one step");
  try {
    const trace = state.tracer.step(elements.promptInput.value, {
      temperature: Number(elements.temperatureInput.value),
      topK: Number(elements.topkInput.value),
      strategy: elements.strategyInput.value,
      seed: Number(elements.seedInput.value),
    });
    state.trace = trace;
    const selectedNodeStillExists = trace.nodes.some((node) => node.id === state.selectedNodeId);
    state.selectedNodeId = selectedNodeStillExists ? state.selectedNodeId : trace.nodes.at(-1)?.id ?? null;
    elements.promptInput.value = trace.outputText;
    persistUiState();
    renderRunSummary(buildRunSummary(trace));
    renderDiagram();
    renderInspector();
    setStatus(`Stepped in ${trace.elapsedMs.toFixed(1)} ms`);
  } catch (error) {
    renderRunSummary(error.message, true);
    setStatus("Step failed");
  }
}

function resetTrace() {
  state.trace = null;
  state.selectedNodeId = null;
  persistUiState();
  renderRunSummary(state.bundle ? "Trace cleared. Ready for another step." : "Load a bundle to begin.");
  renderDiagram();
  renderInspector();
  setStatus(state.bundle ? "Bundle ready" : "Idle");
}

function renderInitialState() {
  renderModelSummary();
  renderDiagram();
  renderInspector();
}

function renderModelSummary() {
  if (!state.bundle) {
    elements.modelSummary.innerHTML = `<div class="summary-empty">No bundle loaded</div>`;
    return;
  }

  const model = state.bundle.manifest.model;
  const tokenizer = state.bundle.manifest.tokenizer;
  elements.modelSummary.innerHTML = `
    <div class="badge-row">
      <span class="badge">${model.n_layer} layers</span>
      <span class="badge">${model.n_head} heads</span>
      <span class="badge">${model.n_embd} embd</span>
    </div>
    <div class="tensor-stats" style="margin-top: 6px;">
      <span>Vocab ${formatNumber(model.vocab_size)}</span>
      <span>Block ${model.block_size}</span>
      <span>Params ${formatNumber(model.parameter_count)}</span>
    </div>
    <p class="subtle" style="margin: 6px 0 0;">
      Tokenizer: ${tokenizer ? tokenizer.kind : "token IDs only"}.
      ${tokenizer ? `Source ${tokenizer.meta_path ?? "embedded"}.` : ""}
    </p>
  `;
}

function renderRunSummary(message, isError = false) {
  elements.runSummary.innerHTML = `<span style="color: ${isError ? "var(--danger)" : "var(--muted)"};">${escapeHtml(message)}</span>`;
}

function buildRunSummary(trace) {
  const cropNote = trace.cropped ? " · cropped" : "";
  return `${trace.nextTokenText} [${trace.nextTokenId}] · ${trace.elapsedMs.toFixed(1)} ms${cropNote}`;
}

function renderDiagram() {
  if (!state.trace) {
    elements.diagram.innerHTML = `
      <div class="empty-state">
        <h3>Awaiting execution</h3>
        <p>Load a bundle and run one step to materialize the transformer path.</p>
      </div>
    `;
    return;
  }

  const sectionsMarkup = state.trace.sections
    .map((section) => {
      const nodesMarkup = section.nodes
        .map((node) => {
          const active = node.id === state.selectedNodeId ? "active" : "";
          const metrics = node.metrics
            .map((metric) => `<span class="metric-chip">${escapeHtml(metric)}</span>`)
            .join("");
          return `
            <button class="node-card ${active}" data-node-id="${node.id}" style="--node-accent:${node.accent}; --node-glow:${node.accent};">
              <div class="node-label">
                <span>${escapeHtml(node.label)}</span>
                <span class="section-meta">${node.tensors.length} tensors</span>
              </div>
              <div class="node-subtitle">${escapeHtml(node.subtitle)}</div>
              <div class="node-metrics">${metrics}</div>
            </button>
          `;
        })
        .join("");

      return `
        <section class="diagram-section">
          <div class="section-title">
            <div class="section-label">${escapeHtml(section.title)}</div>
            <div class="section-meta">${escapeHtml(section.meta)}</div>
          </div>
          ${nodesMarkup}
        </section>
      `;
    })
    .join("");

  elements.diagram.innerHTML = sectionsMarkup;
  elements.diagram.querySelectorAll("[data-node-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedNodeId = button.getAttribute("data-node-id");
      persistUiState();
      renderDiagram();
      renderInspector();
    });
  });
}

function renderInspector() {
  if (!state.trace || !state.selectedNodeId) {
    elements.inspectorHint.textContent = "Select a model element";
    elements.inspectorContent.innerHTML = `
      <div class="empty-state">
        <h3>No trace selected</h3>
        <p>Click a node in the diagram to inspect its tensors.</p>
      </div>
    `;
    return;
  }

  const node = state.trace.nodes.find((entry) => entry.id === state.selectedNodeId);
  if (!node) {
    elements.inspectorHint.textContent = "Select a model element";
    return;
  }

  elements.inspectorHint.textContent = node.label;
  const header = document.createElement("div");
  header.className = "inspector-header";
  header.innerHTML = `
    <h3>${escapeHtml(node.label)}</h3>
    <p class="subtle" style="margin: 0;">${escapeHtml(node.subtitle)}</p>
  `;

  const badgeRow = document.createElement("div");
  badgeRow.className = "badge-row";
  for (const metric of node.metrics) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = metric;
    badgeRow.appendChild(badge);
  }
  if (badgeRow.childElementCount > 0) {
    header.appendChild(badgeRow);
  }

  if (node.meta.tokenIds) {
    const tokenStrip = document.createElement("div");
    tokenStrip.className = "token-strip";
    node.meta.tokenIds.forEach((id, index) => {
      const chip = document.createElement("span");
      chip.className = "token-chip";
      chip.textContent = `${index}: ${node.meta.tokenTexts[index]} [${id}]`;
      tokenStrip.appendChild(chip);
    });
    header.appendChild(tokenStrip);
  }

  const content = document.createElement("div");
  content.className = "inspector-content";
  content.appendChild(header);

  if (node.meta.topPredictions) {
    content.appendChild(renderPredictionsCard(node.meta.topPredictions));
  }

  const groupedHeadTensors = node.tensors.filter((tensor) => tensor.shape.length === 3 && tensor.shape[0] > 1);
  const standaloneTensors =
    groupedHeadTensors.length > 1
      ? node.tensors.filter((tensor) => !groupedHeadTensors.includes(tensor))
      : node.tensors;

  if (groupedHeadTensors.length > 1) {
    content.appendChild(renderHeadTensorGroup(groupedHeadTensors));
  }

  standaloneTensors.forEach((tensor) => {
    content.appendChild(renderTensorCard(tensor));
  });

  elements.inspectorContent.innerHTML = "";
  elements.inspectorContent.appendChild(content);
  wireInspectorControls();
}

function renderPredictionsCard(predictions) {
  const card = document.createElement("section");
  card.className = "tensor-card";
  card.innerHTML = `
    <div class="tensor-topline">
      <div class="tensor-name">Top Probabilities</div>
      <div class="tensor-shape">12 candidates</div>
    </div>
  `;
  const list = document.createElement("ol");
  list.className = "predictions-list";
  predictions.forEach((entry) => {
    const item = document.createElement("li");
    item.innerHTML = `
      <span class="pred-token">${escapeHtml(entry.token)} <span class="tensor-shape">[${entry.id}]</span></span>
      <span class="pred-prob">${(entry.prob * 100).toFixed(2)}%</span>
    `;
    list.appendChild(item);
  });
  card.appendChild(list);
  return card;
}

function renderTensorCard(tensor) {
  const card = document.createElement("section");
  card.className = "tensor-card";
  const descriptor = tensorDescriptor(tensor.name);

  const stats = tensorStats(tensor.data);
  card.innerHTML = `
    <div class="tensor-topline">
      <div class="tensor-name">${escapeHtml(descriptor.title)}</div>
      <div class="tensor-shape">[${tensor.shape.join(" × ")}]</div>
    </div>
    ${descriptor.formula ? `<div class="tensor-formula">${escapeHtml(descriptor.formula)}</div>` : ""}
    <div class="tensor-stats">
      <span>min ${formatStat(stats.min)}</span>
      <span>mean ${formatStat(stats.mean)}</span>
      <span>max ${formatStat(stats.max)}</span>
    </div>
  `;

  if (tensor.shape.length === 3) {
    card.appendChild(renderHeadStack(tensor));
    return card;
  }

  if (tensor.shape.length > 3) {
    const controls = document.createElement("div");
    controls.className = "tensor-controls";
    const selectors = new Array(tensor.shape.length - 2).fill(0);
    tensor.shape.slice(0, -2).forEach((size, index) => {
      const label = document.createElement("label");
      label.className = "field";
      label.innerHTML = `<span>Axis ${index}</span>`;
      const range = document.createElement("input");
      range.type = "range";
      range.min = "0";
      range.max = String(Math.max(0, size - 1));
      range.value = "0";
      range.addEventListener("input", () => {
        selectors[index] = Number(range.value);
        renderTensorVisual();
      });
      label.appendChild(range);
      controls.appendChild(label);
    });
    card.appendChild(controls);

    const shell = document.createElement("div");
    card.appendChild(shell);

    function renderTensorVisual() {
      shell.innerHTML = "";
      const sliced = sliceTensor(tensor, selectors);
      const axis = inferAxes(tensor, sliced.shape[0], sliced.shape[1]);
      if (sliced.shape.length === 1 || sliced.shape[0] === 1) {
        shell.appendChild(renderVectorCanvas(flattenSlice(sliced), axis.xLabels, {
          title: descriptor.title,
          meta: descriptor.formula ? `${descriptor.formula} · [${sliced.shape.join(" × ")}]` : `[${sliced.shape.join(" × ")}]`,
        }));
      } else {
        shell.appendChild(renderHeatmapCanvas(sliced.data, sliced.shape[0], sliced.shape[1], axis, {
          title: descriptor.title,
          meta: descriptor.formula ? `${descriptor.formula} · [${sliced.shape.join(" × ")}]` : `[${sliced.shape.join(" × ")}]`,
        }));
      }
    }

    renderTensorVisual();
    return card;
  }

  if (tensor.shape.length === 1) {
    const axis = inferAxes(tensor, 1, tensor.shape[0]);
    card.appendChild(renderVectorCanvas(tensor.data, axis.xLabels, {
      title: descriptor.title,
      meta: descriptor.formula ? `${descriptor.formula} · [${tensor.shape.join(" × ")}]` : `[${tensor.shape.join(" × ")}]`,
    }));
    return card;
  }

  card.appendChild(renderHeatmapCanvas(tensor.data, tensor.shape[0], tensor.shape[1], inferAxes(tensor, tensor.shape[0], tensor.shape[1]), {
    title: descriptor.title,
    meta: descriptor.formula ? `${descriptor.formula} · [${tensor.shape.join(" × ")}]` : `[${tensor.shape.join(" × ")}]`,
  }));
  return card;
}

function flattenSlice(slice) {
  if (slice.shape.length === 1) {
    return slice.data;
  }
  if (slice.shape[0] === 1) {
    return slice.data;
  }
  return slice.data;
}

function renderHeadStack(tensor) {
  const stack = document.createElement("div");
  stack.className = "head-stack";
  for (let index = 0; index < tensor.shape[0]; index += 1) {
    const slice = sliceTensor(tensor, [index]);
    const headCard = document.createElement("div");
    headCard.className = "head-card";
    const axis = inferAxes(tensor, slice.shape[0], slice.shape[1]);
    headCard.innerHTML = `
      <div class="head-label">
        <span>Head ${index}</span>
        <span>[${slice.shape.join(" × ")}]</span>
      </div>
    `;
    if (slice.shape[0] === 1) {
      headCard.appendChild(renderVectorCanvas(flattenSlice(slice), axis.xLabels));
    } else {
      headCard.appendChild(renderHeatmapCanvas(slice.data, slice.shape[0], slice.shape[1], axis));
    }
    stack.appendChild(headCard);
  }
  return stack;
}

function renderHeadTensorGroup(tensors) {
  const card = document.createElement("section");
  card.className = "tensor-card head-group-card";
  const headCount = tensors[0].shape[0];
  const orderedTensors = [...tensors].sort(compareHeadTensorOrder);
  syncHeadTensorVisibility(orderedTensors.map((tensor) => tensor.name));
  const visibleTensors = orderedTensors.filter((tensor) => state.headTensorVisibility.has(tensor.name));

  const header = document.createElement("div");
  header.className = "head-group-header";
  header.innerHTML = `
    <div class="tensor-topline">
      <div class="tensor-name">Per-Head Maps</div>
      <div class="tensor-shape">${visibleTensors.length}/${orderedTensors.length} visible</div>
    </div>
  `;
  header.appendChild(renderHeadTensorSelector(orderedTensors));
  card.appendChild(header);

  const stack = document.createElement("div");
  stack.className = "head-group-stack";

  for (let headIndex = 0; headIndex < headCount; headIndex += 1) {
    const section = document.createElement("section");
    section.className = "head-section";
    section.innerHTML = `<div class="head-section-title">Head ${headIndex}</div>`;
    const grid = document.createElement("div");
    grid.className = "head-grid";

    visibleTensors.forEach((tensor) => {
      const slice = sliceTensor(tensor, [headIndex]);
      const axis = inferAxes(tensor, slice.shape[0], slice.shape[1]);
      const descriptor = tensorDescriptor(tensor.name);
      const tile = document.createElement("div");
      tile.className = "head-tile";
      tile.innerHTML = `
        <div class="head-tile-title">${escapeHtml(descriptor.title)}</div>
        ${descriptor.formula ? `<div class="tensor-formula">${escapeHtml(descriptor.formula)}</div>` : ""}
        <div class="tensor-shape">[${slice.shape.join(" × ")}]</div>
      `;
      if (slice.shape[0] === 1) {
        tile.appendChild(renderVectorCanvas(flattenSlice(slice), axis.xLabels, {
          interactive: true,
          title: `Head ${headIndex} · ${descriptor.title}`,
          meta: `[${slice.shape.join(" × ")}]`,
        }));
      } else {
        tile.appendChild(renderHeatmapCanvas(slice.data, slice.shape[0], slice.shape[1], axis, {
          interactive: true,
          title: `Head ${headIndex} · ${descriptor.title}`,
          meta: `${descriptor.formula ? `${descriptor.formula} · ` : ""}[${slice.shape.join(" × ")}]`,
        }));
      }
      grid.appendChild(tile);
    });

    section.appendChild(grid);
    stack.appendChild(section);
  }

  card.appendChild(stack);
  return card;
}

function renderHeadTensorSelector(tensors) {
  const wrapper = document.createElement("div");
  wrapper.className = "map-selector";
  wrapper.innerHTML = `
    <div class="map-selector-actions">
      <button class="ghost-button selector-action" type="button" data-map-action="all">All</button>
      <button class="ghost-button selector-action" type="button" data-map-action="none">None</button>
    </div>
  `;

  const chips = document.createElement("div");
  chips.className = "map-chip-row";
  tensors.forEach((tensor) => {
    const descriptor = tensorDescriptor(tensor.name);
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `map-chip ${state.headTensorVisibility.has(tensor.name) ? "active" : ""}`;
    chip.setAttribute("data-map-toggle", tensor.name);
    chip.textContent = shortMapLabel(descriptor.title);
    chips.appendChild(chip);
  });
  wrapper.appendChild(chips);
  return wrapper;
}

function renderVectorCanvas(values, xLabels = null, options = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "axis-shell";
  const canvas = document.createElement("canvas");
  const width = options.popup ? 1080 : 520;
  const height = options.popup ? 260 : 120;
  canvas.width = width;
  canvas.height = height;
  const shell = document.createElement("div");
  shell.className = "visual-shell";
  shell.appendChild(canvas);
  wrapper.appendChild(shell);
  drawVector(canvas, values);
  if (xLabels) {
    wrapper.appendChild(renderXAxisLabels(values.length, xLabels));
  }
  if (options.interactive !== false) {
    shell.classList.add("plot-clickable");
    shell.addEventListener("click", () => {
      openPlotModal({
        type: "vector",
        values,
        xLabels,
        title: options.title ?? "Vector",
        meta: options.meta ?? "",
      });
    });
  }
  return wrapper;
}

function renderHeatmapCanvas(values, rows, cols, axis = {}, options = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "axis-shell";
  const canvas = document.createElement("canvas");
  const targetWidth = options.popup ? Math.max(760, Math.min(1320, cols * 16 + 96)) : Math.max(320, Math.min(760, cols * 8 + 56));
  const targetHeight = options.popup ? Math.max(360, Math.min(920, rows * 24 + 96)) : Math.max(140, Math.min(320, rows * 12 + 28));
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const shell = document.createElement("div");
  shell.className = "visual-shell";
  shell.appendChild(canvas);
  wrapper.appendChild(shell);
  drawHeatmap(canvas, values, rows, cols, axis);
  if (options.interactive !== false) {
    shell.classList.add("plot-clickable");
    shell.addEventListener("click", () => {
      openPlotModal({
        type: "heatmap",
        values,
        rows,
        cols,
        axis,
        title: options.title ?? "Heatmap",
        meta: options.meta ?? "",
      });
    });
  }
  return wrapper;
}

function openPlotModal(spec) {
  state.modalPlot = spec;
  elements.plotTitle.textContent = spec.title ?? "Plot";
  elements.plotMeta.textContent = spec.meta ?? "";
  elements.plotBody.innerHTML = "";
  if (spec.type === "vector") {
    elements.plotBody.appendChild(
      renderVectorCanvas(spec.values, spec.xLabels, {
        popup: true,
        interactive: false,
      })
    );
  } else {
    elements.plotBody.appendChild(
      renderHeatmapCanvas(spec.values, spec.rows, spec.cols, spec.axis, {
        popup: true,
        interactive: false,
      })
    );
  }
  elements.plotModal.classList.remove("hidden");
  elements.plotModal.setAttribute("aria-hidden", "false");
}

function closePlotModal() {
  state.modalPlot = null;
  elements.plotBody.innerHTML = "";
  elements.plotModal.classList.add("hidden");
  elements.plotModal.setAttribute("aria-hidden", "true");
}

function wireInspectorControls() {
  elements.inspectorContent.querySelectorAll("[data-map-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const name = button.getAttribute("data-map-toggle");
      if (state.headTensorVisibility.has(name)) {
        state.headTensorVisibility.delete(name);
      } else {
        state.headTensorVisibility.add(name);
      }
      persistUiState();
      renderInspector();
    });
  });

  elements.inspectorContent.querySelectorAll("[data-map-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.getAttribute("data-map-action");
      if (action === "all") {
        state.headTensorVisibility = new Set(state.headTensorOptions);
      } else if (action === "none") {
        state.headTensorVisibility = new Set();
      }
      persistUiState();
      renderInspector();
    });
  });
}

function drawVector(canvas, values) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#07131b";
  ctx.fillRect(0, 0, width, height);

  let maxAbs = 0;
  values.forEach((value) => {
    if (Number.isFinite(value)) {
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  });
  maxAbs = maxAbs || 1;

  const zeroY = height / 2;
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  ctx.lineTo(width, zeroY);
  ctx.stroke();

  const barWidth = width / Math.max(values.length, 1);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    const scaled = (value / maxAbs) * (height * 0.42);
    const x = index * barWidth;
    const y = scaled >= 0 ? zeroY - scaled : zeroY;
    const h = Math.abs(scaled);
    ctx.fillStyle = scaled >= 0 ? "rgba(121,240,217,0.78)" : "rgba(255,126,140,0.78)";
    ctx.fillRect(x, y, Math.max(1, barWidth - 1), Math.max(1, h));
  }
}

function drawHeatmap(canvas, values, rows, cols, axis = {}) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0d1826";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const xTicks = buildAxisTicks(cols, axis.xLabels ?? createDimensionLabels(cols, "x"));
  const yTicks = buildAxisTicks(rows, axis.yLabels ?? createDimensionLabels(rows, "y"));
  const leftPad = yTicks.length > 0 ? 68 : 8;
  const bottomPad = xTicks.length > 0 ? 52 : 6;
  const topPad = 8;
  const rightPad = 6;
  const plotWidth = Math.max(10, canvas.width - leftPad - rightPad);
  const plotHeight = Math.max(10, canvas.height - topPad - bottomPad);
  const cellWidth = plotWidth / Math.max(cols, 1);
  const cellHeight = plotHeight / Math.max(rows, 1);

  let maxAbs = 0;
  values.forEach((value) => {
    if (Number.isFinite(value)) {
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  });
  maxAbs = maxAbs || 1;

  const temp = document.createElement("canvas");
  temp.width = cols;
  temp.height = rows;
  const tempCtx = temp.getContext("2d");
  const image = tempCtx.createImageData(cols, rows);

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    const color = heatColor(Number.isFinite(value) ? value / maxAbs : 0);
    const pixel = index * 4;
    image.data[pixel] = color[0];
    image.data[pixel + 1] = color[1];
    image.data[pixel + 2] = color[2];
    image.data[pixel + 3] = 255;
  }

  tempCtx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(temp, leftPad, topPad, plotWidth, plotHeight);

  ctx.strokeStyle = "rgba(255,255,255,0.14)";
  ctx.lineWidth = 1;
  ctx.strokeRect(leftPad + 0.5, topPad + 0.5, plotWidth, plotHeight);

  ctx.fillStyle = "#a8b7c7";
  ctx.font = "10px Segoe UI";
  ctx.textBaseline = "middle";

  for (const { index, label } of xTicks) {
    const x = leftPad + (index + 0.5) * cellWidth;
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(x + 0.5, topPad + plotHeight);
    ctx.lineTo(x + 0.5, topPad + plotHeight + 4);
    ctx.stroke();
    ctx.save();
    ctx.translate(x, topPad + plotHeight + 8);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = "right";
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }

  for (const { index, label } of yTicks) {
    const y = topPad + (index + 0.5) * cellHeight;
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(leftPad - 4, y + 0.5);
    ctx.lineTo(leftPad, y + 0.5);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(label, leftPad - 8, y);
  }
}

function renderXAxisLabels(size, labels) {
  if (size <= 48) {
    const axis = document.createElement("div");
    axis.className = "axis-x axis-grid";
    axis.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
    for (let index = 0; index < size; index += 1) {
      const span = document.createElement("span");
      span.textContent = shortLabel(labels[index] ?? "");
      axis.appendChild(span);
    }
    return axis;
  }

  const axis = document.createElement("div");
  axis.className = "axis-x axis-sparse";
  buildAxisTicks(size, labels).forEach(({ index, label }) => {
    const span = document.createElement("span");
    span.textContent = label;
    span.style.left = `${size === 1 ? 0 : (index / (size - 1)) * 100}%`;
    span.style.transform = "translateX(-50%)";
    axis.appendChild(span);
  });
  return axis;
}

function renderYAxisLabels(size, labels) {
  if (size <= 32) {
    const axis = document.createElement("div");
    axis.className = "axis-y axis-grid";
    axis.style.gridTemplateRows = `repeat(${size}, minmax(0, 1fr))`;
    for (let index = 0; index < size; index += 1) {
      const span = document.createElement("span");
      span.textContent = shortLabel(labels[index] ?? "");
      axis.appendChild(span);
    }
    return axis;
  }

  const axis = document.createElement("div");
  axis.className = "axis-y axis-sparse";
  buildAxisTicks(size, labels).forEach(({ index, label }) => {
    const span = document.createElement("span");
    span.textContent = label;
    span.style.top = `${size === 1 ? 0 : (index / (size - 1)) * 100}%`;
    span.style.transform = "translateY(-50%)";
    axis.appendChild(span);
  });
  return axis;
}

function buildAxisTicks(size, labels) {
  if (Array.isArray(labels) && labels.length === size && size <= 24) {
    return labels.map((label, index) => ({ index, label: shortLabel(label) }));
  }

  const ticks = [];
  if (size === 1) {
    return [{ index: 0, label: Array.isArray(labels) && labels[0] ? shortLabel(labels[0]) : "0" }];
  }
  const tickCount = Math.min(Array.isArray(labels) && labels.length === size ? 12 : 8, size);
  const seen = new Set();
  for (let tick = 0; tick < tickCount; tick += 1) {
    const index = Math.round((tick / (tickCount - 1)) * (size - 1));
    if (seen.has(index)) {
      continue;
    }
    seen.add(index);
    const label = Array.isArray(labels) && labels.length === size ? shortLabel(labels[index]) : String(index);
    ticks.push({ index, label });
  }
  return ticks;
}

function inferAxes(tensor, rows, cols) {
  const tokenLabels = guessTokenLabels(rows, cols);
  if (tensor.name.includes("attention") || tensor.name.includes("scores")) {
    return {
      xLabels: tokenLabels.cols,
      yLabels: tokenLabels.rows,
    };
  }
  if (tensor.name.includes("queries") || tensor.name.includes("keys") || tensor.name.includes("values") || tensor.name.includes("head_outputs")) {
    return {
      xLabels: createDimensionLabels(cols, "d"),
      yLabels: tokenLabels.rows,
    };
  }
  if (tensor.name.includes("token_embeddings") || tensor.name.includes("position_embeddings") || tensor.name.includes("residual") || tensor.name === "input" || tensor.name === "output" || tensor.name.includes("projected_output") || tensor.name.includes("mlp_out")) {
    return {
      xLabels: createDimensionLabels(cols, "f"),
      yLabels: tokenLabels.rows,
    };
  }
  if (tensor.name.includes("logits")) {
    return {
      xLabels: createDimensionLabels(cols, "v"),
      yLabels: tokenLabels.rows,
    };
  }
  if (tensor.shape.length === 1) {
    return {
      xLabels: createDimensionLabels(cols, "i"),
      yLabels: null,
    };
  }
  return {
    xLabels: createDimensionLabels(cols, "x"),
    yLabels: createDimensionLabels(rows, "y"),
  };
}

function guessTokenLabels(rows, cols) {
  const node = state.trace?.nodes.find((entry) => entry.id === state.selectedNodeId);
  const tokenTexts = node?.meta?.tokenTexts ?? null;
  return {
    rows: tokenTexts && tokenTexts.length === rows ? tokenTexts : createDimensionLabels(rows, "t"),
    cols: tokenTexts && tokenTexts.length === cols ? tokenTexts : createDimensionLabels(cols, "t"),
  };
}

function createDimensionLabels(size, prefix) {
  const labels = new Array(size).fill("");
  if (size <= 64) {
    for (let index = 0; index < size; index += 1) {
      labels[index] = `${prefix}${index}`;
    }
    return labels;
  }
  const tickCount = Math.min(8, size);
  for (let tick = 0; tick < tickCount; tick += 1) {
    const index = Math.round((tick / (tickCount - 1)) * (size - 1));
    labels[index] = `${prefix}${index}`;
  }
  return labels;
}

function tensorDescriptor(name) {
  const formulas = {
    scores_raw: "(Q · K^T) / sqrt(d_head)",
    scores_masked: "mask(scores_raw, causal) -> -inf above diagonal",
    attention_probs: "softmax(scores_masked)",
  };
  return {
    title: name,
    formula: formulas[name] ?? "",
  };
}

function compareHeadTensorOrder(left, right) {
  const order = new Map([
    ["keys", 0],
    ["queries", 1],
    ["values", 2],
    ["scores_raw", 3],
    ["scores_masked", 4],
    ["attention_probs", 5],
    ["head_outputs", 6],
  ]);
  const leftOrder = order.get(left.name) ?? 99;
  const rightOrder = order.get(right.name) ?? 99;
  if (leftOrder !== rightOrder) {
    return leftOrder - rightOrder;
  }
  return left.name.localeCompare(right.name);
}

function syncHeadTensorVisibility(names) {
  if (!state.headTensorVisibility) {
    state.headTensorVisibility = new Set(names.filter((name) => !DEFAULT_HIDDEN_HEAD_MAPS.has(name)));
    state.headTensorOptions = [...names];
    persistUiState();
    return;
  }

  const previousOptions = state.headTensorOptions;
  const overlap = names.some((name) => previousOptions.includes(name));
  if (!overlap) {
    state.headTensorVisibility = new Set(names.filter((name) => !DEFAULT_HIDDEN_HEAD_MAPS.has(name)));
    state.headTensorOptions = [...names];
    persistUiState();
    return;
  }

  const next = new Set();
  names.forEach((name) => {
    if (state.headTensorVisibility.has(name)) {
      next.add(name);
    }
    if (!previousOptions.includes(name)) {
      next.add(name);
    }
  });
  state.headTensorVisibility = next;
  state.headTensorOptions = [...names];
  persistUiState();
}

function shortMapLabel(label) {
  const mapping = {
    queries: "Q",
    keys: "K",
    values: "V",
    scores_raw: "Raw Scores",
    scores_masked: "Masked Scores",
    attention_probs: "Attention",
    head_outputs: "Head Out",
  };
  return mapping[label] ?? label;
}

const DEFAULT_HIDDEN_HEAD_MAPS = new Set(["keys", "queries", "values"]);

function persistUiState() {
  try {
    const payload = {
      prompt: elements.promptInput.value,
      temperature: elements.temperatureInput.value,
      topK: elements.topkInput.value,
      strategy: elements.strategyInput.value,
      seed: elements.seedInput.value,
      selectedNodeId: state.selectedNodeId,
      headTensorVisibility: state.headTensorVisibility ? [...state.headTensorVisibility] : null,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Ignore storage failures; the UI should still work without persistence.
  }
}

function restoreUiState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return;
    }
    const saved = JSON.parse(raw);
    if (typeof saved.prompt === "string") {
      elements.promptInput.value = saved.prompt;
    }
    if (saved.temperature !== undefined) {
      elements.temperatureInput.value = saved.temperature;
    }
    if (saved.topK !== undefined) {
      elements.topkInput.value = saved.topK;
    }
    if (saved.strategy !== undefined) {
      elements.strategyInput.value = saved.strategy;
    }
    if (saved.seed !== undefined) {
      elements.seedInput.value = saved.seed;
    }
    if (typeof saved.selectedNodeId === "string") {
      state.selectedNodeId = saved.selectedNodeId;
    }
    if (Array.isArray(saved.headTensorVisibility)) {
      state.headTensorVisibility = new Set(saved.headTensorVisibility);
    }
  } catch {
    // Ignore invalid saved state and start fresh.
  }
}

function shortLabel(value) {
  const text = String(value);
  if (text === "\n" || text === "\\n") {
    return "\\n";
  }
  return text.length > 8 ? `${text.slice(0, 7)}…` : text;
}

function heatColor(value) {
  const clamped = Math.max(-1, Math.min(1, value));
  if (clamped >= 0) {
    const t = clamped;
    return [
      lerp(18, 255, t),
      lerp(36, 184, t),
      lerp(48, 112, t),
    ];
  }
  const t = Math.abs(clamped);
  return [
    lerp(12, 146, t),
    lerp(30, 167, t),
    lerp(55, 255, t),
  ];
}

function lerp(a, b, t) {
  return Math.round(a + (b - a) * t);
}

function setStatus(text) {
  elements.statusPill.textContent = text;
}

function formatStat(value) {
  return Number.isFinite(value) ? value.toFixed(4) : "n/a";
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
