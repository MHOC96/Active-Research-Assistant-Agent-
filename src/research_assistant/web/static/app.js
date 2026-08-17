const STORAGE_KEY = "research-assistant-ui";
const HISTORY_KEY = "research-assistant-history";
const MAX_HISTORY = 8;

const EXAMPLE_QUERIES = [
  "Compare RAG and GraphRAG",
  "What is transformer self-attention?",
  "How does retrieval-augmented generation reduce hallucinations?",
];

const PROGRESS_STEPS = ["analyze", "retrieve", "discover", "ingest", "format"];

const queryInput = document.getElementById("query");
const queryEditor = document.getElementById("query-editor");
const queryBackdrop = document.getElementById("query-backdrop");
const segmentCitations = document.getElementById("segment-citations");
const citationStyleSelect = document.getElementById("citation-style");
const fastModeInput = document.getElementById("fast-mode");
const submitBtn = document.getElementById("submit-btn");
const cancelBtn = document.getElementById("cancel-btn");
const cancelLoadingBtn = document.getElementById("cancel-loading-btn");
const clearBtn = document.getElementById("clear-btn");
const rerunBtn = document.getElementById("rerun-btn");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");
const bundleBtn = document.getElementById("bundle-btn");
const resultOutput = document.getElementById("result-output");
const resultToolbar = document.getElementById("result-toolbar");
const statsRow = document.getElementById("stats-row");
const emptyState = document.getElementById("empty-state");
const loading = document.getElementById("loading");
const loadingTimer = document.getElementById("loading-timer");
const progressSteps = document.getElementById("progress-steps");
const warnings = document.getElementById("warnings");
const healthStatus = document.getElementById("health-status");
const healthLabel = document.getElementById("health-label");
const charCount = document.getElementById("char-count");
const exampleChips = document.getElementById("example-chips");
const helpBtn = document.getElementById("help-btn");
const helpPanel = document.getElementById("help-panel");
const historyList = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const removeSelectedBtn = document.getElementById("remove-selected-btn");
const historySelectAllWrap = document.getElementById("history-select-all-wrap");
const historySelectAll = document.getElementById("history-select-all");
const toast = document.getElementById("toast");

let latestPlainResult = "";
let viewingCachedResult = false;
let activeHistoryId = null;
let lastDisplayedState = null;
let snapshotBeforeRun = null;
let activeAbortController = null;
let activeRequestId = null;
let selectedHistoryIds = new Set();
let loadingInterval = null;
let progressInterval = null;
let progressIndex = 0;
let activeCitationSpans = [];
let activeSegmentId = null;
let activeTotalSourceCount = 0;

function loadPreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (saved.citationStyle) citationStyleSelect.value = saved.citationStyle;
    if (typeof saved.fastMode === "boolean") fastModeInput.checked = saved.fastMode;
    if (saved.draftQuery) queryInput.value = saved.draftQuery;
  } catch {
    /* ignore invalid storage */
  }
  updateCharCount();
}

function savePreferences() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      citationStyle: citationStyleSelect.value,
      fastMode: fastModeInput.checked,
      draftQuery: queryInput.value,
    }),
  );
}

function currentRequestOptions() {
  return {
    query: queryInput.value.trim(),
    style: citationStyleSelect.value,
    fast: fastModeInput.checked,
  };
}

function requestCacheKey({ query, style, fast }) {
  return JSON.stringify({
    query: query.trim(),
    style,
    fast,
  });
}

function normalizeHistoryItem(item, index) {
  return {
    ...item,
    id: item.id || `legacy-${index}-${item.query.slice(0, 24)}`,
  };
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]").map(normalizeHistoryItem);
  } catch {
    return [];
  }
}

function writeHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
}

function findCachedEntry(options) {
  const key = requestCacheKey(options);
  return readHistory().find((item) => item.cacheKey === key && item.result);
}

function updateCharCount() {
  charCount.textContent = `${queryInput.value.length} / 8000`;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => toast.classList.add("hidden"), 2400);
}

async function loadCitationStyles() {
  const response = await fetch("/api/citation-styles");
  const styles = await response.json();
  citationStyleSelect.innerHTML = styles
    .map((style) => `<option value="${style.id}">${style.id.toUpperCase()} - ${style.description}</option>`)
    .join("");
  loadPreferences();
}

function renderExamples() {
  exampleChips.innerHTML = EXAMPLE_QUERIES.map(
    (query) => `<button type="button" class="chip" data-query="${escapeHtml(query)}">${escapeHtml(query)}</button>`,
  ).join("");
  exampleChips.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.dataset.query;
      updateCharCount();
      savePreferences();
      queryInput.focus();
    });
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    healthStatus.classList.remove("ok", "error");
    if (data.ok) {
      healthLabel.textContent = "APIs ready";
      healthStatus.classList.add("ok");
      return;
    }
    const errors = [...(data.config_errors || []), ...(data.service_errors || [])];
    healthLabel.textContent = errors[0] || "Configuration issue";
    healthStatus.classList.add("error");
  } catch {
    healthLabel.textContent = "Server unavailable";
    healthStatus.classList.add("error");
  }
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderQueryHighlights(spans, { totalSourceCount = 0 } = {}) {
  activeCitationSpans = spans || [];
  activeSegmentId = null;
  activeTotalSourceCount = totalSourceCount;

  if (!activeCitationSpans.length) {
    queryEditor.classList.remove("has-highlights");
    queryInput.readOnly = false;
    queryBackdrop.innerHTML = "";
    segmentCitations.classList.add("hidden");
    segmentCitations.innerHTML = "";
    return;
  }

  const text = queryInput.value;
  let html = "";
  let cursor = 0;
  const sorted = [...activeCitationSpans].sort((a, b) => a.start - b.start);

  for (const span of sorted) {
    const safeStart = Math.max(0, Math.min(span.start, text.length));
    const safeEnd = Math.max(safeStart, Math.min(span.end, text.length));
    if (safeStart > cursor) {
      html += escapeHtml(text.slice(cursor, safeStart));
    }
    const hasCitations = Array.isArray(span.citations) && span.citations.length > 0;
    const className = hasCitations ? "cite-span" : "cite-span cite-span-empty";
    html += `<mark class="${className}" data-segment-id="${escapeHtml(span.segment_id)}">${escapeHtml(text.slice(safeStart, safeEnd))}</mark>`;
    cursor = safeEnd;
  }
  html += escapeHtml(text.slice(cursor));

  queryBackdrop.innerHTML = html;
  queryEditor.classList.add("has-highlights");
  queryInput.readOnly = true;
  segmentCitations.classList.add("hidden");
  segmentCitations.innerHTML = "";
  syncQueryBackdropScroll();
}

function findSpanAtPosition(position) {
  if (!activeCitationSpans.length) return null;

  const sorted = [...activeCitationSpans].sort((a, b) => a.start - b.start);
  for (let index = 0; index < sorted.length; index += 1) {
    const span = sorted[index];
    const isLast = index === sorted.length - 1;
    const inRange = isLast
      ? position >= span.start && position <= span.end
      : position >= span.start && position < span.end;
    if (inRange) return span;
  }
  return null;
}

function findSpanById(segmentId) {
  return activeCitationSpans.find((span) => span.segment_id === segmentId) || null;
}

function setActiveSegment(segmentId) {
  activeSegmentId = segmentId;
  queryBackdrop.querySelectorAll(".cite-span").forEach((mark) => {
    mark.classList.toggle("active", mark.dataset.segmentId === segmentId);
  });
}

function renderSegmentCitations(span) {
  if (!span) {
    segmentCitations.classList.add("hidden");
    segmentCitations.innerHTML = "";
    return;
  }

  const citations = span.citations || [];
  const segmentIndex = activeCitationSpans.findIndex((item) => item.segment_id === span.segment_id);
  const segmentLabel =
    segmentIndex >= 0
      ? `Sentence ${segmentIndex + 1} of ${activeCitationSpans.length}`
      : "Selected sentence";
  const totalNote =
    activeTotalSourceCount > citations.length
      ? `<p class="segment-citations-meta">${activeTotalSourceCount} total source(s) in Results. Showing ${citations.length} for this sentence.</p>`
      : "";

  const cards =
    citations.length > 0
      ? citations
          .map(
            (citation) => `
        <div class="segment-citation-card">
          <div class="segment-citation-meta">
            <span class="segment-citation-source">${escapeHtml(citation.source_label || citation.source)}</span>
          </div>
          <div>${escapeHtml(citation.reference)}</div>
          ${
            citation.in_text
              ? `<div class="segment-citation-intext">In-text: ${escapeHtml(citation.in_text)}</div>`
              : ""
          }
          ${
            citation.url && citation.url.startsWith("http")
              ? `<div class="segment-citation-intext"><a href="${citation.url}" target="_blank" rel="noopener noreferrer">Open source</a></div>`
              : ""
          }
        </div>`,
          )
          .join("")
      : `<p class="segment-citation-empty">No sources in the Results list matched this sentence.</p>`;

  segmentCitations.innerHTML = `
    <div class="segment-citations-header">
      <div>
        <h3>Citations for selected text</h3>
        <p>"${escapeHtml(span.text)}"</p>
        ${totalNote}
      </div>
      <span class="segment-citation-count">${citations.length} source(s)</span>
    </div>
    <p class="segment-citations-meta">${escapeHtml(segmentLabel)} · Search: ${escapeHtml(span.search_query || "")}</p>
    ${cards}
  `;
  segmentCitations.classList.remove("hidden");
}

function showSegmentById(segmentId) {
  const span = findSpanById(segmentId);
  if (!span) {
    setActiveSegment(null);
    segmentCitations.classList.add("hidden");
    return;
  }
  setActiveSegment(span.segment_id);
  renderSegmentCitations(span);
  segmentCitations.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showSegmentAtPosition(position) {
  const span = findSpanAtPosition(position);
  if (!span) {
    setActiveSegment(null);
    segmentCitations.classList.add("hidden");
    return;
  }
  showSegmentById(span.segment_id);
}

function clearCitationHighlights() {
  renderQueryHighlights([]);
}

function syncQueryBackdropScroll() {
  queryBackdrop.scrollTop = queryInput.scrollTop;
  queryBackdrop.scrollLeft = queryInput.scrollLeft;
}

function formatResultHtml(answer) {
  if (answer.startsWith("INSUFFICIENT_EVIDENCE:")) {
    return `<p class="error-text">${escapeHtml(answer)}</p>`;
  }

  if (answer.startsWith("References\n\n")) {
    const body = answer.slice("References\n\n".length).trim();
    const entries = body.split("\n\n").filter(Boolean);
    const entriesHtml = entries
      .map((entry) => `<div class="reference-entry">${escapeHtml(entry)}</div>`)
      .join("");
    return `<div class="references-title">References</div>${entriesHtml}`;
  }

  return `<div>${escapeHtml(answer)}</div>`;
}

function renderStats(data, { fromCache = false } = {}) {
  const badges = [];

  if (fromCache) {
    badges.push(`<span class="stat-badge cached">Saved result</span>`);
  } else {
    badges.push(`<span class="stat-badge">${data.elapsed_seconds}s</span>`);
  }

  badges.push(
    `<span class="stat-badge">${data.citation_style.toUpperCase()}</span>`,
    `<span class="stat-badge ${data.sufficient ? "success" : "warning"}">${data.sufficient ? "Sufficient evidence" : "Limited evidence"}</span>`,
  );

  if (data.papers_ingested > 0) {
    badges.push(`<span class="stat-badge">${data.papers_ingested} paper(s) ingested</span>`);
  } else if (!fromCache) {
    badges.push(`<span class="stat-badge">Used cached index</span>`);
  }

  if (data.source_count > 0) {
    badges.push(`<span class="stat-badge">${data.source_count} source(s)</span>`);
  }

  if (fromCache && data.elapsed_seconds) {
    badges.push(`<span class="stat-badge">Originally ${data.elapsed_seconds}s</span>`);
  }

  statsRow.innerHTML = badges.join("");
  statsRow.classList.remove("hidden");
}

function displayResult(data, { fromCache = false, historyId = null } = {}) {
  latestPlainResult = data.answer;
  viewingCachedResult = fromCache;
  activeHistoryId = historyId;
  lastDisplayedState = { data, fromCache, historyId };

  resultOutput.innerHTML = formatResultHtml(data.answer);
  resultOutput.classList.remove("hidden");
  emptyState.classList.add("hidden");
  resultToolbar.classList.remove("hidden");
  rerunBtn.classList.toggle("hidden", !fromCache);
  renderStats(data, { fromCache });

  renderQueryHighlights(data.citation_spans || [], {
    totalSourceCount: data.source_count || 0,
  });

  if (!data.citations_valid && data.citation_errors?.length) {
    showWarnings(data.citation_errors);
  } else {
    showWarnings([]);
  }

  highlightActiveHistory(historyId);
}

function highlightActiveHistory(historyId) {
  historyList.querySelectorAll(".history-item").forEach((item) => {
    item.classList.toggle("active", historyId !== null && item.dataset.id === historyId);
  });
}

function setProgressStep(stepName) {
  const items = progressSteps.querySelectorAll("li");
  const targetIndex = PROGRESS_STEPS.indexOf(stepName);
  items.forEach((item, index) => {
    item.classList.remove("active", "done");
    if (index < targetIndex) item.classList.add("done");
    if (index === targetIndex) item.classList.add("active");
  });
}

function startLoadingUi() {
  let seconds = 0;
  progressIndex = 0;
  loadingTimer.textContent = "0s";
  setProgressStep(PROGRESS_STEPS[0]);

  loadingInterval = setInterval(() => {
    seconds += 1;
    loadingTimer.textContent = `${seconds}s`;
  }, 1000);

  progressInterval = setInterval(() => {
    progressIndex = Math.min(progressIndex + 1, PROGRESS_STEPS.length - 1);
    setProgressStep(PROGRESS_STEPS[progressIndex]);
  }, 8000);
}

function stopLoadingUi() {
  clearInterval(loadingInterval);
  clearInterval(progressInterval);
  loadingInterval = null;
  progressInterval = null;
}

function setLoading(isLoading) {
  loading.classList.toggle("hidden", !isLoading);
  submitBtn.classList.toggle("hidden", isLoading);
  cancelBtn.classList.toggle("hidden", !isLoading);
  submitBtn.disabled = isLoading;
  rerunBtn.disabled = isLoading;

  if (isLoading) {
    emptyState.classList.add("hidden");
    resultOutput.classList.add("hidden");
    resultToolbar.classList.add("hidden");
    statsRow.classList.add("hidden");
    startLoadingUi();
  } else {
    stopLoadingUi();
  }
}

function createRequestId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `req-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function cancelActiveRequest() {
  if (activeAbortController) {
    activeAbortController.abort();
    activeAbortController = null;
  }

  if (activeRequestId) {
    const requestId = activeRequestId;
    fetch(`/api/query/cancel/${encodeURIComponent(requestId)}`, { method: "POST" }).catch(() => {});
    activeRequestId = null;
  }
}

function restoreAfterCancel() {
  if (snapshotBeforeRun) {
    displayResult(snapshotBeforeRun.data, {
      fromCache: snapshotBeforeRun.fromCache,
      historyId: snapshotBeforeRun.historyId,
    });
    return;
  }
  clearDisplayedResult();
}

function showWarnings(messages) {
  if (!messages.length) {
    warnings.classList.add("hidden");
    warnings.textContent = "";
    return;
  }
  warnings.classList.remove("hidden");
  warnings.textContent = messages.join("\n");
}

function formatHistoryMeta(item) {
  const savedTag = item.result ? '<span class="history-saved-tag">saved</span>' : "";
  const style = item.style || "unknown";
  const elapsed = item.elapsed ?? item.result?.elapsed_seconds ?? "?";
  return `${escapeHtml(style)} | ${escapeHtml(String(elapsed))}s${savedTag}`;
}

function updateHistorySelectionUi() {
  const history = readHistory();
  const selectedCount = selectedHistoryIds.size;

  removeSelectedBtn.disabled = selectedCount === 0;
  removeSelectedBtn.textContent =
    selectedCount > 0 ? `Remove selected (${selectedCount})` : "Remove selected";

  historySelectAllWrap.classList.toggle("hidden", history.length === 0);

  if (history.length === 0) {
    historySelectAll.checked = false;
    historySelectAll.indeterminate = false;
    return;
  }

  if (selectedCount === 0) {
    historySelectAll.checked = false;
    historySelectAll.indeterminate = false;
  } else if (selectedCount === history.length) {
    historySelectAll.checked = true;
    historySelectAll.indeterminate = false;
  } else {
    historySelectAll.checked = false;
    historySelectAll.indeterminate = true;
  }
}

function setHistoryItemSelected(id, selected) {
  if (selected) selectedHistoryIds.add(id);
  else selectedHistoryIds.delete(id);

  const row = historyList.querySelector(`.history-item[data-id="${CSS.escape(id)}"]`);
  if (row) row.classList.toggle("selected", selected);

  updateHistorySelectionUi();
}

function loadHistory() {
  const history = readHistory();
  const knownIds = new Set(history.map((item) => item.id));
  selectedHistoryIds = new Set([...selectedHistoryIds].filter((id) => knownIds.has(id)));

  if (!history.length) {
    historyList.innerHTML = '<li class="history-empty">No recent queries yet.</li>';
    updateHistorySelectionUi();
    return;
  }

  historyList.innerHTML = history
    .map(
      (item, index) => `
        <li class="history-item ${selectedHistoryIds.has(item.id) ? "selected" : ""}" data-id="${escapeHtml(item.id)}">
          <input
            type="checkbox"
            class="history-select"
            data-id="${escapeHtml(item.id)}"
            aria-label="Select query"
            ${selectedHistoryIds.has(item.id) ? "checked" : ""}
          />
          <button type="button" class="history-open" data-index="${index}">
            ${escapeHtml(item.query.slice(0, 120))}${item.query.length > 120 ? "..." : ""}
            <span class="history-meta">${formatHistoryMeta(item)}</span>
          </button>
          <button
            type="button"
            class="history-rerun ghost small"
            data-index="${index}"
            title="Re-run pipeline"
            aria-label="Re-run pipeline"
          >Re-run</button>
          <button
            type="button"
            class="history-remove ghost small"
            data-id="${escapeHtml(item.id)}"
            title="Remove from history"
            aria-label="Remove from history"
          >Remove</button>
        </li>`,
    )
    .join("");

  historyList.querySelectorAll(".history-select").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      setHistoryItemSelected(checkbox.dataset.id, checkbox.checked);
    });
  });

  historyList.querySelectorAll(".history-open").forEach((button) => {
    button.addEventListener("click", () => {
      openHistoryItem(Number(button.dataset.index));
    });
  });

  historyList.querySelectorAll(".history-rerun").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      openHistoryItem(Number(button.dataset.index), { force: true });
    });
  });

  historyList.querySelectorAll(".history-remove").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      removeHistoryItems([button.dataset.id]);
    });
  });

  highlightActiveHistory(activeHistoryId);
  updateHistorySelectionUi();
}

function removeHistoryItems(ids) {
  if (!ids.length) return;

  const idSet = new Set(ids);
  const removedActive = activeHistoryId !== null && idSet.has(activeHistoryId);
  const history = readHistory().filter((item) => !idSet.has(item.id));

  ids.forEach((id) => selectedHistoryIds.delete(id));
  writeHistory(history);
  loadHistory();

  if (removedActive) {
    clearDisplayedResult();
  }

  showToast(ids.length === 1 ? "Removed 1 item" : `Removed ${ids.length} items`);
}

function applyHistoryOptions(item) {
  queryInput.value = item.query;
  citationStyleSelect.value = item.style || citationStyleSelect.value;
  fastModeInput.checked = item.fast ?? true;
  updateCharCount();
  savePreferences();
}

function openHistoryItem(index, { force = false } = {}) {
  const history = readHistory();
  const item = history[index];
  if (!item) return;

  applyHistoryOptions(item);

  if (!force && item.result) {
    displayResult(item.result, { fromCache: true, historyId: item.id });
    showToast("Loaded saved result");
    return;
  }

  if (!item.result) {
    showToast("No saved result for this query - running pipeline");
  }

  runQuery({ force: true });
}

function saveHistory(entry) {
  const history = readHistory().filter((item) => item.cacheKey !== entry.cacheKey);
  history.unshift(entry);
  writeHistory(history);
  loadHistory();
}

function buildHistoryEntry(options, data) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    cacheKey: requestCacheKey(options),
    query: data.query,
    style: options.style,
    fast: options.fast,
    elapsed: data.elapsed_seconds,
    savedAt: new Date().toISOString(),
    result: {
      answer: data.answer,
      citation_style: data.citation_style,
      citations_valid: data.citations_valid,
      citation_errors: data.citation_errors || [],
      sufficient: data.sufficient,
      papers_ingested: data.papers_ingested,
      source_count: data.source_count,
      citation_spans: data.citation_spans || [],
      elapsed_seconds: data.elapsed_seconds,
    },
  };
}

function clearDisplayedResult() {
  latestPlainResult = "";
  viewingCachedResult = false;
  activeHistoryId = null;
  lastDisplayedState = null;
  resultOutput.classList.add("hidden");
  resultOutput.innerHTML = "";
  emptyState.classList.remove("hidden");
  resultToolbar.classList.add("hidden");
  statsRow.classList.add("hidden");
  rerunBtn.classList.add("hidden");
  clearCitationHighlights();
  showWarnings([]);
  highlightActiveHistory(null);
}

async function runQuery({ force = false } = {}) {
  const options = currentRequestOptions();
  const query = options.query;

  if (query.length < 3) {
    showToast("Please enter at least 3 characters.");
    return;
  }

  if (!force) {
    const cached = findCachedEntry(options);
    if (cached?.result) {
      displayResult(cached.result, { fromCache: true, historyId: cached.id });
      showToast("Loaded saved result");
      return;
    }
  }

  cancelActiveRequest();
  snapshotBeforeRun = lastDisplayedState;
  activeAbortController = new AbortController();
  const { signal } = activeAbortController;
  activeRequestId = createRequestId();
  const requestId = activeRequestId;

  setLoading(true);
  showWarnings([]);
  clearCitationHighlights();
  viewingCachedResult = false;
  activeHistoryId = null;
  highlightActiveHistory(null);

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_id: requestId,
        query,
        citation_style: options.style,
        fast: options.fast,
      }),
      signal,
    });

    if (response.status === 499) {
      restoreAfterCancel();
      showToast("Request cancelled");
      return;
    }

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }

    const entry = buildHistoryEntry(options, data);
    saveHistory(entry);
    displayResult(entry.result, { fromCache: false, historyId: entry.id });
    savePreferences();
    showToast("Result ready");
  } catch (error) {
    if (error.name === "AbortError") {
      restoreAfterCancel();
      showToast("Request cancelled");
      return;
    }

    latestPlainResult = "";
    resultOutput.innerHTML = `<p class="error-text">Error: ${escapeHtml(error.message)}</p>`;
    resultOutput.classList.remove("hidden");
    emptyState.classList.add("hidden");
    rerunBtn.classList.add("hidden");
    showToast("Request failed");
  } finally {
    activeAbortController = null;
    activeRequestId = null;
    snapshotBeforeRun = null;
    setLoading(false);
  }
}

async function copyResult() {
  if (!latestPlainResult) return;
  await navigator.clipboard.writeText(latestPlainResult);
  showToast("Copied to clipboard");
}

function downloadResult() {
  if (!latestPlainResult) return;
  const blob = new Blob([latestPlainResult], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "research-assistant-result.txt";
  link.click();
  URL.revokeObjectURL(url);
  showToast("Download started");
}

async function downloadBundle() {
  if (!lastDisplayedState?.data) return;

  const { data } = lastDisplayedState;
  bundleBtn.disabled = true;

  try {
    const response = await fetch("/api/export/bundle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: data.query,
        answer: data.answer,
        citation_spans: data.citation_spans || [],
        citation_style: data.citation_style,
      }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Export failed (${response.status})`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "research-assistant-bundle.zip";
    link.click();
    URL.revokeObjectURL(url);
    showToast("Bundle download started");
  } catch (error) {
    showToast(error.message || "Bundle export failed");
  } finally {
    bundleBtn.disabled = false;
  }
}

function handleCancelRequest() {
  cancelActiveRequest();
}

submitBtn.addEventListener("click", () => runQuery());
cancelBtn.addEventListener("click", handleCancelRequest);
cancelLoadingBtn.addEventListener("click", handleCancelRequest);
rerunBtn.addEventListener("click", () => runQuery({ force: true }));
clearBtn.addEventListener("click", () => {
  queryInput.value = "";
  updateCharCount();
  savePreferences();
  clearDisplayedResult();
});

copyBtn.addEventListener("click", copyResult);
downloadBtn.addEventListener("click", downloadResult);
bundleBtn.addEventListener("click", downloadBundle);

removeSelectedBtn.addEventListener("click", () => {
  removeHistoryItems([...selectedHistoryIds]);
});

clearHistoryBtn.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  selectedHistoryIds.clear();
  activeHistoryId = null;
  loadHistory();
  clearDisplayedResult();
  showToast("History cleared");
});

historySelectAll.addEventListener("change", () => {
  const history = readHistory();
  if (historySelectAll.checked) {
    history.forEach((item) => selectedHistoryIds.add(item.id));
  } else {
    selectedHistoryIds.clear();
  }
  loadHistory();
});

helpBtn.addEventListener("click", () => {
  helpPanel.classList.toggle("hidden");
});

queryInput.addEventListener("input", () => {
  updateCharCount();
  savePreferences();
  if (activeCitationSpans.length) {
    queryInput.readOnly = false;
    clearCitationHighlights();
  }
});

queryBackdrop.addEventListener("click", (event) => {
  const mark = event.target.closest("mark.cite-span");
  if (!mark?.dataset.segmentId) return;
  event.preventDefault();
  showSegmentById(mark.dataset.segmentId);
});

queryInput.addEventListener("click", () => {
  if (!activeCitationSpans.length) return;
  window.setTimeout(() => {
    const position = queryInput.selectionStart ?? 0;
    showSegmentAtPosition(position);
  }, 0);
});

queryInput.addEventListener("keyup", () => {
  if (!activeCitationSpans.length) return;
  const position = queryInput.selectionStart ?? 0;
  showSegmentAtPosition(position);
});

queryInput.addEventListener("scroll", syncQueryBackdropScroll);

queryInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    runQuery();
  }
});

citationStyleSelect.addEventListener("change", savePreferences);
fastModeInput.addEventListener("change", savePreferences);

renderExamples();
loadCitationStyles();
checkHealth();
loadHistory();
