/* GenAIv2 UI (API-first)
 *
 * - Chat via POST /api/chat
 * - Tools via GET /api/tools
 * - Sessions via GET /api/sessions + /api/sessions/{id}/messages + DELETE
 * - Envelopes trace via GET /api/requests/{request_id}/envelopes
 *
 * UX:
 * - No explicit role labels in the transcript ("assistant"/"user")
 * - Per-message Details panel (lazy-loaded) shows envelopes + steps executed
 */

const API_BASE = "/api";
const el = (id) => document.getElementById(id);

let messagesEl = null;
let bannerEl = null;
let promptEl = null;
let sendBtn = null;
let clearBtn = null;
let toolsListEl = null;
let refreshToolsBtn = null;

let sessionsListEl = null;
let reloadSessionsBtn = null;
let newChatBtn = null;
let deleteChatBtn = null;

let sessionPill = null;
let lastRequestPill = null;
let latencyPill = null;

let currentSessionId = null;
let lastRequestId = null;

// In-memory transcript for the current UI view
let chatItems = [];

function safeSetText(node, text) {
  if (!node) return;
  node.textContent = text;
}

function showBanner(kind, msg) {
  if (!bannerEl) return;
  bannerEl.style.display = "block";
  bannerEl.textContent = msg;
  bannerEl.dataset.kind = kind || "error";
}

function hideBanner() {
  if (!bannerEl) return;
  bannerEl.style.display = "none";
  bannerEl.textContent = "";
  bannerEl.dataset.kind = "";
}

function setCurrentSession(sessionId) {
  currentSessionId = sessionId;
  if (currentSessionId) {
    localStorage.setItem("genai.session_id", currentSessionId);
    safeSetText(sessionPill, `session: ${currentSessionId.slice(0, 8)}…`);
    if (deleteChatBtn) deleteChatBtn.disabled = false;
  } else {
    localStorage.removeItem("genai.session_id");
    safeSetText(sessionPill, "session: -");
    if (deleteChatBtn) deleteChatBtn.disabled = true;
  }

  // mark active in list
  if (sessionsListEl) {
    [...sessionsListEl.querySelectorAll(".sessionItem")].forEach((n) => {
      n.classList.toggle("active", n.dataset.sessionId === currentSessionId);
    });
  }
}

function setLastRequest(requestId, latencyMs) {
  lastRequestId = requestId || null;
  safeSetText(lastRequestPill, `last request: ${requestId ? requestId.slice(0, 8) + "…" : "-"}`);
  safeSetText(latencyPill, `latency: ${Number.isFinite(latencyMs) ? Math.round(latencyMs) + "ms" : "-"}`);
}

function getSelectedTools() {
  if (!toolsListEl) return [];
  const checked = [...toolsListEl.querySelectorAll('input[type="checkbox"]:checked')];
  return checked.map((c) => c.value);
}

// ---------- UUID helper (browser compatibility) ----------
// Some environments (older browsers / restricted contexts) do not expose crypto.randomUUID().
// Generate a RFC4122 v4 UUID using crypto.getRandomValues() as a safe fallback.
function generateUUID() {
  try {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
  } catch (_) {
    // fall through to getRandomValues
  }

  const cryptoObj = window.crypto;
  if (!cryptoObj || typeof cryptoObj.getRandomValues !== "function") {
    // Last resort: not cryptographically strong, but avoids hard failure.
    return `sid_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
  }

  const bytes = new Uint8Array(16);
  cryptoObj.getRandomValues(bytes);
  // Per RFC4122 section 4.4
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant

  const hex = [...bytes].map((b) => b.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10, 16).join("")}`;
}

// ---------- API ----------
async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  const ct = res.headers.get("content-type") || "";
  const isJson = ct.includes("application/json");
  const data = isJson ? await res.json().catch(() => null) : await res.text().catch(() => null);

  if (!res.ok) {
    const msg = (data && data.detail) ? data.detail : `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return { res, data };
}

// ---------- Envelope / details helpers ----------
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[c]));
}

function pickRequestResponseFromEnvelopes(envelopes) {
  const req = envelopes.find((e) => (e.stage === "request") || (e.envelope?.metadata?.message_type === "user_request"));
  const answer = [...envelopes].reverse().find((e) => {
    const mt = e.envelope?.metadata?.message_type;
    return e.stage === "answer" || mt === "answer" || mt === "response";
  });
  return { requestEnv: req?.envelope || null, responseEnv: answer?.envelope || null };
}

function extractStepsExecuted(responseEnv) {
  const payload = responseEnv?.payload || {};
  const fc = payload.final_context || payload.finalContext || payload.context || null;
  const steps = fc?.steps_executed || fc?.stepsExecuted || payload.steps_executed || payload.stepsExecuted || [];
  return Array.isArray(steps) ? steps : [];
}

function renderChecklist(checklistEl, responseEnv) {
  const executed = extractStepsExecuted(responseEnv);
  if (!checklistEl) return;
  if (!executed.length) {
    checklistEl.style.display = "none";
    checklistEl.innerHTML = "";
    return;
  }

  checklistEl.innerHTML = "";

  const title = document.createElement("div");
  title.className = "checklist-title";
  title.textContent = "Steps executed";
  checklistEl.appendChild(title);

  const ul = document.createElement("ul");
  ul.className = "checklist-items";

  executed.forEach((s) => {
    const li = document.createElement("li");

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.disabled = true;
    cb.checked = s.status === "success" || s.status === "skipped";

    const label = document.createElement("div");
    const sid = String(s.id || "");
    const st = String(s.status || "");
    const err = s.error ? ` · error: ${String(s.error)}` : "";

    label.innerHTML = `
      <div class="step-id">[${escapeHtml(sid)}]</div>
      <div class="step-desc">${escapeHtml(st)}${escapeHtml(err)}</div>
    `;

    li.appendChild(cb);
    li.appendChild(label);
    ul.appendChild(li);
  });

  checklistEl.appendChild(ul);
  checklistEl.style.display = "block";
}

function formatDetails(envelopes) {
  const { requestEnv, responseEnv } = pickRequestResponseFromEnvelopes(envelopes);
  const mt = responseEnv?.metadata || {};
  const rid = mt.request_id || requestEnv?.metadata?.request_id || "—";
  const sid = mt.session_id || requestEnv?.metadata?.session_id || "—";
  const src = mt.source || "—";
  const ts = mt.timestamp || "—";

  const metaLine = `source=${src} · request_id=${rid} · session_id=${sid} · ts=${ts}`;
  const jsonBlock = JSON.stringify({ envelopes }, null, 2);
  return `${metaLine}\n\n---\n\n${jsonBlock}`;
}

async function loadEnvelopesForRequest(requestId) {
  const { data } = await apiFetch(`/requests/${encodeURIComponent(requestId)}/envelopes?limit=200`);
  return data?.envelopes || [];
}

// ---------- Rendering ----------
function createMessageNode(item) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${item.role === "user" ? "user" : "bot"}`;

  const primary = document.createElement("div");
  primary.className = "primary";
  primary.textContent = item.content || "";

  const metaLine = document.createElement("div");
  metaLine.className = "metaLine";
  metaLine.style.display = "none";

  const details = document.createElement("details");
  details.className = "details";

  const summary = document.createElement("summary");
  summary.textContent = "Details (envelopes)";

  const pre = document.createElement("pre");
  pre.textContent = "";

  // Checklist is considered "Details" content; keep it inside the details element.
  const checklist = document.createElement("div");
  checklist.className = "checklist";
  checklist.style.display = "none";

  details.appendChild(summary);
  details.appendChild(checklist);
  details.appendChild(pre);

  details.addEventListener("toggle", async () => {
    if (!details.open) return;
    // Resolve envelopes either from the live chat response (preferred) or by lazy loading from the API.
    let envs = Array.isArray(item.envelopes) ? item.envelopes : null;
    if (!envs || envs.length === 0) {
      if (!item.request_id) {
        pre.textContent = "(no request_id available for this message)";
        return;
      }
      try {
        envs = await loadEnvelopesForRequest(item.request_id);
      } catch (e) {
        pre.textContent = `Failed to load envelopes: ${e.message}`;
        return;
      }
    }

    const { responseEnv } = pickRequestResponseFromEnvelopes(envs);
    if (!pre.textContent || pre.textContent.trim().length === 0) {
      pre.textContent = formatDetails(envs);
    }
    renderChecklist(checklist, responseEnv);
  });

  wrap.appendChild(primary);
  wrap.appendChild(metaLine);
  wrap.appendChild(details);

  // If we already have envelopes from live chat, pre-fill
  if (item.envelopes && item.envelopes.length) {
    pre.textContent = formatDetails(item.envelopes);
    // Keep details collapsed by default; render checklist only when Details is opened.
    // This prevents "Steps executed" from appearing in the main transcript.
    // (Still available inside Details.)
  }

  return wrap;
}

function renderMessages() {
  if (!messagesEl) return;
  messagesEl.innerHTML = "";
  for (const item of chatItems) {
    messagesEl.appendChild(createMessageNode(item));
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function pushMessage(item) {
  chatItems.push(item);
  renderMessages();
}

function clearLocalChat() {
  chatItems = [];
  renderMessages();
}

// ---------- Tools ----------
async function loadTools() {
  if (!toolsListEl) return;
  const { data } = await apiFetch("/tools", { method: "GET" });
  const tools = data?.tools || [];

  const prev = new Set(getSelectedTools());
  toolsListEl.innerHTML = "";

  for (const t of tools) {
    const row = document.createElement("label");
    row.className = "toolRow";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = t.name;
    cb.checked = prev.has(t.name);

    const name = document.createElement("div");
    name.className = "toolName";
    name.textContent = t.name;

    const desc = document.createElement("div");
    desc.className = "toolDesc";
    desc.textContent = t.description || "";

    const col = document.createElement("div");
    col.appendChild(name);
    col.appendChild(desc);

    row.appendChild(cb);
    row.appendChild(col);
    toolsListEl.appendChild(row);
  }
}

// ---------- Sessions ----------
function renderSessionsList(sessions) {
  if (!sessionsListEl) return;
  sessionsListEl.innerHTML = "";
  for (const s of sessions) {
    const item = document.createElement("div");
    item.className = "sessionItem";
    item.dataset.sessionId = s.session_id;

    const title = document.createElement("div");
    title.className = "sessionTitle";
    title.textContent = s.session_id;

    const meta = document.createElement("div");
    meta.className = "sessionMeta";
    meta.textContent = `messages: ${s.message_count || 0} · last: ${s.last_seen_at || "-"}`;

    item.appendChild(title);
    item.appendChild(meta);

    item.addEventListener("click", async () => {
      setCurrentSession(s.session_id);
      await loadMessagesForCurrentSession();
    });

    sessionsListEl.appendChild(item);
  }
}

async function loadSessions() {
  const { data } = await apiFetch("/sessions?limit=50&offset=0");
  const sessions = data?.sessions || [];
  renderSessionsList(sessions);
  // restore active marker
  setCurrentSession(currentSessionId);
}

async function loadMessagesForCurrentSession() {
  if (!currentSessionId) {
    chatItems = [];
    renderMessages();
    return;
  }
  const { data } = await apiFetch(`/sessions/${encodeURIComponent(currentSessionId)}/messages?limit=500`);
  const msgs = data?.messages || [];
  chatItems = msgs.map((m) => ({
    role: m.role === "assistant" ? "assistant" : "user",
    content: m.content || "",
    request_id: m.request_id || null,
    envelopes: null,
  }));
  renderMessages();
}

async function createNewSession() {
  hideBanner();
  const sid = generateUUID();
  setCurrentSession(sid);
  chatItems = [];
  renderMessages();
  setLastRequest(null, null);
}

async function deleteCurrentSession() {
  if (!currentSessionId) return;
  try {
    await apiFetch(`/sessions/${encodeURIComponent(currentSessionId)}`, { method: "DELETE" });
    await loadSessions();
    await createNewSession();
  } catch (e) {
    showBanner("error", `Falha ao apagar sessão: ${e.message}`);
  }
}

// ---------- Chat ----------
async function sendChat() {
  hideBanner();
  const msg = (promptEl?.value || "").trim();
  if (!msg) return;

  if (!currentSessionId) {
    await createNewSession();
  }

  pushMessage({ role: "user", content: msg, request_id: null, envelopes: null });
  if (promptEl) promptEl.value = "";

  const payload = {
    message: msg,
    enabled_tools: getSelectedTools(),
  };

  const t0 = performance.now();
  try {
    const { res, data } = await apiFetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-session-id": currentSessionId || "",
      },
      body: JSON.stringify(payload),
    });

    const sid = res.headers.get("x-session-id");
    if (sid && sid !== currentSessionId) setCurrentSession(sid);

    const t1 = performance.now();

    const answer =
      data?.response?.payload?.answer ??
      data?.response?.payload?.message ??
      data?.response?.payload?.text ??
      data?.answer ??
      data?.message ??
      data?.text ??
      "";

    const reqId = data?.response?.metadata?.request_id || data?.request?.metadata?.request_id || null;

    // Pre-fill envelopes for this message so Details works immediately (no extra API call)
    const envelopes = [];
    if (data?.request) envelopes.push({ stage: "request", envelope: data.request });
    if (data?.response) envelopes.push({ stage: "answer", envelope: data.response });

    pushMessage({ role: "assistant", content: answer, request_id: reqId, envelopes });
    setLastRequest(reqId, t1 - t0);

    try { await loadSessions(); } catch (_) {}
  } catch (e) {
    showBanner("error", `Erro: ${e.message}`);
  }
}

// ---------- Init ----------
function bindDom() {
  messagesEl = el("messages");
  bannerEl = el("banner");
  promptEl = el("prompt");
  sendBtn = el("sendBtn");
  clearBtn = el("clearBtn");

  toolsListEl = el("toolsList");
  refreshToolsBtn = el("refreshToolsBtn");

  sessionPill = el("sessionId");
  lastRequestPill = el("requestId");
  latencyPill = el("latencyMs");

  sessionsListEl = el("sessionsList");
  reloadSessionsBtn = el("reloadSessionsBtn");
  newChatBtn = el("newChatBtn");
  deleteChatBtn = el("deleteChatBtn");

  sendBtn?.addEventListener("click", sendChat);
  clearBtn?.addEventListener("click", clearLocalChat);

  promptEl?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      sendChat();
    }
  });

  refreshToolsBtn?.addEventListener("click", async () => {
    try { await loadTools(); } catch (e) { showBanner("warn", `Tools indisponíveis: ${e.message}`); }
  });

  reloadSessionsBtn?.addEventListener("click", async () => {
    try { await loadSessions(); } catch (e) { showBanner("warn", `Falha ao listar sessões: ${e.message}`); }
  });

  newChatBtn?.addEventListener("click", createNewSession);
  deleteChatBtn?.addEventListener("click", deleteCurrentSession);
}

(async function init() {
  bindDom();

  const persisted = localStorage.getItem("genai.session_id");
  if (persisted) setCurrentSession(persisted);
  else setCurrentSession(null);

  try { await loadTools(); } catch (e) { showBanner("warn", `Tools indisponíveis: ${e.message}`); }
  try { await loadSessions(); } catch (e) { showBanner("warn", `Falha ao listar sessões: ${e.message}`); }

  if (currentSessionId) {
    try { await loadMessagesForCurrentSession(); } catch (e) { /* ignore */ }
  }
})();
