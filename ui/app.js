/* GenAIv2 UI - app.js
 *
 * Alinhado com ui/index.html e ui/styles.css fornecidos pelo utilizador.
 * Funcionalidades:
 * - Tools dinâmicas (GET /api/tools) com toggles
 * - Chat (POST /api/chat)
 * - Banner de erro e hints
 * - “Detalhes” com envelopes request/response por mensagem
 * - Checklist de steps (steps_executed)
 */

const messagesEl = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const refreshToolsBtn = document.getElementById("refreshToolsBtn");
const toolsListEl = document.getElementById("toolsList");
const bannerEl = document.getElementById("banner");

const sessionIdEl = document.getElementById("sessionId");
const requestIdEl = document.getElementById("requestId");
const latencyMsEl = document.getElementById("latencyMs");

// -----------------------------
// Banner helpers
// -----------------------------
function showBanner(msg) {
  if (!bannerEl) return;
  bannerEl.style.display = "block";
  bannerEl.textContent = msg;
}
function hideBanner() {
  if (!bannerEl) return;
  bannerEl.style.display = "none";
  bannerEl.textContent = "";
}

// -----------------------------
// Badges helpers
// -----------------------------
function setBadges({ sessionId, requestId, latencyMs }) {
  if (sessionIdEl && sessionId) sessionIdEl.textContent = sessionId;
  if (requestIdEl && requestId) requestIdEl.textContent = requestId;
  if (latencyMsEl && latencyMs != null) latencyMsEl.textContent = `${latencyMs}ms`;
}

// -----------------------------
// Tools
// -----------------------------
function getEnabledTools() {
  if (!toolsListEl) return [];
  const enabled = [];
  const inputs = toolsListEl.querySelectorAll("input[type=checkbox]");
  for (const cb of inputs) {
    if (cb.checked) enabled.push(cb.value);
  }
  return enabled;
}

async function loadTools() {
  hideBanner();

  try {
    const prevEnabled = new Set(getEnabledTools());

    const res = await fetch("/api/tools", { method: "GET" });
    if (!res.ok) {
      showBanner(`Falha ao obter tools (/api/tools): HTTP ${res.status}`);
      return;
    }

    const data = await res.json();
    const tools = data?.tools || [];

    if (!toolsListEl) return;
    toolsListEl.innerHTML = "";

    if (!tools.length) {
      const div = document.createElement("div");
      div.className = "toolRow";
      div.textContent = "Sem tools disponíveis (ou MCP indisponível).";
      toolsListEl.appendChild(div);
      return;
    }

    for (const t of tools) {
      const name = t?.name || "";
      const desc = t?.description || "";

      const row = document.createElement("label");
      row.className = "toolRow";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = name;
      cb.checked = prevEnabled.has(name); // preserva seleção ao refrescar

      const nameSpan = document.createElement("span");
      nameSpan.className = "toolName";
      nameSpan.textContent = name;

      const descSpan = document.createElement("span");
      descSpan.className = "toolDesc";
      descSpan.textContent = desc ? `— ${desc}` : "";

      row.appendChild(cb);
      row.appendChild(nameSpan);
      row.appendChild(descSpan);
      toolsListEl.appendChild(row);
    }
  } catch (e) {
    showBanner(`Erro ao carregar tools: ${String(e)}`);
  }
}

// -----------------------------
// Chat rendering
// -----------------------------
function clearChat() {
  if (!messagesEl) return;
  messagesEl.innerHTML = "";
  hideBanner();
}

function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerText = text;
  messagesEl.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

function addAssistantMessageShell() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";

  const primary = document.createElement("div");
  primary.className = "primary";
  primary.textContent = "";

  const metaLine = document.createElement("div");
  metaLine.className = "metaLine";
  metaLine.textContent = "";

  const checklist = document.createElement("div");
  checklist.className = "checklist";
  // Meta/Steps devem aparecer apenas dentro de Details (envelopes)
  metaLine.style.display = "none";
  checklist.style.display = "none";


  const details = document.createElement("details");
  details.className = "details";

  const summary = document.createElement("summary");
  summary.textContent = "Details (envelopes)";

  const pre = document.createElement("pre");
  pre.textContent = "";

  details.appendChild(summary);
  details.appendChild(pre);

  wrap.appendChild(primary);
  wrap.appendChild(metaLine);
  wrap.appendChild(checklist);
  wrap.appendChild(details);

  messagesEl.appendChild(wrap);
  wrap.scrollIntoView({ behavior: "smooth", block: "end" });

  return { wrap, primary, metaLine, checklist, pre };
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    }[c])
  );
}

function renderChecklist(checklistEl, responseEnvelope) {
  const payload = responseEnvelope?.payload || {};
  const executed = payload?.final_context?.steps_executed || [];
  if (!executed.length) return;

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

function formatDetailsEnvelopePair(requestEnv, responseEnv) {
  const mt = responseEnv?.metadata || {};
  const payload = responseEnv?.payload || {};
  const executed = payload?.final_context?.steps_executed || [];

  const metaLine = `source=${mt.source || "—"} · request_id=${mt.request_id || requestEnv?.metadata?.request_id || "—"} · session_id=${mt.session_id || requestEnv?.metadata?.session_id || "—"} · ts=${mt.timestamp || "—"}`;

  let stepsBlock = "Steps executed\n";
  if (!executed.length) {
    stepsBlock += "(none)\n";
  } else {
    for (const s of executed) {
      const sid = String(s?.id ?? "—");
      const st = String(s?.status ?? "—");
      const err = s?.error ? ` · error: ${String(s.error)}` : "";
      stepsBlock += `- [${sid}] ${st}${err}\n`;
    }
  }

  const jsonBlock = JSON.stringify({ request: requestEnv, response: responseEnv }, null, 2);

  return `${metaLine}\n\n${stepsBlock}\n---\n\n${jsonBlock}`;
}


// -----------------------------
// API call: /api/chat
// -----------------------------
async function sendMessage() {
  hideBanner();

  const text = (promptEl.value || "").trim();
  if (!text) return;

  const enabled_tools = getEnabledTools();

  // UI-side "request envelope" para debug local no painel Details
  const requestEnvelope = {
    metadata: {
      schema_version: "1.1",
      message_type: "request",
      request_id: null,
      timestamp: new Date().toISOString(),
      source: "ui",
      user_id: null,
      session_id: null,
      trace: null,
      timings_ms: null,
    },
    payload: { message: text, enabled_tools },
  };

  addUserMessage(text);
  promptEl.value = "";

  const { primary, metaLine, checklist, pre } = addAssistantMessageShell();
  primary.textContent = "A processar...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, enabled_tools }),
    });

    const latency = res.headers.get("x-latency-ms");
    const reqId = res.headers.get("x-request-id");
    const sessId = res.headers.get("x-session-id");

    setBadges({
      sessionId: sessId || "—",
      requestId: reqId || "—",
      latencyMs: latency != null ? Number(latency) : null,
    });

    requestEnvelope.metadata.request_id = reqId;
    requestEnvelope.metadata.session_id = sessId;

    const outEnv = await res.json();

    // meta line por mensagem
    //const mt = outEnv?.metadata || {};
    //metaLine.textContent = `source=${mt.source || "—"} · request_id=${mt.request_id || reqId || "—"} · session_id=${mt.session_id || sessId || "—"} · ts=${mt.timestamp || "—"}`;

    // erro API (HTTP != 2xx) ou envelope error
    const apiErr = outEnv?.payload?.error;
    if (!res.ok || apiErr) {
      const err = apiErr || { message: `HTTP ${res.status}`, code: "HTTP_ERROR", detail: null };

      primary.textContent = err.message || "Erro";
      pre.textContent = formatDetailsEnvelopePair(requestEnvelope, outEnv);

      // UX: TOOL_NOT_AVAILABLE -> mostrar available tools e sugerir refresh
      if (err.code === "TOOL_NOT_AVAILABLE") {
        const available = err?.detail?.available || [];
        if (available.length) {
          primary.textContent += `\n\nTools disponíveis: ${available.join(", ")}`;
        } else {
          primary.textContent += `\n\nSem lista de tools disponíveis no erro. Faz Refresh.`;
        }
        // Também ajuda a refrescar automaticamente
        await loadTools();
      }
      return;
    }

    // sucesso
    primary.textContent = outEnv?.payload?.answer || "";
    pre.textContent = formatDetailsEnvelopePair(requestEnvelope, outEnv);
    //renderChecklist(checklist, outEnv);
  } catch (e) {
    const errEnv = {
      metadata: {
        schema_version: "1.1",
        message_type: "error",
        request_id: requestEnvelope.metadata.request_id,
        timestamp: new Date().toISOString(),
        source: "ui",
        user_id: null,
        session_id: requestEnvelope.metadata.session_id,
        trace: null,
        timings_ms: null,
      },
      payload: { error: { code: "UI_ERROR", message: String(e), detail: null } },
    };

    primary.textContent = `Erro: ${String(e)}`;
    metaLine.textContent = `source=ui · request_id=${requestEnvelope.metadata.request_id || "—"} · session_id=${requestEnvelope.metadata.session_id || "—"}`;
    pre.textContent = formatDetailsEnvelopePair(requestEnvelope, errEnv);
    showBanner(`Erro de rede/JS: ${String(e)}`);
  }
}

// -----------------------------
// Event wiring
// -----------------------------
sendBtn?.addEventListener("click", sendMessage);

clearBtn?.addEventListener("click", clearChat);

refreshToolsBtn?.addEventListener("click", loadTools);

promptEl?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// init
loadTools();
try {
  promptEl?.focus();
} catch (_) {}
