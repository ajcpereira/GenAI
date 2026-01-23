# ui/ui.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("", response_class=HTMLResponse)
async def ui_index():
    # Minimal ChatGPT-like UI:
    # - One input box
    # - "Send" button
    # - Main response shown
    # - Details collapsible (metadata + payload JSON)
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>GenAI UI</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#0b0f14; color:#e6edf3;}
    .wrap { max-width: 980px; margin: 0 auto; padding: 24px; }
    .card { background:#111826; border:1px solid #1f2a37; border-radius: 14px; padding: 16px; margin-bottom: 16px; }
    .row { display:flex; gap: 12px; }
    textarea { width: 100%; min-height: 60px; resize: vertical; background:#0b1220; color:#e6edf3; border:1px solid #233041; border-radius: 10px; padding: 12px; }
    button { background:#2563eb; color:white; border:none; border-radius: 10px; padding: 12px 16px; cursor:pointer; }
    button:disabled { opacity: .6; cursor:not-allowed; }
    .muted { color:#9aa4b2; font-size: 13px; }
    pre { white-space: pre-wrap; word-break: break-word; background:#0b1220; border:1px solid #233041; border-radius: 10px; padding: 12px; overflow:auto; }
    details > summary { cursor:pointer; }
    .checklist { margin: 10px 0 0; padding-left: 18px; }
    .checklist li { margin: 6px 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2 style="margin: 0 0 12px;">GenAI UI</h2>
    <div class="muted" style="margin-bottom: 16px;">
      API endpoint: <code>/api/chat</code>. Response shows main content, expand Details for full JSON.
    </div>

    <div class="card">
      <div class="row">
        <textarea id="msg" placeholder="Type a message..."></textarea>
        <div style="display:flex; flex-direction:column; gap: 8px;">
          <button id="send">Send</button>
          <button id="clear" style="background:#374151;">Clear</button>
        </div>
      </div>
      <div class="muted" style="margin-top:8px;">
        Tip: use <code>x-request-id</code> via API; UI auto-generates one.
      </div>
    </div>

    <div id="out"></div>
  </div>

<script>
const out = document.getElementById("out");
const btn = document.getElementById("send");
const clr = document.getElementById("clear");
const msg = document.getElementById("msg");

function uuidv4() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
    const r = Math.random()*16|0, v = c === "x" ? r : (r&0x3|0x8);
    return v.toString(16);
  });
}

function esc(s){ return (s ?? "").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

function renderPlanSteps(plan){
  const steps = plan?.steps || [];
  if (!Array.isArray(steps) || steps.length === 0) return "";
  const items = steps.map(s => `<li><input type="checkbox" disabled /> <strong>${esc(s.id)}</strong> — ${esc(s.description)}</li>`).join("");
  return `<div class="muted" style="margin-top:10px;">Steps</div><ul class="checklist">${items}</ul>`;
}

function renderEnvelope(env){
  const intent = env?.payload?.user_intent?.summary || "";
  const plan = env?.payload?.plan || null;

  const main = `
    <div class="card">
      <div class="muted">Intent</div>
      <div>${esc(intent) || "<span class='muted'>(no intent)</span>"}</div>
      ${plan ? renderPlanSteps(plan) : ""}
      <details style="margin-top:12px;">
        <summary>Details (expand)</summary>
        <div class="muted" style="margin-top:8px;">Full envelope JSON</div>
        <pre>${esc(JSON.stringify(env, null, 2))}</pre>
      </details>
    </div>
  `;
  return main;
}

async function send(){
  const text = msg.value.trim();
  if (!text) return;

  btn.disabled = true;
  const rid = uuidv4();

  try{
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-request-id": rid
      },
      body: JSON.stringify({ message: text })
    });

    const data = await resp.json().catch(() => null);

    if (!resp.ok){
      out.insertAdjacentHTML("afterbegin", `
        <div class="card">
          <div><strong>Request failed</strong> (${resp.status})</div>
          <details style="margin-top:12px;">
            <summary>Details (expand)</summary>
            <pre>${esc(JSON.stringify(data ?? {error:"no json"}, null, 2))}</pre>
          </details>
        </div>
      `);
      return;
    }

    out.insertAdjacentHTML("afterbegin", renderEnvelope(data));
    msg.value = "";
  } finally {
    btn.disabled = false;
  }
}

btn.addEventListener("click", send);
msg.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") send();
});
clr.addEventListener("click", () => { out.innerHTML = ""; msg.value=""; msg.focus(); });
</script>

</body>
</html>
        """.strip()
    )


@router.get("/healthz")
async def ui_healthz():
    return {"status": "ok"}
