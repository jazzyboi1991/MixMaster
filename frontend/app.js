const API_BASE = ""; // same origin as the FastAPI server serving this page
const CHAT_ENDPOINT = `${API_BASE}/api/chat`;
const CHAT_STREAM_ENDPOINT = `${API_BASE}/api/chat/stream`;
const HEALTH_ENDPOINT = `${API_BASE}/health`;

// 브라우저 탭별 고유 대화 세션 구분을 위해 threadId 생성
const threadId = "session_" + Math.random().toString(36).substring(2, 11);

const root = document.documentElement;
const landingView = document.getElementById("landingView");
const consoleView = document.getElementById("consoleView");
const landingForm = document.getElementById("landingForm");
const landingInput = document.getElementById("landingInput");
const consoleForm = document.getElementById("consoleForm");
const consoleInput = document.getElementById("consoleInput");
const consoleSubmit = document.getElementById("consoleSubmit");
const chatLog = document.getElementById("chatLog");
const progressLine = document.getElementById("progressLine");
const statusPanel = document.getElementById("statusPanel");
const sourcePanel = document.getElementById("sourcePanel");
const sourceList = document.getElementById("sourceList");
const metricInstrument = document.getElementById("metricInstrument");
const metricProblem = document.getElementById("metricProblem");
const stepList = document.getElementById("stepList");
const engineStatus = document.getElementById("engineStatus");

let inFlight = false;

/* ---------- Ambient background, translated from gradient.glsl ---------- */
function animateAmbient(timestampMs) {
  const syncTime = (timestampMs * 0.001) * 0.7;
  const x = 50 + 28 * Math.cos(syncTime);
  const y = 50 + 18 * Math.sin(syncTime);
  root.style.setProperty("--glow-x", `${x}%`);
  root.style.setProperty("--glow-y", `${y}%`);
  requestAnimationFrame(animateAmbient);
}
requestAnimationFrame(animateAmbient);


/* ---------- Health check ---------- */
async function checkHealth() {
  try {
    const res = await fetch(HEALTH_ENDPOINT);
    if (!res.ok) throw new Error("unhealthy");
    engineStatus.textContent = "RAG Engine Active";
    engineStatus.classList.remove("offline");
  } catch {
    engineStatus.textContent = "RAG Engine Offline";
    engineStatus.classList.add("offline");
  }
}
checkHealth();

/* ---------- Chat rendering ---------- */
function addBubble({ role, name, body, variant }) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}${variant ? ` ${variant}` : ""}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = name;

  const bodyEl = document.createElement("div");
  bodyEl.className = "body";
  bodyEl.textContent = body;

  bubble.append(roleEl, bodyEl);
  chatLog.appendChild(bubble);
  bubble.scrollIntoView({ behavior: "smooth", block: "end" });
  return bubble;
}

const STEP_ORDER = ["analyze", "retrieve", "verify"];

function resetSteps() {
  stepList.querySelectorAll(".step").forEach((el) => {
    el.classList.remove("done", "current", "pulse", "reveal");
  });
}

// Parses a raw SSE text chunk buffer into complete {event, data} records,
// returning the leftover partial buffer so callers can keep appending to it.
function parseSSEBuffer(buffer) {
  const events = [];
  const rawEvents = buffer.split("\n\n");
  const leftover = rawEvents.pop() ?? "";

  for (const raw of rawEvents) {
    if (!raw.trim()) continue;
    let eventName = "message";
    let dataLines = [];
    for (const line of raw.split("\n")) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }
    events.push({ event: eventName, data: dataLines.join("\n") });
  }

  return { events, leftover };
}

// Applies the real backend step states as they stream in. The stage right
// after the last completed one gets a "pulse" to show it's actively running;
// completed stages get a one-shot "reveal" slide-in the moment they land.
function updateSteps(steps = []) {
  const stageDone = {
    analyze: steps.includes("analyze_query"),
    retrieve: steps.includes("call_model"),
    verify: steps.includes("call_model") || steps.includes("ask_clarification"),
  };
  let lastDoneIndex = -1;
  STEP_ORDER.forEach((key, i) => {
    if (stageDone[key]) lastDoneIndex = i;
  });

  STEP_ORDER.forEach((key, i) => {
    const el = stepList.querySelector(`[data-step="${key}"]`);
    const justCompleted = stageDone[key] && !el.classList.contains("done");
    el.classList.toggle("done", stageDone[key]);
    el.classList.toggle("pulse", i === lastDoneIndex + 1);
    el.classList.remove("current");
    if (justCompleted) {
      el.classList.remove("reveal");
      void el.offsetWidth; // restart the animation even if classes are unchanged
      el.classList.add("reveal");
    }
  });
}

function markAllStepsSettled() {
  stepList.querySelectorAll(".step").forEach((el) => el.classList.remove("pulse"));
}

function updateDetailsPanel(result) {
  const { collected_details: details = {}, sources = [], steps = [] } = result;
  updateSteps(steps);

  const hasSolutionInfo = sources.length > 0 || (details.instrument && details.instrument !== "None");

  if (hasSolutionInfo) {
    metricInstrument.textContent = details.instrument && details.instrument !== "None" ? details.instrument : "—";
    metricProblem.textContent = details.problem_type && details.problem_type !== "None" ? details.problem_type : "—";
    statusPanel.hidden = false;
  } else {
    statusPanel.hidden = true;
  }

  if (sources.length > 0) {
    sourceList.innerHTML = "";
    sources.forEach((src) => {
      const item = document.createElement("div");
      item.className = "source-item";
      item.textContent = `• ${src}`;
      sourceList.appendChild(item);
    });
    sourcePanel.hidden = false;
  } else {
    sourcePanel.hidden = true;
  }
}

/* ---------- Backend call ---------- */
async function askMixMaster(query) {
  if (inFlight) return;
  inFlight = true;
  consoleSubmit.disabled = true;
  progressLine.hidden = false;
  resetSteps();

  addBubble({ role: "user", name: "Producer", body: query });

  try {
    const res = await fetch(CHAT_STREAM_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, thread_id: threadId }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `서버 오류 (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result = null;
    let streamError = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const { events, leftover } = parseSSEBuffer(buffer);
      buffer = leftover;

      for (const { event, data } of events) {
        if (!data) continue;
        const parsed = JSON.parse(data);
        if (event === "step") {
          updateSteps(parsed.steps);
        } else if (event === "final") {
          result = parsed;
        } else if (event === "error") {
          streamError = parsed.detail;
        }
      }
    }

    if (streamError) throw new Error(streamError);
    if (!result) throw new Error("서버로부터 최종 응답을 받지 못했습니다.");

    markAllStepsSettled();

    if (result.clarification_needed) {
      addBubble({
        role: "ai",
        name: "MixMaster Assistant",
        body: result.response,
      });
    } else {
      addBubble({
        role: "ai",
        name: "MixMaster Assistant (RAG Verified)",
        body: result.response,
        variant: "solution",
      });
    }

    updateDetailsPanel(result);
  } catch (err) {
    markAllStepsSettled();
    addBubble({
      role: "ai",
      name: "MixMaster Assistant",
      body: `죄송합니다, 요청을 처리하는 중 문제가 발생했습니다: ${err.message}`,
      variant: "error",
    });
  } finally {
    progressLine.hidden = true;
    consoleSubmit.disabled = false;
    inFlight = false;
  }
}

/* ---------- View transition ---------- */
function enterConsole(firstQuery) {
  resetSteps();

  // Reveal zPm6I as a fixed overlay parked below the viewport, then let it
  // rise up over the ScjNR landing view.
  consoleView.hidden = false;
  landingView.hidden = true;
  void consoleView.offsetHeight; // force reflow so the transform below animates
  requestAnimationFrame(() => {
    consoleView.classList.add("rising");
  });

  addBubble({
    role: "ai",
    name: "MixMaster Assistant",
    body: "안녕하세요. 작업 중이신 오디오 트랙의 믹싱 및 마스터링 고민을 입력해 주세요. RAG와 대화형 지식 분석을 기반으로 최적의 플러그인 체인과 세팅 가이드를 제안해 드립니다.",
  });

  askMixMaster(firstQuery);
  consoleInput.focus();
}

/* ---------- Event wiring ---------- */
landingForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = landingInput.value.trim();
  if (!query) return;
  landingInput.value = "";
  enterConsole(query);
});

consoleForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = consoleInput.value.trim();
  if (!query || inFlight) return;
  consoleInput.value = "";
  askMixMaster(query);
});
