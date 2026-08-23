const form = document.getElementById("scan-form");
const input = document.getElementById("url-input");
const btn = document.getElementById("scan-btn");
const resultBox = document.getElementById("result");
const historyList = document.getElementById("history-list");

const VERDICT_EMOJI = { Safe: "✅", Suspicious: "⚠️", Scam: "🚨" };

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function timeAgo(iso) {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.round(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return new Date(iso).toLocaleDateString();
}

function renderResult(data) {
  resultBox.classList.remove("hidden");

  if (data.error) {
    resultBox.innerHTML = `<p class="error-box">😕 ${escapeHtml(data.error)}</p>`;
    return;
  }

  const emoji = VERDICT_EMOJI[data.verdict] || "";
  const reasonsHtml = data.reasons.map(r => `<li>${escapeHtml(r)}</li>`).join("");
  const websiteChip = data.website_probability !== null
    ? `<div class="score-chip">Website analysis<strong>${data.website_probability}%</strong></div>` : "";
  const durationHtml = data.duration_seconds !== undefined
    ? `<span class="duration-tag">⏱ ${data.duration_seconds}s</span>` : "";

  resultBox.innerHTML = `
    <div class="verdict-row">
      <span class="badge ${data.verdict}">${emoji} ${data.verdict}</span>
      <span class="risk-percent">${data.risk_percent}% risk</span>
      ${durationHtml}
    </div>
    <div class="meter"><div class="meter-fill ${data.verdict}" style="width:${data.risk_percent}%"></div></div>
    <div class="scores">
      <div class="score-chip">ML model (${escapeHtml(data.ml_model)})<strong>${data.ml_probability}%</strong></div>
      <div class="score-chip">Deep learning (CNN)<strong>${data.dl_probability}%</strong></div>
      ${websiteChip}
    </div>
    <p class="reasons-title">Why</p>
    <ul class="reasons">${reasonsHtml}</ul>
  `;
}

async function loadHistory() {
  const res = await fetch("/api/history");
  const rows = await res.json();

  if (!rows.length) {
    historyList.innerHTML = `<p class="empty-state">No scans yet — try one above 👆</p>`;
    return;
  }

  historyList.innerHTML = rows.map(r => `
    <div class="history-row">
      <span class="history-badge ${r.verdict}">${VERDICT_EMOJI[r.verdict] || ""} ${escapeHtml(r.verdict)}</span>
      <span class="history-url" title="${escapeHtml(r.url)}">${escapeHtml(r.url)}</span>
      <span class="history-risk">${r.risk_percent}%</span>
      ${r.duration_seconds !== null && r.duration_seconds !== undefined
        ? `<span class="history-duration">⏱ ${r.duration_seconds}s</span>` : ""}
      <span class="history-time">${timeAgo(r.scanned_at)}</span>
    </div>
  `).join("");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = input.value.trim();
  if (!url) return;

  btn.disabled = true;
  btn.querySelector("span").textContent = "Scanning...";
  resultBox.classList.remove("hidden");
  resultBox.innerHTML = `<p class="loading">🔎 Analyzing URL, page content, and models...</p>`;

  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    renderResult(data);
    loadHistory();
  } catch (err) {
    resultBox.innerHTML = `<p class="error-box">😕 Request failed: ${escapeHtml(String(err))}</p>`;
  } finally {
    btn.disabled = false;
    btn.querySelector("span").textContent = "Scan";
  }
});

loadHistory();
