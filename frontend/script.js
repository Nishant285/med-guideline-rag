const thread = document.getElementById("thread");
const composer = document.getElementById("composer");
const input = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

// Auto-grow the textarea as the user types, up to a max height (set in CSS).
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
});

// Enter submits the question; Shift+Enter still inserts a newline, in case
// someone wants to write a multi-part question.
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addUserMessage(question);
  input.value = "";
  input.style.height = "auto";
  setLoading(true);

  const loadingEl = addLoadingMessage();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    loadingEl.remove();
    addAssistantMessage(data.answer, data.sources);
  } catch (err) {
    loadingEl.remove();
    addErrorMessage(err.message || "Something went wrong.");
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  sendBtn.disabled = isLoading;
  input.disabled = isLoading;
}

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "message user";
  el.innerHTML = `<div class="bubble"></div>`;
  el.querySelector(".bubble").textContent = text;
  thread.appendChild(el);
  scrollToBottom();
}

function addLoadingMessage() {
  const el = document.createElement("div");
  el.className = "loading";
  el.textContent = "Searching guidelines and drafting a grounded answer...";
  thread.appendChild(el);
  scrollToBottom();
  return el;
}

function addErrorMessage(message) {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = `
    <p class="role-label">WHO Guideline Assistant</p>
    <p class="answer-text error-text"></p>
  `;
  el.querySelector(".error-text").textContent = message;
  thread.appendChild(el);
  scrollToBottom();
}

/**
 * Renders the assistant's answer, converting every
 * "[source: filename.pdf, page 12]" citation the model wrote into a
 * numbered footnote marker, deduplicated so repeated citations of the
 * same chunk share one footnote number - mirroring how a real printed
 * guideline's footnotes work.
 */
function addAssistantMessage(answerText, sources) {
  const el = document.createElement("div");
  el.className = "message assistant";

  const roleLabel = document.createElement("p");
  roleLabel.className = "role-label";
  roleLabel.textContent = "WHO Guideline Assistant";
  el.appendChild(roleLabel);

  const answerEl = document.createElement("div");
  answerEl.className = "answer-text";

  const citationPattern = /[\[【]\s*source:\s*([^,]+),\s*page\s*(\d+)\s*[\]】]/gi;
  const footnoteKeyToNumber = new Map();
  const footnoteEntries = [];

  // Strip stray markdown bold/italic markers in case the model adds them
  // despite instructions not to - keeps the reading pane looking clean
  // regardless of small model formatting slips.
  const cleanedText = answerText.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\*(.*?)\*/g, "$1");

  const renderedText = cleanedText.replace(citationPattern, (match, file, page) => {
    const key = `${file.trim()}|${page}`;
    if (!footnoteKeyToNumber.has(key)) {
      footnoteKeyToNumber.set(key, footnoteEntries.length + 1);
      const matchingChunk = sources.find(
        (s) => s.source === file.trim() && String(s.page) === page
      );
      footnoteEntries.push({
        file: file.trim(),
        page,
        excerpt: matchingChunk ? matchingChunk.text.slice(0, 220) + "..." : null,
      });
    }
    const num = footnoteKeyToNumber.get(key);
    return `<sup class="footnote-marker" data-footnote="${num}">[${num}]</sup>`;
  });

  // Paragraph-split on blank lines for basic readable formatting.
  const paragraphs = renderedText
    .split(/\n\s*\n/)
    .map((p) => `<p>${p.trim()}</p>`)
    .join("");
  answerEl.innerHTML = paragraphs;
  el.appendChild(answerEl);

  if (footnoteEntries.length > 0) {
    const footnotesEl = document.createElement("div");
    footnotesEl.className = "footnotes";
    footnotesEl.innerHTML = footnoteEntries
      .map((f, i) => {
        const excerptHtml = f.excerpt
          ? `<span class="footnote-excerpt"></span>`
          : "";
        return `
          <div class="footnote-entry">
            <span class="footnote-num">[${i + 1}]</span>
            <span>
              <strong></strong>, page ${f.page}
              ${excerptHtml}
            </span>
          </div>
        `;
      })
      .join("");

    // Fill in text content via DOM (not innerHTML) to avoid any injection
    // risk from PDF text content.
    footnotesEl.querySelectorAll(".footnote-entry").forEach((entryEl, i) => {
      entryEl.querySelector("strong").textContent = footnoteEntries[i].file;
      const excerptEl = entryEl.querySelector(".footnote-excerpt");
      if (excerptEl) excerptEl.textContent = footnoteEntries[i].excerpt;
    });

    el.appendChild(footnotesEl);
  }

  thread.appendChild(el);
  scrollToBottom();
}

function scrollToBottom() {
  thread.scrollTop = thread.scrollHeight;
}
