(function () {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const loadForm = byId("load-form");
  const videoInput = byId("video-input");
  const loadButton = byId("load-btn");
  const loadLabel = byId("load-btn-label");
  const videoStatus = byId("video-status");
  const toggleManual = byId("toggle-manual");
  const manualPanel = byId("manual-transcript-panel");
  const manualForm = byId("manual-form");
  const transcriptInput = byId("transcript-input");
  const manualSubmit = byId("manual-submit");
  const manualSubmitLabel = byId("manual-submit-label");
  const videoPreview = byId("video-preview");
  const videoThumb = byId("video-thumb");
  const videoIdLabel = byId("video-id-label");
  const videoMeta = byId("video-meta");
  const openVideoLink = byId("open-video-link");
  const chatForm = byId("chat-form");
  const questionInput = byId("question-input");
  const sendButton = byId("send-btn");
  const sendLabel = byId("send-btn-label");
  const chatMessages = byId("chat-messages");
  const welcomeState = byId("welcome-state");
  const typingIndicator = byId("typing-indicator");
  const chatStatus = byId("chat-status");
  const chatSubtitle = byId("chat-subtitle");
  const suggestionButtons = Array.from(document.querySelectorAll("[data-question]"));

  let activeVideoId = null;
  let loadingVideo = false;
  let sendingQuestion = false;

  function setStatus(message, type) {
    videoStatus.textContent = message;
    videoStatus.className = "inline-status" + (type ? " " + type : "");
    videoStatus.hidden = !message;
  }

  function setLoadingVideo(loading) {
    loadingVideo = loading;
    loadButton.disabled = loading;
    loadButton.classList.toggle("is-loading", loading);
    loadLabel.textContent = loading ? "Indexing transcript…" : "Index video →";
    if (loading) {
      videoStatus.textContent = "Fetching captions, creating timestamped chunks and building the search index. Long videos can take a moment.";
      videoStatus.className = "inline-status";
      videoStatus.hidden = false;
    }
  }

  function setSendingQuestion(sending) {
    sendingQuestion = sending;
    sendButton.disabled = sending || !activeVideoId;
    sendButton.classList.toggle("is-loading", sending);
    sendLabel.textContent = sending ? "Thinking…" : "Ask TubeGPT";
    typingIndicator.hidden = !sending;
    if (sending) {
      chatStatus.textContent = "Retrieving evidence";
      chatStatus.parentElement.classList.add("is-ready");
    } else {
      chatStatus.textContent = activeVideoId ? "Index ready" : "Awaiting video";
    }
    scrollToBottom();
  }

  async function requestJson(url, payload) {
    let response;
    try {
      response = await fetch(url, {
        method: payload ? "POST" : "GET",
        headers: payload ? { "Content-Type": "application/json" } : {},
        body: payload ? JSON.stringify(payload) : undefined,
        credentials: "same-origin"
      });
    } catch (_error) {
      throw new Error("Could not reach the server. Check that the app is running and try again.");
    }

    let data;
    try {
      data = await response.json();
    } catch (_error) {
      throw new Error("The server returned an unreadable response. Check the app logs.");
    }
    if (!response.ok) {
      throw new Error(data && data.error ? data.error : "The request failed. Please try again.");
    }
    return data;
  }

  function createElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function appendMessage(role, text, sources, isError, citationAudit) {
    if (welcomeState && welcomeState.isConnected) welcomeState.hidden = true;

    const isUser = role === "user";
    const message = createElement("article", "message" + (isUser ? " user" : "") + (isError ? " message-error" : ""));
    const avatar = createElement("div", "message-avatar", isUser ? "Y" : "✳");
    avatar.setAttribute("aria-hidden", "true");
    const body = createElement("div", "message-body");
    const author = createElement("p", "message-author", isUser ? "YOU" : "TUBEGPT");
    const bubble = createElement("div", "message-bubble", text || "");
    body.append(author, bubble);

    if (!isUser && citationAudit && typeof citationAudit.valid === "boolean") {
      const verified = citationAudit.valid;
      const label = verified
        ? "Source labels checked"
        : "Source-label check needs review";
      const note = verified
        ? "Every citation label used by the answer exists in the retrieved source list. This checks label integrity, not whether a claim is factually supported."
        : "Some citation labels are missing or unknown. Verify the answer against the evidence cards before relying on it.";
      const audit = createElement("div", "citation-audit " + (verified ? "valid" : "warning"));
      audit.appendChild(createElement("span", "citation-audit-icon", verified ? "✓" : "⚠"));
      const copy = createElement("span", "citation-audit-copy");
      copy.append(
        createElement("strong", "", label),
        createElement("small", "", note)
      );
      audit.appendChild(copy);
      body.appendChild(audit);
    }

    if (!isUser && Array.isArray(sources) && sources.length) {
      body.appendChild(createElement("div", "source-heading", "Retrieved transcript evidence"));
      const sourceList = createElement("div", "source-list");
      sources.forEach((source) => sourceList.appendChild(createSourceCard(source)));
      body.appendChild(sourceList);
    }

    message.append(avatar, body);
    chatMessages.appendChild(message);
    scrollToBottom();
    return message;
  }

  function createSourceCard(source) {
    const start = Number.isFinite(Number(source.start_seconds)) ? Math.max(0, Math.floor(Number(source.start_seconds))) : 0;
    const safeVideoId = activeVideoId || "";
    const expectedPrefix = "https://www.youtube.com/watch?v=";
    const href = typeof source.youtube_url === "string" && source.youtube_url.startsWith(expectedPrefix)
      ? source.youtube_url
      : expectedPrefix + encodeURIComponent(safeVideoId) + "&t=" + start + "s";

    const link = createElement("a", "source-card");
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.setAttribute("aria-label", "Open source " + (source.id || "") + " at " + (source.start_timestamp || "timestamp"));

    const tag = createElement("span", "source-tag", source.id || "SRC");
    const content = createElement("span", "source-content");
    const meta = createElement("span", "source-meta");
    const timingLabel = source.timestamp_available === false
      ? "Timestamp not provided"
      : (source.start_timestamp || "00:00") + " – " + (source.end_timestamp || "00:00");
    meta.appendChild(createElement("span", "", timingLabel));
    meta.appendChild(createElement("span", "source-arrow", "↗"));
    content.append(meta, createElement("span", "source-text", source.text || "Open the video to verify this passage."));
    link.append(tag, content);
    return link;
  }

  function scrollToBottom() {
    window.requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  function clearConversation() {
    chatMessages.querySelectorAll(".message").forEach((message) => message.remove());
    if (welcomeState) welcomeState.hidden = true;
  }

  function updateVideoPreview(data) {
    const videoId = data.video_id;
    videoPreview.hidden = false;
    videoThumb.src = "https://img.youtube.com/vi/" + encodeURIComponent(videoId) + "/hqdefault.jpg";
    videoThumb.alt = "Thumbnail for YouTube video " + videoId;
    videoThumb.onerror = function () {
      this.removeAttribute("src");
      this.alt = "YouTube thumbnail is unavailable";
    };
    videoIdLabel.textContent = "YouTube video · " + videoId;
    const words = Number(data.word_count || 0).toLocaleString();
    const chunks = Number(data.chunk_count || 0).toLocaleString();
    const timeInfo = data.timestamp_available === false ? "timestamps not provided" : "timestamps preserved";
    const inputInfo = data.input_mode === "pasted" ? "pasted transcript" : (data.language || "captions");
    videoMeta.textContent = words + " words indexed · " + chunks + " chunks · " + timeInfo;
    videoMeta.title = inputInfo;
    openVideoLink.href = "https://www.youtube.com/watch?v=" + encodeURIComponent(videoId);
  }

  function onIndexSuccess(data) {
    activeVideoId = data.video_id;
    updateVideoPreview(data);
    const chunks = Number(data.chunk_count || 0).toLocaleString();
    const manual = data.input_mode === "pasted";
    setStatus(
      (manual ? "Pasted transcript indexed successfully. " : "Transcript indexed successfully. ")
        + chunks + " searchable chunks are ready.",
      "success"
    );
    manualPanel.hidden = true;
    toggleManual.setAttribute("aria-expanded", "false");
    questionInput.disabled = false;
    questionInput.placeholder = "Ask about this video… (Enter to send, Shift + Enter for a new line)";
    sendButton.disabled = false;
    suggestionButtons.forEach((button) => { button.disabled = false; });
    chatStatus.textContent = "Index ready";
    chatStatus.parentElement.classList.add("is-ready");
    chatSubtitle.textContent = manual
      ? (data.timestamp_available
          ? "Evidence from pasted transcript · timestamps preserved"
          : "Evidence from pasted transcript · no timestamps supplied")
      : "Answers grounded in " + (data.language || "video") + " captions";

    clearConversation();
    const intro = manual
      ? (data.timestamp_available
          ? "Your pasted transcript is indexed. Ask a question and I’ll retrieve relevant passages with their copied timestamps, then attach source links for verification."
          : "Your pasted transcript is indexed. Ask a question and I’ll retrieve relevant passages. The source cards link to the video, but exact timestamps were not supplied in the pasted text.")
      : "Your video is indexed and ready. Ask a question and I’ll search the transcript, answer from the retrieved passages, and attach timestamp links so you can verify the evidence.";
    appendMessage("assistant", intro, []);
    questionInput.focus();
  }

  function revealManualFallback() {
    manualPanel.hidden = false;
    toggleManual.setAttribute("aria-expanded", "true");
  }

  toggleManual.addEventListener("click", () => {
    const shouldOpen = manualPanel.hidden;
    manualPanel.hidden = !shouldOpen;
    toggleManual.setAttribute("aria-expanded", String(shouldOpen));
    if (shouldOpen) transcriptInput.focus();
  });

  loadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (loadingVideo) return;

    const videoUrl = videoInput.value.trim();
    if (!videoUrl) {
      setStatus("Paste a YouTube URL or video ID first.", "error");
      return;
    }

    setLoadingVideo(true);
    try {
      const data = await requestJson("/api/load-video", { video_url: videoUrl });
      onIndexSuccess(data);
    } catch (error) {
      setStatus(error.message || "Could not index this video.", "error");
      const message = String(error.message || "").toLowerCase();
      if (message.includes("transcript") || message.includes("caption") || message.includes("blocked")) {
        revealManualFallback();
      }
    } finally {
      setLoadingVideo(false);
    }
  });

  manualForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (loadingVideo) return;

    const videoUrl = videoInput.value.trim();
    const transcriptText = transcriptInput.value.trim();
    if (!videoUrl) {
      setStatus("Enter the YouTube URL first so source cards link to the right video.", "error");
      videoInput.focus();
      return;
    }
    if (!transcriptText) {
      setStatus("Paste the video's transcript first.", "error");
      transcriptInput.focus();
      return;
    }

    loadingVideo = true;
    loadButton.disabled = true;
    manualSubmit.disabled = true;
    manualSubmit.classList.add("is-loading");
    manualSubmitLabel.textContent = "Indexing pasted transcript…";
    videoStatus.textContent = "Preparing timestamp-aware chunks and building the semantic search index.";
    videoStatus.className = "inline-status";
    videoStatus.hidden = false;

    try {
      const data = await requestJson("/api/load-transcript", {
        video_url: videoUrl,
        transcript_text: transcriptText
      });
      onIndexSuccess(data);
      transcriptInput.value = "";
    } catch (error) {
      setStatus(error.message || "Could not index the pasted transcript.", "error");
    } finally {
      loadingVideo = false;
      loadButton.disabled = false;
      manualSubmit.disabled = false;
      manualSubmit.classList.remove("is-loading");
      manualSubmitLabel.textContent = "Index pasted transcript →";
    }
  });

  async function askQuestion(question) {
    if (!activeVideoId || sendingQuestion) return;
    const cleanedQuestion = String(question || "").trim();
    if (!cleanedQuestion) return;

    appendMessage("user", cleanedQuestion, []);
    questionInput.value = "";
    resizeQuestionInput();
    setSendingQuestion(true);

    try {
      const data = await requestJson("/api/chat", {
        video_id: activeVideoId,
        question: cleanedQuestion
      });
      appendMessage("assistant", data.answer || "No answer was returned.", data.sources || [], false, data.citation_audit);
    } catch (error) {
      appendMessage("assistant", error.message || "Something went wrong. Please try again.", [], true);
    } finally {
      setSendingQuestion(false);
      if (!questionInput.disabled) questionInput.focus();
    }
  }

  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion(questionInput.value);
  });

  questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (chatForm.requestSubmit) chatForm.requestSubmit();
      else askQuestion(questionInput.value);
    }
  });

  questionInput.addEventListener("input", resizeQuestionInput);

  function resizeQuestionInput() {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 110) + "px";
  }

  suggestionButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const question = button.getAttribute("data-question") || "";
      if (!activeVideoId || sendingQuestion) return;
      askQuestion(question);
    });
  });

  // Light startup check makes a misconfigured deployment easier to diagnose.
  requestJson("/api/health").catch(() => {
    setStatus("The API is not reachable. Check the deployment or start the Flask server.", "error");
  });
})();
