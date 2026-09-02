document.addEventListener('DOMContentLoaded', () => {
  const loadForm = document.getElementById('load-form');
  const videoInput = document.getElementById('video-input');
  const loadBtn = document.getElementById('load-btn');
  const videoStatus = document.getElementById('video-status');
  const chatBadge = document.getElementById('chat-badge');
  
  const chatForm = document.getElementById('chat-form');
  const questionInput = document.getElementById('question-input');
  const sendBtn = document.getElementById('send-btn');
  const chatMessages = document.getElementById('chat-messages');

  let currentVideoId = null;

  // Function to show status messages
  function showStatus(message, type) {
    videoStatus.className = `status ${type} show`;
    
    if (type === 'loading') {
      videoStatus.innerHTML = `<div class="spinner"></div> ${message}`;
    } else {
      videoStatus.textContent = message;
    }
    
    // Auto-hide success/error messages after a while if desired
    if (type !== 'loading') {
      setTimeout(() => {
        if (videoStatus.className.includes(type)) {
          // Keep it visible but you could fade it out
        }
      }, 5000);
    }
  }

  // Load video event listener
  loadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = videoInput.value.trim();
    if (!url) return;

    // Clear previous chat history when a new video is loaded
    chatMessages.innerHTML = `
      <div class="message bot">
        <div class="bubble">
          Load a YouTube video above, then ask anything about its content.<br><br>
          Try: <em>"Can you summarize the video?"</em> or <em>"What topics are discussed?"</em>
        </div>
      </div>
    `;

    // UI state: Loading
    loadBtn.disabled = true;
    videoInput.disabled = true;
    showStatus('Fetching transcript and generating embeddings...', 'loading');
    
    try {
      const response = await fetch('/api/load-video', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ video_url: url })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        currentVideoId = data.video_id;
        showStatus(data.message, 'success');
        
        // Enable chat interface
        chatBadge.textContent = 'Ready to Chat';
        chatBadge.className = 'badge badge-ready';
        questionInput.disabled = false;
        sendBtn.disabled = false;
        questionInput.focus();
      } else {
        showStatus(data.error || 'Failed to load video.', 'error');
        currentVideoId = null;
        chatBadge.textContent = 'No video loaded';
        chatBadge.className = 'badge badge-idle';
        questionInput.disabled = true;
        sendBtn.disabled = true;
      }
    } catch (error) {
      showStatus('Network error occurred while loading video.', 'error');
    } finally {
      loadBtn.disabled = false;
      videoInput.disabled = false;
    }
  });

  // Helper to add a message bubble
  function addMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'bubble';
    
    // Simple markdown formatting (bold, newlines)
    let formattedText = text.replace(/\n/g, '<br>');
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    bubbleDiv.innerHTML = formattedText;
    msgDiv.appendChild(bubbleDiv);
    
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Helper to add typing indicator
  function addTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot typing';
    msgDiv.id = 'typing-indicator';
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'bubble';
    
    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'typing-indicator';
    indicatorDiv.innerHTML = '<span></span><span></span><span></span>';
    
    bubbleDiv.appendChild(indicatorDiv);
    msgDiv.appendChild(bubbleDiv);
    
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
      indicator.remove();
    }
  }

  // Chat event listener
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!currentVideoId) {
      alert('Please load a video first!');
      return;
    }
    
    const question = questionInput.value.trim();
    if (!question) return;
    
    // Add user message
    addMessage(question, 'user');
    questionInput.value = '';
    
    // Disable inputs while waiting
    questionInput.disabled = true;
    sendBtn.disabled = true;
    
    // Show typing indicator
    addTypingIndicator();
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_id: currentVideoId,
          question: question
        })
      });
      
      const data = await response.json();
      removeTypingIndicator();
      
      if (response.ok) {
        addMessage(data.answer, 'bot');
      } else {
        addMessage(`**Error:** ${data.error || 'Failed to get answer.'}`, 'bot');
      }
    } catch (error) {
      removeTypingIndicator();
      addMessage('**Error:** Network error occurred.', 'bot');
    } finally {
      questionInput.disabled = false;
      sendBtn.disabled = false;
      questionInput.focus();
    }
  });
});
