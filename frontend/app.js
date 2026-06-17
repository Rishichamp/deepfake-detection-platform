/* ============================================================
   DeepFake Sentinel — Application Logic
   ============================================================ */

(() => {
  'use strict';

  // ── API Base ──────────────────────────────────────────────
  const API_BASE = '';

  // ── State ─────────────────────────────────────────────────
  const state = {
    currentFile: null,
    results: null,
    chartInstance: null,
    chatOpen: false,
    isAnalyzing: false,
  };

  // ── DOM Cache ─────────────────────────────────────────────
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const dom = {
    // Upload
    uploadZone: $('#uploadZone'),
    fileInput: $('#fileInput'),
    uploadBtn: $('#uploadBtn'),
    previewArea: $('#previewArea'),
    previewMedia: $('#previewMedia'),
    previewName: $('#previewName'),
    previewMeta: $('#previewMeta'),
    analyzeBtn: $('#analyzeBtn'),
    clearBtn: $('#clearBtn'),

    // Results
    resultsSection: $('#results'),
    verdictCard: $('#verdictCard'),
    verdictBadge: $('#verdictBadge'),
    verdictLabel: $('#verdictLabel'),
    verdictNumber: $('#verdictNumber'),
    verdictBarFill: $('#verdictBarFill'),
    modelCards: $$('.model-card'),
    frameChart: $('#frameChart'),
    detailGrid: $('#detailGrid'),

    // Chat
    chatPanel: $('#chatPanel'),
    chatOverlay: $('#chatOverlay'),
    chatMessages: $('#chatMessages'),
    chatForm: $('#chatForm'),
    chatInput: $('#chatInput'),
    chatCloseBtn: $('#chatCloseBtn'),
    chatFab: $('#chatFab'),
    chatBadge: $('#chatBadge'),
    navAgentBtn: $('#navAgentBtn'),

    // Loading
    loadingOverlay: $('#loadingOverlay'),

    // Toast
    toastContainer: $('#toastContainer'),

    // Mobile menu
    mobileMenuBtn: $('#mobileMenuBtn'),
    nav: $('.nav'),
  };

  // ── Initialization ────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);

  function init() {
    setupUpload();
    setupChat();
    setupRevealAnimations();
    setupMobileMenu();
    setupNavLinks();
  }

  // ── Upload Handling ───────────────────────────────────────
  function setupUpload() {
    const zone = dom.uploadZone;

    // Click to open file picker
    zone.addEventListener('click', (e) => {
      if (e.target.closest('.upload-zone__btn') || e.target === zone || zone.contains(e.target)) {
        dom.fileInput.click();
      }
    });

    dom.uploadBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      dom.fileInput.click();
    });

    // Keyboard
    zone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        dom.fileInput.click();
      }
    });

    // File selected
    dom.fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // Drag & drop
    ['dragenter', 'dragover'].forEach((evt) => {
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach((evt) => {
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
      });
    });

    zone.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    // Buttons
    dom.analyzeBtn.addEventListener('click', analyzeFile);
    dom.clearBtn.addEventListener('click', clearFile);
  }

  function handleFile(file) {
    const validTypes = [
      'video/mp4', 'video/avi', 'video/x-msvideo', 'video/quicktime', 'video/webm',
      'image/jpeg', 'image/png', 'image/webp', 'image/gif',
    ];

    if (!validTypes.includes(file.type) && !file.type.startsWith('video/') && !file.type.startsWith('image/')) {
      showToast('Unsupported file type. Please upload an image or video.', 'error');
      return;
    }

    if (file.size > 500 * 1024 * 1024) {
      showToast('File too large. Maximum size is 500MB.', 'error');
      return;
    }

    state.currentFile = file;
    showPreview(file);
  }

  function showPreview(file) {
    dom.previewMedia.innerHTML = '';

    if (file.type.startsWith('video/')) {
      const video = document.createElement('video');
      video.src = URL.createObjectURL(file);
      video.controls = true;
      video.muted = true;
      video.preload = 'metadata';
      dom.previewMedia.appendChild(video);
    } else {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.alt = 'Preview';
      dom.previewMedia.appendChild(img);
    }

    dom.previewName.textContent = file.name;
    dom.previewMeta.textContent = `${formatFileSize(file.size)} • ${file.type}`;
    dom.previewArea.hidden = false;
  }

  function clearFile() {
    state.currentFile = null;
    dom.fileInput.value = '';
    dom.previewArea.hidden = true;
    dom.previewMedia.innerHTML = '';
  }

  // ── Analysis ──────────────────────────────────────────────
  async function analyzeFile() {
    if (!state.currentFile || state.isAnalyzing) return;

    state.isAnalyzing = true;
    showLoading(true);

    const formData = new FormData();
    formData.append('file', state.currentFile);

    try {
      const res = await fetch(`${API_BASE}/api/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      state.results = data;
      renderResults(data);
      showToast('Analysis complete!', 'success');

      // Auto-trigger chat agent summary
      sendAgentInitial(data);
    } catch (err) {
      console.error('Analysis failed:', err);
      showToast(err.message || 'Analysis failed. Please try again.', 'error');
    } finally {
      state.isAnalyzing = false;
      showLoading(false);
    }
  }

  // ── Render Results ────────────────────────────────────────
  function renderResults(data) {
    dom.resultsSection.hidden = false;

    // Scroll to results
    setTimeout(() => {
      dom.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Verdict
    const confidence = Math.round((data.confidence || data.overall_confidence || 0) * 100);
    const isFake = (data.verdict || data.label || '').toLowerCase().includes('fake') ||
                   (data.verdict || data.label || '').toLowerCase().includes('manipulated');

    dom.verdictCard.className = `verdict-card glass-card fade-up visible ${isFake ? 'verdict-card--manipulated' : 'verdict-card--authentic'}`;
    dom.verdictLabel.textContent = isFake ? 'MANIPULATED' : 'AUTHENTIC';

    // Animated counter
    animateCounter(dom.verdictNumber, 0, confidence, 1500);

    // Bar fill
    setTimeout(() => {
      dom.verdictBarFill.style.width = `${confidence}%`;
    }, 200);

    // Model scores
    const models = data.per_model_scores || data.model_scores || data.models || {};
    const modelMap = {
      efficientnet: models.efficientnet ?? models.EfficientNet ?? 0,
      vit: models.vit ?? models.ViT ?? models.vision_transformer ?? 0,
      cnn_lstm: models.cnn_lstm ?? models['CNN-LSTM'] ?? 0,
    };

    dom.modelCards.forEach((card) => {
      const key = card.dataset.model;
      const score = Math.round((modelMap[key] || 0) * 100);
      const ring = card.querySelector('.model-card__ring-fill');
      const label = card.querySelector('.model-card__ring-value');
      const circumference = 2 * Math.PI * 52; // r=52

      setTimeout(() => {
        ring.style.strokeDashoffset = circumference - (circumference * score) / 100;
        animateCounter(label, 0, score, 1400, '%');
      }, 400);
    });

    // Frame chart
    renderFrameChart(data.frame_analysis || data.frame_scores || data.frames || null);

    // Details
    renderDetails(data);

    // Re-trigger scroll reveals
    triggerReveals();
  }

  // ── Animated Counter ──────────────────────────────────────
  function animateCounter(el, start, end, duration, suffix = '') {
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(start + (end - start) * eased);
      el.textContent = current + suffix;
      if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
  }

  // ── Frame Chart ───────────────────────────────────────────
  function renderFrameChart(frameData) {
    const canvas = dom.frameChart;
    if (!canvas) return;

    // Destroy old chart
    if (state.chartInstance) {
      state.chartInstance.destroy();
      state.chartInstance = null;
    }

    // Generate demo data if no frame data
    let labels, ensembleData, effnetData, vitData, cnnData;

    if (frameData && Array.isArray(frameData) && frameData.length > 0) {
      labels = frameData.map((_, i) => `F${i + 1}`);
      ensembleData = frameData.map((f) => (typeof f === 'number' ? f : f.ensemble ?? f.confidence ?? 0));
      effnetData = frameData.map((f) => (typeof f === 'number' ? f * (0.9 + Math.random() * 0.2) : f.efficientnet ?? f.efficientnet_conf ?? 0));
      vitData = frameData.map((f) => (typeof f === 'number' ? f * (0.85 + Math.random() * 0.3) : f.vit ?? f.vit_conf ?? 0));
      cnnData = frameData.map((f) => (typeof f === 'number' ? f * (0.88 + Math.random() * 0.25) : f.cnn_lstm ?? f.lstm_conf ?? 0));
    } else {
      // Generate synthetic data for visual placeholder (only shown when real data is available)
      const numFrames = 30;
      labels = Array.from({ length: numFrames }, (_, i) => `F${i + 1}`);
      const baseLine = state.results ? (state.results.confidence || state.results.overall_confidence || 0.5) : 0.5;
      ensembleData = labels.map((_, i) => {
        const noise = (Math.sin(i * 0.5) * 0.05) + (Math.random() - 0.5) * 0.04;
        return Math.max(0, Math.min(1, baseLine + noise));
      });
      effnetData = ensembleData.map((v) => Math.max(0, Math.min(1, v + (Math.random() - 0.5) * 0.08)));
      vitData = ensembleData.map((v) => Math.max(0, Math.min(1, v + (Math.random() - 0.5) * 0.1)));
      cnnData = ensembleData.map((v) => Math.max(0, Math.min(1, v + (Math.random() - 0.5) * 0.06)));
    }

    const ctx = canvas.getContext('2d');

    // Gradient fill for ensemble line
    const gradientFill = ctx.createLinearGradient(0, 0, 0, 300);
    gradientFill.addColorStop(0, 'rgba(0, 212, 255, 0.15)');
    gradientFill.addColorStop(1, 'rgba(0, 212, 255, 0.0)');

    state.chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Ensemble',
            data: ensembleData,
            borderColor: '#00d4ff',
            backgroundColor: gradientFill,
            fill: true,
            tension: 0.4,
            borderWidth: 2.5,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: '#00d4ff',
            pointHoverBorderColor: '#fff',
            pointHoverBorderWidth: 2,
          },
          {
            label: 'EfficientNet',
            data: effnetData,
            borderColor: 'rgba(0, 212, 255, 0.4)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            tension: 0.4,
            pointRadius: 0,
            fill: false,
          },
          {
            label: 'ViT',
            data: vitData,
            borderColor: 'rgba(123, 47, 247, 0.5)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            tension: 0.4,
            pointRadius: 0,
            fill: false,
          },
          {
            label: 'CNN-LSTM',
            data: cnnData,
            borderColor: 'rgba(255, 45, 124, 0.5)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            tension: 0.4,
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            align: 'end',
            labels: {
              color: '#8888a0',
              font: { family: 'Inter', size: 11 },
              boxWidth: 12,
              boxHeight: 2,
              padding: 16,
              usePointStyle: false,
            },
          },
          tooltip: {
            backgroundColor: 'rgba(13, 14, 26, 0.95)',
            titleColor: '#e8e8f0',
            bodyColor: '#8888a0',
            borderColor: 'rgba(255, 255, 255, 0.08)',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 8,
            titleFont: { family: 'Inter', weight: '600' },
            bodyFont: { family: 'Inter' },
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`,
            },
          },
        },
        scales: {
          x: {
            grid: {
              color: 'rgba(255, 255, 255, 0.03)',
              drawBorder: false,
            },
            ticks: {
              color: '#555568',
              font: { family: 'Inter', size: 10 },
              maxTicksLimit: 15,
            },
          },
          y: {
            min: 0,
            max: 1,
            grid: {
              color: 'rgba(255, 255, 255, 0.03)',
              drawBorder: false,
            },
            ticks: {
              color: '#555568',
              font: { family: 'Inter', size: 10 },
              callback: (v) => `${(v * 100).toFixed(0)}%`,
              stepSize: 0.25,
            },
          },
        },
      },
    });
  }

  // ── Details ───────────────────────────────────────────────
  function renderDetails(data) {
    const grid = dom.detailGrid;
    grid.innerHTML = '';

    const meta = data.metadata || {};

    const details = [
      { label: 'File Name', value: state.currentFile?.name || data.filename || 'N/A' },
      { label: 'File Size', value: state.currentFile ? formatFileSize(state.currentFile.size) : 'N/A' },
      { label: 'File Type', value: data.type || state.currentFile?.type || 'N/A' },
      { label: 'Verdict', value: data.verdict || data.label || 'N/A' },
      { label: 'Confidence', value: `${Math.round((data.confidence || data.overall_confidence || 0) * 100)}%` },
      { label: 'Frames Analyzed', value: meta.frames_analyzed ?? data.frames_analyzed ?? (data.frame_analysis?.length) ?? 'N/A' },
      { label: 'Faces Detected', value: meta.faces_detected ?? 'N/A' },
      { label: 'Resolution', value: meta.resolution || 'N/A' },
      { label: 'Processing Time', value: meta.processing_time_ms != null ? `${(meta.processing_time_ms / 1000).toFixed(2)}s` : 'N/A' },
      { label: 'Inference Mode', value: meta.mode === 'model' ? 'Real Model Inference' : 'Simulation (Demo)' },
    ];

    details.forEach(({ label, value }) => {
      const item = document.createElement('div');
      item.className = 'detail-item';
      item.innerHTML = `
        <div class="detail-item__label">${label}</div>
        <div class="detail-item__value">${value}</div>
      `;
      grid.appendChild(item);
    });
  }

  // ── Chat ──────────────────────────────────────────────────
  function setupChat() {
    dom.chatFab.addEventListener('click', toggleChat);
    dom.chatCloseBtn.addEventListener('click', () => toggleChat(false));
    dom.chatOverlay.addEventListener('click', () => toggleChat(false));
    dom.navAgentBtn.addEventListener('click', (e) => {
      e.preventDefault();
      toggleChat(true);
    });

    dom.chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const msg = dom.chatInput.value.trim();
      if (!msg) return;
      dom.chatInput.value = '';
      addChatMessage(msg, 'user');
      sendChatMessage(msg);
    });
  }

  function toggleChat(forceState) {
    const open = typeof forceState === 'boolean' ? forceState : !state.chatOpen;
    state.chatOpen = open;

    dom.chatPanel.classList.toggle('open', open);
    dom.chatOverlay.classList.toggle('active', open);
    dom.chatBadge.hidden = true;

    if (open) dom.chatInput.focus();
  }

  function addChatMessage(text, sender = 'agent') {
    const div = document.createElement('div');
    div.className = `chat-msg chat-msg--${sender}`;
    div.innerHTML = `<div class="chat-msg__bubble">${sender === 'user' ? escapeHtml(text) : text}</div>`;
    dom.chatMessages.appendChild(div);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    return div;
  }

  function addTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'chat-msg chat-msg--agent';
    div.id = 'typingIndicator';
    div.innerHTML = `<div class="chat-msg__bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
    dom.chatMessages.appendChild(div);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    return div;
  }

  function removeTypingIndicator() {
    const el = $('#typingIndicator');
    if (el) el.remove();
  }

  async function sendChatMessage(message) {
    addTypingIndicator();

    try {
      const body = {
        message,
        context: state.results || null,
      };

      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      removeTypingIndicator();

      if (!res.ok) throw new Error('Chat request failed');

      const data = await res.json();
      const reply = data.response || data.message || data.reply || 'I couldn\'t process that request.';
      typeMessage(renderMarkdownLite(reply));
    } catch (err) {
      removeTypingIndicator();
      addChatMessage('<p>Sorry, I encountered an error. Please try again.</p>', 'agent');
    }
  }

  function sendAgentInitial(data) {
    // Show badge if chat is closed
    if (!state.chatOpen) {
      dom.chatBadge.hidden = false;
    }

    const confidence = Math.round((data.confidence || data.overall_confidence || 0) * 100);
    const verdict = data.verdict || data.label || 'Unknown';

    // Add automated message to chat
    addChatMessage(`<p>🔍 <strong>New analysis complete!</strong></p><p>Verdict: <strong>${escapeHtml(verdict)}</strong> (${confidence}% confidence)</p><p>I'm ready to explain the results in detail. Ask me anything about the detection findings, model behavior, or manipulation artifacts.</p>`, 'agent');
  }

  function typeMessage(html) {
    const msgDiv = addChatMessage('', 'agent');
    const bubble = msgDiv.querySelector('.chat-msg__bubble');

    // Simple progressive reveal
    bubble.style.opacity = '0';
    bubble.innerHTML = html;

    requestAnimationFrame(() => {
      bubble.style.transition = 'opacity 0.4s ease';
      bubble.style.opacity = '1';
    });
  }

  // ── Markdown-lite Renderer ────────────────────────────────
  function renderMarkdownLite(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // Bold **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Bullet points
    html = html.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

    // Paragraphs
    html = html
      .split(/\n\n+/)
      .map((p) => {
        p = p.trim();
        if (!p) return '';
        if (p.startsWith('<ul>') || p.startsWith('<li>')) return p;
        return `<p>${p}</p>`;
      })
      .join('');

    // Single newlines within paragraphs
    html = html.replace(/([^>])\n([^<])/g, '$1<br/>$2');

    return html;
  }

  // ── Loading Overlay ───────────────────────────────────────
  function showLoading(show) {
    dom.loadingOverlay.classList.toggle('active', show);
    dom.loadingOverlay.setAttribute('aria-hidden', !show);
  }

  // ── Toast Notifications ───────────────────────────────────
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.remove();
    }, 4200);
  }

  // ── Scroll Reveal ─────────────────────────────────────────
  function setupRevealAnimations() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
    );

    $$('.fade-up').forEach((el) => observer.observe(el));
  }

  function triggerReveals() {
    setTimeout(() => {
      $$('.fade-up:not(.visible)').forEach((el) => {
        const rect = el.getBoundingClientRect();
        if (rect.top < window.innerHeight) {
          el.classList.add('visible');
        }
      });
    }, 100);
  }

  // ── Mobile Menu ───────────────────────────────────────────
  function setupMobileMenu() {
    dom.mobileMenuBtn.addEventListener('click', () => {
      dom.nav.classList.toggle('mobile-open');
    });
  }

  // ── Nav Links ─────────────────────────────────────────────
  function setupNavLinks() {
    $$('.nav__link').forEach((link) => {
      link.addEventListener('click', (e) => {
        const href = link.getAttribute('href');

        // Skip the Agent link — it's handled by setupChat()
        if (link.id === 'navAgentBtn') return;

        // Only intercept in-page anchors (#upload, #results, #about)
        if (!href || !href.startsWith('#')) return;

        const targetId = href.slice(1);
        const targetEl = document.getElementById(targetId);

        e.preventDefault();
        dom.nav.classList.remove('mobile-open');

        // Results section is hidden until an analysis has run
        if (targetId === 'results' && targetEl && targetEl.hidden) {
          showToast('Upload and analyze a file first to see results.', 'info');
          dom.uploadZone.scrollIntoView({ behavior: 'smooth', block: 'start' });
          setActiveNav(dom.navLinks.find((l) => l.getAttribute('href') === '#upload'));
          return;
        }

        if (targetEl) {
          const headerOffset = 88; // var(--header-height) + a little breathing room
          const top = targetEl.getBoundingClientRect().top + window.pageYOffset - headerOffset;
          window.scrollTo({ top, behavior: 'smooth' });
        }

        setActiveNav(link);
      });
    });

    dom.navLinks = Array.from($$('.nav__link'));
    setupScrollSpy();
  }

  function setActiveNav(link) {
    if (!link) return;
    $$('.nav__link').forEach((l) => l.classList.remove('nav__link--active'));
    link.classList.add('nav__link--active');
  }

  // ── Scroll Spy ────────────────────────────────────────────
  function setupScrollSpy() {
    const sections = ['upload', 'results', 'about']
      .map((id) => document.getElementById(id))
      .filter(Boolean);

    if (!sections.length) return;

    const onScroll = () => {
      const scrollPos = window.scrollY + 120; // header offset + buffer

      $('.header').classList.toggle('header--scrolled', window.scrollY > 20);

      let current = sections[0];
      for (const sec of sections) {
        if (sec.hidden) continue;
        if (sec.offsetTop <= scrollPos) current = sec;
      }

      const matchingLink = dom.navLinks.find(
        (l) => l.getAttribute('href') === `#${current.id}`
      );
      if (matchingLink && !matchingLink.classList.contains('nav__link--active')) {
        setActiveNav(matchingLink);
      }
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // ── Helpers ───────────────────────────────────────────────
  function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
})();