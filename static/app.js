let ws;
let autoChatActive = false;
let streamingMessages = {}; // model_id -> element
let currentSessionId = null;

// Markdown setup
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
});

let reconnectTimer = null;

function connect() {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        console.log('Connected to chatroom');
    };

    ws.onclose = () => {
        console.log('Disconnected, reconnecting in 2s...');
        reconnectTimer = setTimeout(connect, 2000);
    };

    ws.onerror = () => {
        try { ws.close(); } catch(e) {}
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
}

// Reconnect when mobile browser comes back from background
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && (!ws || ws.readyState > 1)) {
        connect();
    }
});

function handleMessage(data) {
    switch (data.type) {
        case 'init':
            populateModelSelect(data.available_models);
            loadActiveModels(data.active_models);
            document.getElementById('user-profile').value = data.user_profile || '';
            loadHistory(data.history);
            if (data.auto_chat_active) {
                autoChatActive = true;
                updateAutoChatButton();
            }
            // Sessions
            if (data.sessions) {
                currentSessionId = data.current_session_id;
                renderSessions(data.sessions, data.current_session_id);
            }
            // Model memory
            renderModelMemory();
            break;

        case 'available_models':
            populateModelSelect(data.models);
            break;

        case 'message':
            const msgColor = data.role === 'user' ? '#00b4d8' : (data.color || activeModels[data.role]?.color || '#888');
            addMessage(data.role === 'user' ? 'User' : data.name, data.content, msgColor);
            break;

        case 'thinking':
            showThinking(data.model_id, data.name, data.color);
            break;

        case 'start':
            removeThinking(data.model_id);
            startStreaming(data.model_id, data.name, data.color);
            break;

        case 'token':
            appendToken(data.model_id, data.token);
            break;

        case 'end':
            endStreaming(data.model_id);
            break;

        case 'error':
            addSystemMessage(data.message);
            break;

        case 'system':
            addSystemMessage(data.message);
            break;

        case 'model_added':
            if (data.config) {
                activeModels[data.model_id] = data.config;
            }
            renderActiveModels();
            renderModelMemory();
            break;

        case 'model_removed':
            delete activeModels[data.model_id];
            renderActiveModels();
            renderModelMemory();
            break;

        case 'model_updated':
            if (data.config) {
                activeModels[data.model_id] = data.config;
            }
            break;

        case 'auto_chat_status':
            autoChatActive = data.active;
            updateAutoChatButton();
            break;

        case 'killed':
            autoChatActive = false;
            updateAutoChatButton();
            for (const mid of Object.keys(streamingMessages)) {
                endStreaming(mid);
            }
            for (const mid of Object.keys(thinkingElements)) {
                removeThinking(mid);
            }
            addSystemMessage(data.message);
            break;

        case 'history_cleared':
            document.getElementById('messages').innerHTML = '';
            addSystemMessage('History cleared.');
            break;

        // Session management
        case 'sessions_list':
            currentSessionId = data.current_session_id;
            renderSessions(data.sessions, data.current_session_id);
            break;

        case 'session_loaded':
            currentSessionId = data.session_id;
            document.getElementById('messages').innerHTML = '';
            if (data.history && data.history.length) {
                loadHistory(data.history);
            }
            break;
    }
}

// ═══ Sessions ═══

function renderSessions(sessions, currentId) {
    const container = document.getElementById('sessions-list');
    container.innerHTML = '';

    if (!sessions || sessions.length === 0) {
        container.innerHTML = '<div class="sessions-empty">No sessions yet</div>';
        return;
    }

    for (const s of sessions) {
        const item = document.createElement('div');
        item.className = 'session-item' + (s.id === currentId ? ' active' : '');
        item.onclick = (e) => {
            if (e.target.classList.contains('session-delete') || e.target.closest('.session-delete')) return;
            if (e.target.hasAttribute('contenteditable')) return;
            switchSession(s.id);
        };

        // Format date
        let dateStr = '';
        if (s.updated_at) {
            const d = new Date(s.updated_at);
            const now = new Date();
            const diffMs = now - d;
            const diffDays = Math.floor(diffMs / 86400000);
            if (diffDays === 0) {
                dateStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            } else if (diffDays === 1) {
                dateStr = 'Yesterday';
            } else if (diffDays < 7) {
                dateStr = d.toLocaleDateString([], { weekday: 'short' });
            } else {
                dateStr = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
            }
        }

        item.innerHTML = `
            <div class="session-info">
                <div class="session-title" title="${escapeHtml(s.title)}">${escapeHtml(s.title)}</div>
                <div class="session-meta">
                    <span>${dateStr}</span>
                    ${s.message_count ? `<span class="session-msg-count">${s.message_count} msgs</span>` : ''}
                </div>
            </div>
            <button class="session-delete" onclick="event.stopPropagation();deleteSession('${s.id}')" title="Delete">&times;</button>
        `;

        // Double-click title to rename
        const titleEl = item.querySelector('.session-title');
        titleEl.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            titleEl.contentEditable = 'true';
            titleEl.focus();
            // Select all text
            const range = document.createRange();
            range.selectNodeContents(titleEl);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
        });

        titleEl.addEventListener('blur', () => {
            titleEl.contentEditable = 'false';
            const newTitle = titleEl.textContent.trim();
            if (newTitle && newTitle !== s.title) {
                renameSession(s.id, newTitle);
            }
        });

        titleEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                titleEl.blur();
            }
            if (e.key === 'Escape') {
                titleEl.textContent = s.title;
                titleEl.blur();
            }
        });

        container.appendChild(item);
    }
}

function createSession() {
    safeSend({ action: 'create_session' });
}

function switchSession(sessionId) {
    if (sessionId === currentSessionId) return;
    safeSend({ action: 'switch_session', session_id: sessionId });
}

function deleteSession(sessionId) {
    if (!confirm('Delete this session?')) return;
    safeSend({ action: 'delete_session', session_id: sessionId });
}

function renameSession(sessionId, newTitle) {
    safeSend({ action: 'rename_session', session_id: sessionId, title: newTitle });
}

// ═══ Settings Modal ═══

function openSettings() {
    const overlay = document.getElementById('settings-overlay-modal');
    overlay.classList.add('open');
    // Refresh model memory when opening
    renderModelMemory();
    // Init slider fills in settings
    const addModelSection = document.getElementById('add-model-section');
    if (addModelSection) initAllSliderFills(addModelSection);
}

function closeSettings() {
    const overlay = document.getElementById('settings-overlay-modal');
    overlay.classList.remove('open');
}

// ═══ Model Memory ═══

function renderModelMemory() {
    const container = document.getElementById('model-memory-list');
    if (!container) return;
    container.innerHTML = '';

    if (Object.keys(activeModels).length === 0) {
        container.innerHTML = '<div class="sessions-empty">No active models</div>';
        return;
    }

    for (const [mid, cfg] of Object.entries(activeModels)) {
        const item = document.createElement('div');
        item.className = 'memory-item';
        const memory = cfg.memory || '';
        item.innerHTML = `
            <div class="memory-item-header">
                <span class="dot" style="background:${cfg.color}"></span>
                <span class="name">${escapeHtml(cfg.display_name)}</span>
            </div>
            <textarea placeholder="Things this model should always remember..." rows="3">${escapeHtml(memory)}</textarea>
            <button class="btn-sm" onclick="saveModelMemory('${mid}', this)">Save</button>
        `;
        container.appendChild(item);
    }
}

function saveModelMemory(modelId, btn) {
    const item = btn.closest('.memory-item');
    const textarea = item.querySelector('textarea');
    const memory = textarea.value;
    safeSend({ action: 'update_model_memory', model_id: modelId, memory: memory });
    // Visual feedback
    const orig = btn.textContent;
    btn.textContent = 'Saved!';
    btn.style.color = 'var(--success)';
    setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 1500);
}

// ═══ Model Select ═══

function populateModelSelect(models) {
    const sel = document.getElementById('model-select');
    sel.innerHTML = '<option value="">Select a model...</option>';
    for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        const sizeMB = (m.size / 1e9).toFixed(1);
        opt.textContent = `${m.name} (${sizeMB}GB)`;
        sel.appendChild(opt);
    }
}

function refreshModels() {
    const host = document.getElementById('ollama-host').value.trim() || 'http://localhost:11434';
    safeSend({ action: 'refresh_models', host: host });
}

// ═══ Slider Fill ═══

function updateSliderFill(input) {
    const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${pct}%, var(--border) ${pct}%, var(--border) 100%)`;
}

function initAllSliderFills(container) {
    container.querySelectorAll('input[type="range"]').forEach(updateSliderFill);
}

// ═══ Active Models ═══

let activeModels = {};

function loadActiveModels(models) {
    activeModels = models || {};
    renderActiveModels();
}

function renderActiveModels() {
    const container = document.getElementById('active-models-list');
    container.innerHTML = '';
    for (const [mid, cfg] of Object.entries(activeModels)) {
        const temp = cfg.temperature ?? 0.7;
        const topP = cfg.top_p ?? 0.9;
        const topK = cfg.top_k ?? 40;
        const repPen = cfg.repeat_penalty ?? 1.1;
        const host = cfg.ollama_host || 'http://localhost:11434';
        const card = document.createElement('div');
        card.className = 'model-card';
        card.style.flexWrap = 'wrap';
        card.innerHTML = `
            <span class="dot" style="background:${cfg.color}"></span>
            <div style="flex:1;min-width:0">
                <div class="name">${escapeHtml(cfg.display_name)}</div>
                <div class="model-id">${escapeHtml(mid)}</div>
            </div>
            <button class="settings-toggle" onclick="toggleModelSettings('${mid}')" title="Settings">&#9881;</button>
            <button class="remove-btn" onclick="removeModel('${mid}')" title="Remove">&times;</button>
            <div class="model-settings" id="settings-${CSS.escape(mid)}">
                <div class="param-group">
                    <label>Ollama Host</label>
                    <input type="text" value="${escapeHtml(host)}"
                        onchange="updateModelParam('${mid}','ollama_host',this.value)">
                </div>
                <div class="param-group">
                    <label>Color</label>
                    <input type="color" value="${cfg.color || '#6c9f5e'}" class="model-color-picker"
                        oninput="updateModelColor('${mid}', this.value, this.closest('.model-card'))">
                </div>
                <div class="param-group" title="Controls randomness. Higher = more creative. Lower = more focused. 0 = deterministic.">
                    <label>Temperature <span id="live-temp-${CSS.escape(mid)}">${temp}</span></label>
                    <input type="range" min="0" max="2" step="0.05" value="${temp}"
                        oninput="updateModelParam('${mid}','temperature',parseFloat(this.value));document.getElementById('live-temp-${CSS.escape(mid)}').textContent=this.value;updateSliderFill(this)">
                </div>
                <div class="param-group" title="Nucleus sampling. Lower = more focused. Higher = considers more options.">
                    <label>Top P <span id="live-topp-${CSS.escape(mid)}">${topP}</span></label>
                    <input type="range" min="0" max="1" step="0.05" value="${topP}"
                        oninput="updateModelParam('${mid}','top_p',parseFloat(this.value));document.getElementById('live-topp-${CSS.escape(mid)}').textContent=this.value;updateSliderFill(this)">
                </div>
                <div class="param-group" title="Limits to top K tokens. Lower = predictable. Higher = diverse.">
                    <label>Top K <span id="live-topk-${CSS.escape(mid)}">${topK}</span></label>
                    <input type="range" min="1" max="200" step="1" value="${topK}"
                        oninput="updateModelParam('${mid}','top_k',parseInt(this.value));document.getElementById('live-topk-${CSS.escape(mid)}').textContent=this.value;updateSliderFill(this)">
                </div>
                <div class="param-group" title="Penalizes repetition. Higher = less repetition. 1.0 = no penalty.">
                    <label>Repeat Penalty <span id="live-rep-${CSS.escape(mid)}">${repPen}</span></label>
                    <input type="range" min="1.0" max="2.0" step="0.05" value="${repPen}"
                        oninput="updateModelParam('${mid}','repeat_penalty',parseFloat(this.value));document.getElementById('live-rep-${CSS.escape(mid)}').textContent=this.value;updateSliderFill(this)">
                </div>
            </div>
        `;
        container.appendChild(card);
        initAllSliderFills(card);
    }
    updateModelCount();
}

function toggleModelSettings(modelId) {
    const el = document.getElementById('settings-' + CSS.escape(modelId));
    if (el) el.classList.toggle('open');
}

function updateModelColor(modelId, color, card) {
    if (activeModels[modelId]) {
        activeModels[modelId].color = color;
    }
    const dot = card.querySelector('.dot');
    if (dot) dot.style.background = color;
    clearTimeout(_updateTimers['color_' + modelId]);
    _updateTimers['color_' + modelId] = setTimeout(() => {
        safeSend({ action: 'update_model', model_id: modelId, color: color });
    }, 300);
}

let _updateTimers = {};
function updateModelParam(modelId, param, value) {
    if (activeModels[modelId]) {
        activeModels[modelId][param] = value;
    }
    const key = modelId + param;
    clearTimeout(_updateTimers[key]);
    _updateTimers[key] = setTimeout(() => {
        const msg = { action: 'update_model', model_id: modelId };
        msg[param] = value;
        safeSend(msg);
    }, 300);
}

function safeSend(data) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addSystemMessage('Not connected — reconnecting...');
        connect();
        setTimeout(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(data));
            } else {
                addSystemMessage('Still disconnected. Try again in a moment.');
            }
        }, 1500);
        return false;
    }
    try {
        ws.send(JSON.stringify(data));
        return true;
    } catch (e) {
        addSystemMessage('Send failed — reconnecting...');
        connect();
        return false;
    }
}

function addModel() {
    const sel = document.getElementById('model-select');
    const modelId = sel.value;
    if (!modelId) {
        addSystemMessage('Select a model first.');
        return;
    }

    if (Object.keys(activeModels).length >= 4) {
        addSystemMessage('Maximum 4 models allowed. Remove one first.');
        return;
    }

    const displayName = document.getElementById('model-display-name').value || modelId.split(':')[0];
    const color = document.getElementById('model-color').value;
    const systemPrompt = document.getElementById('model-system-prompt').value;
    const temperature = parseFloat(document.getElementById('model-temperature').value);
    const topP = parseFloat(document.getElementById('model-top-p').value);
    const topK = parseInt(document.getElementById('model-top-k').value);
    const repeatPenalty = parseFloat(document.getElementById('model-repeat-penalty').value);
    const ollamaHost = document.getElementById('ollama-host').value.trim() || 'http://localhost:11434';

    const payload = {
        action: 'add_model',
        model_id: modelId,
        display_name: displayName,
        system_prompt: systemPrompt,
        color: color,
        temperature: temperature,
        top_p: topP,
        top_k: topK,
        repeat_penalty: repeatPenalty,
        ollama_host: ollamaHost,
    };

    if (!safeSend(payload)) return;

    activeModels[modelId] = {
        display_name: displayName, system_prompt: systemPrompt, color: color,
        temperature, top_p: topP, top_k: topK, repeat_penalty: repeatPenalty,
        ollama_host: ollamaHost, memory: '',
    };
    renderActiveModels();
    updateModelCount();

    // Reset form
    document.getElementById('model-display-name').value = '';
    document.getElementById('model-system-prompt').value = '';
    document.getElementById('model-temperature').value = '0.7';
    document.getElementById('temp-val').textContent = '0.7';
    document.getElementById('model-top-p').value = '0.9';
    document.getElementById('topp-val').textContent = '0.9';
    document.getElementById('model-top-k').value = '40';
    document.getElementById('topk-val').textContent = '40';
    document.getElementById('model-repeat-penalty').value = '1.1';
    document.getElementById('rep-val').textContent = '1.1';
    initAllSliderFills(document.getElementById('add-model-section'));

    addSystemMessage(`Added ${displayName} to chat.`);
}

function updateModelCount() {
    const count = Object.keys(activeModels).length;
    const el = document.getElementById('model-count-indicator');
    if (el) el.textContent = `${count}/4 models active`;
}

function removeModel(modelId) {
    safeSend({ action: 'remove_model', model_id: modelId });
    delete activeModels[modelId];
    renderActiveModels();
    if (Object.keys(activeModels).length < 2 && autoChatActive) {
        safeSend({ action: 'auto_chat_stop' });
    }
}

// ═══ Messages ═══

function addMessage(name, content, color) {
    const container = document.getElementById('messages');
    const msg = createMessageElement(name, content, color);
    container.appendChild(msg);
    scrollToBottom();
}

function createMessageElement(name, content, color, isStreaming = false) {
    const msg = document.createElement('div');
    msg.className = 'message';

    const initial = name.charAt(0).toUpperCase();
    const rendered = isStreaming ? '' : renderMarkdown(content);

    msg.innerHTML = `
        <div class="avatar" style="background:${color}">${initial}</div>
        <div class="body">
            <div class="header">
                <span class="sender" style="color:${color}">${escapeHtml(name)}</span>
                <span class="timestamp">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="content">${rendered}</div>
        </div>
    `;
    return msg;
}

function addSystemMessage(text) {
    const container = document.getElementById('messages');
    const msg = document.createElement('div');
    msg.className = 'message system-msg';
    msg.innerHTML = `<div class="content">${escapeHtml(text)}</div>`;
    container.appendChild(msg);
    scrollToBottom();
}

// ═══ Thinking Indicators ═══

let thinkingElements = {};

function showThinking(modelId, name, color) {
    if (thinkingElements[modelId]) return;
    const container = document.getElementById('messages');
    const msg = createMessageElement(name, '', color, true);
    const contentEl = msg.querySelector('.content');
    contentEl.innerHTML = '<span class="thinking-text">thinking</span><span class="streaming-indicator"></span>';
    msg.classList.add('thinking-msg');
    msg.dataset.modelId = modelId;
    container.appendChild(msg);
    thinkingElements[modelId] = msg;
    scrollToBottom();
}

function removeThinking(modelId) {
    if (thinkingElements[modelId]) {
        thinkingElements[modelId].remove();
        delete thinkingElements[modelId];
    }
}

// ═══ Streaming ═══

function startStreaming(modelId, name, color) {
    const container = document.getElementById('messages');
    const msg = createMessageElement(name, '', color, true);
    const contentEl = msg.querySelector('.content');
    contentEl.innerHTML = '<span class="streaming-indicator"></span>';
    container.appendChild(msg);
    streamingMessages[modelId] = { element: msg, content: '', rawText: '' };
    scrollToBottom();
}

function appendToken(modelId, token) {
    const stream = streamingMessages[modelId];
    if (!stream) return;

    stream.rawText += token;
    const contentEl = stream.element.querySelector('.content');
    contentEl.innerHTML = renderMarkdown(stream.rawText) + '<span class="streaming-indicator"></span>';
    scrollToBottom();
}

function endStreaming(modelId) {
    const stream = streamingMessages[modelId];
    if (!stream) return;

    const cleanText = stream.rawText
        .replace(/<generate_image>[\s\S]*?<\/generate_image>/gi, '')
        .replace(/`?\[GENERATE_IMAGE:\s*.+?\]`?/g, '')
        .replace(/\*\*\s*\*\*/g, '')
        .replace(/```+\s*```*/g, '')
        .replace(/^\s*```+\s*$/gm, '')
        .trim();
    if (!cleanText) {
        stream.element.remove();
        delete streamingMessages[modelId];
        return;
    }
    const contentEl = stream.element.querySelector('.content');
    contentEl.innerHTML = renderMarkdown(cleanText);

    stream.element.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });

    delete streamingMessages[modelId];
}

// ═══ Rendering ═══

function renderMarkdown(text) {
    if (!text) return '';
    try {
        let html = marked.parse(text);
        html = html.replace(/<img /g, '<img onload="forceScrollToBottom()" onerror="this.style.display=\'none\'" ');
        return html;
    } catch (e) {
        return escapeHtml(text);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ═══ History ═══

function loadHistory(history) {
    if (!history || !history.length) return;
    const container = document.getElementById('messages');
    container.innerHTML = '';

    let i = 0;
    const batchSize = 3;
    function renderBatch() {
        const end = Math.min(i + batchSize, history.length);
        const fragment = document.createDocumentFragment();
        for (; i < end; i++) {
            const msg = history[i];
            const color = msg.role === 'user' ? '#00b4d8' : (activeModels[msg.role]?.color || '#888');
            const el = createMessageElement(msg.name, msg.content, color);
            fragment.appendChild(el);
        }
        container.appendChild(fragment);
        if (i < history.length) {
            requestAnimationFrame(renderBatch);
        } else {
            scrollToBottom();
        }
    }
    renderBatch();
}

// ═══ Input ═══

function sendMessage() {
    const input = document.getElementById('user-input');
    const content = input.value.trim();
    if (!content) return;

    if (safeSend({ action: 'send_message', content: content })) {
        input.value = '';
        input.style.height = 'auto';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('user-input');
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    });

    // Escape to kill (or close lightbox/settings)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            // Close settings modal first
            const settings = document.getElementById('settings-overlay-modal');
            if (settings && settings.classList.contains('open')) {
                closeSettings();
                return;
            }
            const lb = document.querySelector('.lightbox-overlay');
            if (lb) { lb.remove(); return; }
            sendKill();
        }
    });

    // Click images to open lightbox
    document.getElementById('messages').addEventListener('click', (e) => {
        if (e.target.tagName === 'IMG' && e.target.closest('.message .content')) {
            const overlay = document.createElement('div');
            overlay.className = 'lightbox-overlay';
            overlay.innerHTML = `<img src="${e.target.src}" alt="Full size">`;
            overlay.addEventListener('click', () => overlay.remove());
            document.body.appendChild(overlay);
        }
    });

    // Auto-refresh models when host input changes (debounced)
    const hostInput = document.getElementById('ollama-host');
    let hostTimer = null;
    hostInput.addEventListener('input', () => {
        clearTimeout(hostTimer);
        hostTimer = setTimeout(() => refreshModels(), 800);
    });

    // Init slider fills for the Add Model form
    initAllSliderFills(document.getElementById('add-model-section'));

    connect();
});

// ═══ Controls ═══

function saveProfile() {
    const profile = document.getElementById('user-profile').value;
    safeSend({ action: 'update_profile', profile: profile });
    // Visual feedback
    const btn = event.target;
    const orig = btn.textContent;
    btn.textContent = 'Saved!';
    btn.style.color = 'var(--success)';
    setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 1500);
}

function toggleAutoChat() {
    const toggle = document.getElementById('auto-chat-toggle');
    if (toggle.checked) {
        const promptEl = document.getElementById('auto-chat-prompt');
        const prompt = promptEl.value.trim();
        safeSend({ action: 'auto_chat_start', prompt: prompt });
        if (prompt) promptEl.value = '';
    } else {
        safeSend({ action: 'auto_chat_stop' });
    }
}

function toggleSection(header) {
    const content = header.nextElementSibling;
    const icon = header.querySelector('.collapse-icon');
    content.classList.toggle('collapsed');
    icon.textContent = content.classList.contains('collapsed') ? '+' : '\u2212';
    // Init slider fills when expanding
    if (!content.classList.contains('collapsed')) {
        initAllSliderFills(content);
    }
}

let autoScrollEnabled = true;

function toggleAutoScroll() {
    const toggle = document.getElementById('auto-scroll-toggle');
    autoScrollEnabled = toggle.checked;
    if (autoScrollEnabled) {
        userScrolled = false;
        scrollToBottom();
    }
}

function updateAutoChatButton() {
    const toggle = document.getElementById('auto-chat-toggle');
    if (toggle) {
        toggle.checked = autoChatActive;
    }
}

function sendKill() {
    safeSend({ action: 'kill' });
}

function clearHistory() {
    safeSend({ action: 'clear_history' });
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    sidebar.classList.toggle('open');
    overlay.classList.toggle('open');
}

// ═══ Scroll ═══

let userScrolled = false;

document.addEventListener('DOMContentLoaded', () => {
    const messages = document.getElementById('messages');
    messages.addEventListener('scroll', () => {
        const atBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 50;
        userScrolled = !atBottom;
    });
});

function scrollToBottom() {
    if (!autoScrollEnabled || userScrolled) return;
    const messages = document.getElementById('messages');
    messages.scrollTop = messages.scrollHeight;
}

function forceScrollToBottom() {
    if (!autoScrollEnabled) return;
    userScrolled = false;
    const messages = document.getElementById('messages');
    messages.scrollTop = messages.scrollHeight;
}
