const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const CONVERSATION_PAGE_SIZE = 40;

const state = {
  models: [], toolModels: [], personas: [], conversations: [], current: null, messages: [], hasMore: false,
  oldestId: null, attachments: [], generating: false, controller: null, generationStartedAt: 0,
  lounge: null, loungePoll: null, loungeView: localStorage.getItem('localai.loungeView') || 'overview', lastActivityPing: 0,
  screenDiagnosticRunning: false, ultimateUsage: null, composing: false,
  conversationTotal: 0, conversationHasMore: false, conversationQuery: '',
  conversationRequest: 0, conversationLoading: false,
  settings: {
    persona: localStorage.getItem('localai.persona') || 'aili',
    tier: localStorage.getItem('localai.tier') || '9b',
    model: '',
    temperature: Number(localStorage.getItem('localai.temperature') || 0.7),
    qualityMode: localStorage.getItem('localai.qualityMode') || 'balanced',
    topP: Number(localStorage.getItem('localai.topP') || 0.95),
    repeatPenalty: Number(localStorage.getItem('localai.repeatPenalty') || 1.1),
    seed: Number(localStorage.getItem('localai.seed') || 0),
    keepAlive: '1m',
    maxTokens: Number(localStorage.getItem('localai.maxTokens') || 0),
  },
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {'Content-Type': 'application/json', ...(options.headers || {})},
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).error?.message || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function escapeHTML(value = '') {
  return String(value).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function inlineMarkdown(text) {
  return text
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function markdown(source = '') {
  const fences = [];
  let text = String(source).replace(/```([^\n]*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const token = `\u0000CODE${fences.length}\u0000`;
    fences.push(`<pre data-lang="${escapeHTML(lang.trim())}"><code>${escapeHTML(code.trimEnd())}</code></pre>`);
    return token;
  });
  const lines = escapeHTML(text).split('\n');
  const output = [];
  let list = null;
  const closeList = () => { if (list) { output.push(`</${list}>`); list = null; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^\u0000CODE\d+\u0000$/.test(line)) { closeList(); output.push(line); continue; }
    if (!line.trim()) { closeList(); output.push(''); continue; }
    const heading = line.match(/^(#{1,3})\s+(.+)/);
    if (heading) { closeList(); const level = heading[1].length; output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`); continue; }
    const bullet = line.match(/^[-*]\s+(.+)/);
    const ordered = line.match(/^\d+[.)]\s+(.+)/);
    if (bullet || ordered) {
      const wanted = bullet ? 'ul' : 'ol';
      if (list !== wanted) { closeList(); list = wanted; output.push(`<${list}>`); }
      output.push(`<li>${inlineMarkdown((bullet || ordered)[1])}</li>`); continue;
    }
    if (line.startsWith('&gt; ')) { closeList(); output.push(`<blockquote>${inlineMarkdown(line.slice(5))}</blockquote>`); continue; }
    closeList(); output.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  closeList();
  return output.join('\n').replace(/\u0000CODE(\d+)\u0000/g, (_, index) => fences[Number(index)]);
}

function formatTime(value) {
  if (!value) return '';
  const date = new Date(value);
  const now = new Date();
  if (date.toDateString() === now.toDateString()) return date.toLocaleTimeString('zh-CN', {hour:'2-digit', minute:'2-digit'});
  return date.toLocaleDateString('zh-CN', {month:'numeric', day:'numeric'});
}

function toast(message, type = '') {
  const node = document.createElement('div');
  node.className = `toast ${type}`;
  node.textContent = message;
  $('#toastStack').append(node);
  setTimeout(() => { node.classList.add('out'); setTimeout(() => node.remove(), 220); }, 3200);
}

function selectedPersona() { return state.personas.find(persona => persona.id === state.settings.persona); }
function selectedModel() { return state.models.find(model => model.id === state.settings.model); }
function avatarURL(persona) { return `/app/assets/${persona === 'shaya' ? 'shaya' : 'aili'}-avatar.png`; }
function fullPortraitURL(persona) { return `/app/assets/${persona === 'shaya' ? 'shaya' : 'aili'}-full.png`; }
function tierLabel(tier) { return ({'4b':'极速档 · 4B','9b':'日用档 · 9B','27b':'高级档 · 27B','ultimate':'究极'})[String(tier).toLowerCase()] || String(tier || '').toUpperCase(); }
function syncSelectedModel() {
  const persona = selectedPersona();
  if (!persona) return;
  if (!persona.models[state.settings.tier]) state.settings.tier = '9b';
  state.settings.model = persona.models[state.settings.tier];
  localStorage.setItem('localai.persona', state.settings.persona);
  localStorage.setItem('localai.tier', state.settings.tier);
  localStorage.setItem('localai.model', state.settings.model);
}

async function bootstrap() {
  try {
    const data = await api('/api/gui/bootstrap');
    state.models = data.models;
    state.toolModels = data.tool_models || [];
    state.personas = data.personas;
    state.conversations = data.conversations;
    state.conversationTotal = Number(data.conversation_total ?? data.conversations.length);
    state.conversationHasMore = Boolean(data.conversation_has_more);
    state.ultimateUsage = data.ultimate_usage || null;
    if (!state.personas.some(persona => persona.id === state.settings.persona)) state.settings.persona = 'aili';
    syncSelectedModel();
    renderPersonaControls(); updateQualityControls(); renderSessions(); renderModelGrid(); updateStorage(data.storage); updateUltimateUsage(state.ultimateUsage); updateServiceStatus(data.ollama_online);
    const preferredConversation = state.conversations.find(item => item.persona === state.settings.persona);
    if (preferredConversation) await loadConversation(preferredConversation.id);
    else showWelcome();
  } catch (error) {
    updateServiceStatus(false);
    toast(`初始化失败：${error.message}`, 'error');
  }
}

function updateServiceStatus(online) {
  const ultimate=Boolean(state.ultimateUsage?.available);
  $('#apiStatusDot').className = `status-dot ${(online||ultimate) ? 'online' : 'offline'}`;
  $('#apiStatusText').textContent = online ? (ultimate?'本地与究极已就绪':'本地服务已就绪') : (ultimate?'究极已就绪 · 本地离线':'Ollama 未连接');
}

function renderPersonaControls() {
  $('#personaSwitch').innerHTML = state.personas.map(persona => `<button class="persona-button ${persona.id === state.settings.persona ? 'active' : ''} ${persona.id}" data-persona="${persona.id}"><img src="${avatarURL(persona.id)}" alt=""><div><b>${escapeHTML(persona.name)}</b><small>${escapeHTML(persona.subtitle)}</small></div></button>`).join('');
  $$('.persona-button').forEach(button => button.onclick = () => switchPersona(button.dataset.persona));
  $('#tierSelect').value = state.settings.tier;
  $('#tierSelectValue').textContent = tierLabel(state.settings.tier);
  updateModelBadge();
  updateQualityControls();
}

function updateModelBadge() {
  const model = selectedModel(); if (!model) return;
  const persona = selectedPersona();
  $('#modelOrb').classList.toggle('uncensored', state.settings.persona === 'aili');
  $('#modelControlLabel').textContent = `${persona?.name || ''}的模型档位`;
  $('#contextPill').textContent = `${Math.round(model.context / 1024)}K 上下文`;
  $('#welcomeTitle').textContent = `${persona?.name || '她'}在这儿陪你`;
  document.body.classList.toggle('persona-aili', state.settings.persona === 'aili');
  document.body.classList.toggle('persona-shaya', state.settings.persona === 'shaya');
  document.body.classList.toggle('tier-ultimate', state.settings.tier === 'ultimate');
  $('#welcomePortrait').src = fullPortraitURL(state.settings.persona);
  $('#welcomePortrait').alt = persona?.name || '';
  $('#profileGlimpse').textContent = state.settings.persona === 'aili'
    ? '我这人就是不太会端着啦，想到哪聊到哪。你想说啥就说啥呀💗'
    : '要说就说清楚一点……我会认真听的。还有，别、别突然夸我。';
  $('#welcomeDescription').textContent = state.settings.persona === 'aili'
    ? '有啥就说啥呗，聊到哪算哪～别那么拘谨。'
    : '我会认真帮你理好的。所、所以别敷衍就好。';
}

function updateQualityControls() {
  const ultimate = state.settings.tier === 'ultimate';
  $('#qualitySwitch').classList.toggle('hidden', ultimate);
  $('#qualityExplainer').classList.toggle('hidden', ultimate);
  $$('#qualitySwitch button').forEach(button => button.classList.toggle('active', button.dataset.quality === state.settings.qualityMode));
}

function qualityLabel(mode) { return ({fast:'快速', balanced:'均衡', deep:'深度', ultimate:'究极'})[mode] || '均衡'; }

function metricsHTML(metadata = {}) {
  const parts = [];
  if (metadata.ultimate) parts.push('究极');
  else if (metadata.quality_mode) parts.push(qualityLabel(metadata.quality_mode));
  if (metadata.tokens_per_second) parts.push(`${Number(metadata.tokens_per_second).toFixed(1)} tok/s`);
  if (metadata.first_token_seconds) parts.push(`首字 ${Number(metadata.first_token_seconds).toFixed(1)}s`);
  if (metadata.context_tokens && metadata.context_limit) parts.push(`上下文 ${metadata.context_tokens}/${metadata.context_limit}`);
  if (metadata.memory_recall_count) parts.push(`召回 ${metadata.memory_recall_count} 条`);
  if (metadata.local_tool_used) parts.push('本地工具校验');
  if (metadata.local_vision_proxy) parts.push('本地视觉预读');
  if (metadata.cost_cny !== undefined) parts.push(`本轮 ¥${Number(metadata.cost_cny).toFixed(6)}`);
  if (metadata.cache_hit_rate) parts.push(`缓存命中 ${Math.round(Number(metadata.cache_hit_rate) * 100)}%`);
  if (metadata.interrupted) parts.push('连接中断');
  return parts.join(' · ');
}

function updateRuntimeInsights(metadata = {}) {
  const recall = Number(metadata.memory_recall_count || 0);
  $('#memoryRecallChip').textContent = recall ? `已召回 ${recall} 条长期原文` : '本轮无需旧原文';
  if (metadata.context_tokens && metadata.context_limit) {
    const percent = Math.min(100, Math.round(metadata.context_tokens / metadata.context_limit * 100));
    $('#contextUsageChip').textContent = `上下文 ${percent}% · ${metadata.context_tokens}/${metadata.context_limit}`;
  } else $('#contextUsageChip').textContent = '上下文待命';
}

async function switchPersona(persona) {
  if (state.generating || persona === state.settings.persona) return;
  state.settings.persona = persona; syncSelectedModel(); renderPersonaControls(); fillMemoryEditor();
  showWelcome();
  toast(`已切换到${selectedPersona()?.name || ''}，对话不会串台`, 'success');
}

async function switchTier(tier) {
  const nextTier=String(tier).toLowerCase();
  const previousTier=state.settings.tier;
  if(nextTier===previousTier)return;
  if(state.generating){renderPersonaControls();return toast('请先停止当前回答再切换档位','error');}
  const persona=selectedPersona();
  if(!persona?.models?.[nextTier]){renderPersonaControls();return toast('这个人格没有该档位','error');}
  state.settings.tier=nextTier;syncSelectedModel();renderPersonaControls();
  if(nextTier==='ultimate'){
    document.body.classList.remove('ultimate-burst');
    requestAnimationFrame(()=>document.body.classList.add('ultimate-burst'));
    setTimeout(()=>document.body.classList.remove('ultimate-burst'),1400);
  }
  try{
    if(state.current){
      const updated=await api(`/api/gui/conversations/${state.current.id}`,{method:'PATCH',body:JSON.stringify({model:state.settings.model})});
      state.current={...state.current,...updated};
      state.conversations=state.conversations.map(item=>item.id===state.current.id?{...item,...updated}:item);
      await loadConversation(state.current.id,true);
    }else{
      renderSessions();
    }
    toast(`已切换到${tierLabel(nextTier)}，当前对话继续`,'success');
  }catch(error){
    state.settings.tier=previousTier;syncSelectedModel();renderPersonaControls();
    toast(`切换失败：${error.message}`,'error');
  }
}

function renderSessions() {
  const query = $('#sessionSearch').value.trim().toLowerCase();
  const items = state.conversations;
  $('#sessionCount').textContent = state.conversationTotal > items.length ? `${items.length}/${state.conversationTotal}` : state.conversationTotal;
  $('#sessionList').innerHTML = items.length ? items.map(item => `
    <button class="session-item ${state.current?.id === item.id ? 'active' : ''}" data-session="${item.id}">
      <img class="session-icon ${item.persona}" src="${avatarURL(item.persona)}" alt="">
      <span class="session-copy"><b>${escapeHTML(item.title)}</b><small>${escapeHTML(item.persona_name)} · ${escapeHTML(tierLabel(item.tier))} · ${item.message_count || 0} 条 · ${formatTime(item.updated_at)}</small></span>
    </button>`).join('') : `<div class="empty-sessions">${query ? '没有找到相关对话' : '还没有对话<br>点击上方开始'}</div>`;
  $('#loadMoreSessions').classList.toggle('hidden', !state.conversationHasMore);
  $('#loadMoreSessions').disabled = state.conversationLoading;
  $('#loadMoreSessions').textContent = state.conversationLoading ? '正在加载…' : '再加载 40 条';
  $$('.session-item').forEach(button => button.onclick = () => loadConversation(Number(button.dataset.session)));
}

function showWelcome() {
  state.current = null; state.messages = [];
  $('#welcomeView').classList.remove('hidden'); $('#messages').innerHTML = '';
  $('#loadOlderButton').classList.add('hidden'); renderSessions(); updateConversationActions(); fillMemoryEditor();
}

async function createConversation() {
  if ($('#sessionSearch').value) {
    $('#sessionSearch').value = '';
    state.conversationQuery = '';
  }
  const conversation = await api('/api/gui/conversations', {method:'POST', body: JSON.stringify({persona:state.settings.persona, model: state.settings.model, title:'新对话'})});
  state.conversations.unshift({...conversation, message_count:0, preview:''});
  state.conversationTotal += 1;
  state.current = conversation; state.messages = []; state.hasMore = false; state.oldestId = null;
  $('#welcomeView').classList.add('hidden'); $('#messages').innerHTML = '';
  renderSessions(); updateConversationActions(); fillMemoryEditor(); $('#messageInput').focus();
  return conversation;
}

async function loadConversation(id, preserveScroll = false) {
  if (state.generating) return toast('请先停止当前生成', 'error');
  const data = await api(`/api/gui/conversations/${id}?limit=300`);
  state.current = data.conversation; state.messages = data.messages; state.hasMore = data.has_more; state.oldestId = data.oldest_id;
  state.settings.persona = state.current.persona; state.settings.tier = state.current.tier; syncSelectedModel();
  renderPersonaControls();
  $('#welcomeView').classList.add('hidden'); renderMessages(); renderSessions(); updateConversationActions(); fillMemoryEditor();
  if (!preserveScroll) scrollBottom(false);
}

async function loadOlder() {
  if (!state.current || !state.hasMore) return;
  const scroll = $('#chatScroll'); const previousHeight = scroll.scrollHeight;
  const data = await api(`/api/gui/conversations/${state.current.id}?limit=300&before=${state.oldestId}`);
  state.messages = [...data.messages, ...state.messages]; state.hasMore = data.has_more; state.oldestId = data.oldest_id;
  renderMessages(); scroll.scrollTop = scroll.scrollHeight - previousHeight;
}

function renderMessages() {
  $('#messages').innerHTML = state.messages.map(message => messageHTML(message)).join('');
  $('#loadOlderButton').classList.toggle('hidden', !state.hasMore);
  bindImagePreviews(); bindMessageActions();
}

function messageHTML(message, streaming = false) {
  const isUser = message.role === 'user';
  const persona = selectedPersona();
  const images = (message.attachments || []).map(item => `<img src="${escapeHTML(item.url || item.data)}" alt="${escapeHTML(item.name || '图片')}" data-preview>`).join('');
  const metricText = metricsHTML(message.metadata || {});
  const messageId = message.id || '';
  const generatedModel=(message.metadata||{}).model;
  const generatedTier=(message.metadata||{}).tier||state.models.find(item=>item.id===generatedModel)?.tier_id||state.settings.tier;
  return `<article class="message ${isUser ? 'user' : 'assistant'}" data-message-id="${message.id || ''}">
    <div class="message-avatar ${isUser ? 'user-avatar' : state.settings.persona}">${isUser ? '<span>你</span>' : `<img src="${avatarURL(state.settings.persona)}" alt="${escapeHTML(persona?.name || 'AI')}">`}</div>
    <div><div class="message-head"><b>${isUser ? '你' : escapeHTML(persona?.name || '本地模型')}</b><span class="message-tier">${isUser ? '' : escapeHTML(tierLabel(generatedTier))}</span><time>${formatTime(message.created_at || new Date())}</time></div>
    ${images ? `<div class="message-images">${images}</div>` : ''}
    <div class="message-content ${streaming ? 'streaming-caret' : ''}">${markdown(message.content || '')}</div>
    <div class="message-footer"><span class="message-metrics">${escapeHTML(metricText)}</span><span class="message-actions"><button data-message-action="copy" data-message-id="${messageId}">复制</button>${isUser ? '' : `<button data-message-action="retry" data-message-id="${messageId}">重新回答</button><button data-message-action="continue" data-message-id="${messageId}">继续</button>`}</span></div></div>
  </article>`;
}

function bindImagePreviews() {
  $$('[data-preview]').forEach(image => image.onclick = () => { $('#lightboxImage').src = image.src; $('#imageLightbox').classList.remove('hidden'); });
}

function bindMessageActions() {
  $$('[data-message-action]').forEach(button => button.onclick = () => {
    const message = state.messages.find(item => String(item.id || '') === button.dataset.messageId);
    if (!message) return;
    if (button.dataset.messageAction === 'copy') return copyText(message.content || '');
    if (state.generating) return toast('请先等待当前回答结束', 'error');
    const prompt = button.dataset.messageAction === 'retry'
      ? '请重新检查并完整回答我上一条问题。事实和计算正确优先，不要复述错误答案。'
      : '请从上一条回答结束处继续，不要重复已经说过的内容。';
    $('#messageInput').value = prompt; autoResize(); sendMessage();
  });
}

function scrollBottom(smooth = true) { $('#chatScroll').scrollTo({top: $('#chatScroll').scrollHeight, behavior: smooth ? 'smooth' : 'auto'}); }
function updateConversationActions() { ['renameButton','exportButton','deleteButton'].forEach(id => $(`#${id}`).disabled = !state.current); }

async function sendMessage() {
  if (state.generating) return stopGeneration();
  if(state.composing)return;
  const text = $('#messageInput').value.trim();
  if (!text && !state.attachments.length) return;
  if (!state.current) await createConversation();
  if (state.current.model !== state.settings.model || state.current.persona !== state.settings.persona) await createConversation();
  const outgoingImages = state.attachments.map(item => ({name:item.name, data:item.data}));
  const optimistic = {role:'user', content:text || '请分析我上传的图片。', attachments:state.attachments, created_at:new Date().toISOString()};
  state.messages.push(optimistic); renderMessages();
  $('#messageInput').value = ''; autoResize(); clearAttachments();
  $('#welcomeView').classList.add('hidden');
  const placeholder = {role:'assistant', content:'', attachments:[], created_at:new Date().toISOString()};
  state.messages.push(placeholder); renderMessages(); scrollBottom();
  const article = $('#messages').lastElementChild; const content = article.querySelector('.message-content'); content.classList.add('streaming-caret');
  setGenerating(true, state.settings.tier === 'ultimate' ? '正在唤醒究极智慧…' : '正在准备模型…');
  state.controller = new AbortController();
  let sawDone = false; let recoverStream = false;
  updateRuntimeInsights({});
  try {
    const response = await fetch('/api/gui/chat', {
      method:'POST', headers:{'Content-Type':'application/json'}, signal:state.controller.signal,
      body:JSON.stringify({conversation_id:state.current.id, persona:state.settings.persona, model:state.settings.model, message:text, images:outgoingImages, temperature:state.settings.temperature, quality_mode:state.settings.tier==='ultimate'?'ultimate':state.settings.qualityMode, top_p:state.settings.topP, repeat_penalty:state.settings.repeatPenalty, seed:state.settings.seed, keep_alive:'1m', max_tokens:state.settings.maxTokens || undefined, client_surface:'main_panel'}),
    });
    if (!response.ok) throw new Error((await response.json()).error?.message || `HTTP ${response.status}`);
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const events = buffer.split('\n\n'); buffer = events.pop() || '';
      for (const block of events) {
        const line = block.split('\n').find(item => item.startsWith('data: ')); if (!line) continue;
        const event = JSON.parse(line.slice(6));
        if (event.type === 'status') setStatus(event.message);
        if (event.type === 'memory_recall') $('#memoryRecallChip').textContent = event.count ? `已召回 ${event.count} 条长期原文` : '本轮无需旧原文';
        if (event.type === 'delta') { placeholder.content += event.content; content.innerHTML = markdown(placeholder.content); scrollBottom(); }
        if (event.type === 'done') {
          sawDone = true; placeholder.id = event.assistant_message_id; placeholder.metadata = event.metrics || {};
          updateRuntimeInsights(placeholder.metadata);
          if (event.finish_reason === 'length') toast('已达单次输出上限，发送“继续”可衔接', 'error');
        }
        if (event.type === 'error') throw new Error(event.message);
      }
    }
    if (!sawDone) throw new Error('回答流意外中断');
    content.classList.remove('streaming-caret');
    renderMessages(); scrollBottom(false);
    await refreshConversations();
    if(placeholder.metadata?.ultimate){api('/api/gui/ultimate-usage').then(data=>{state.ultimateUsage=data;updateUltimateUsage(data);}).catch(()=>{});}
  } catch (error) {
    content.classList.remove('streaming-caret');
    if (error.name === 'AbortError') { placeholder.content = placeholder.content || '[生成已停止]'; content.innerHTML = markdown(placeholder.content); toast('已停止生成'); }
    else { recoverStream = true; placeholder.content = placeholder.content || `[请求失败] ${error.message}`; content.innerHTML = markdown(placeholder.content); toast(`${error.message}，正在从本地记录恢复`, 'error'); }
  } finally {
    setGenerating(false); state.controller = null;
    if (recoverStream && state.current) {
      try { await loadConversation(state.current.id); } catch (_) {}
    }
    syncPersonaFromStorage();
  }
}

function setGenerating(value, text = '正在生成…') {
  state.generating = value; if (value) state.generationStartedAt = Date.now(); $('#sendButton').classList.toggle('stopping', value); $('#generationStatus').classList.toggle('hidden', !value); $('#generationStatusText').textContent = text;
}
function setStatus(text) { $('#generationStatusText').textContent = text; }
function stopGeneration() { if (Date.now() - state.generationStartedAt < 750) return; state.controller?.abort(); }

async function refreshConversations({append=false, query=$('#sessionSearch').value.trim()}={}) {
  const request = ++state.conversationRequest;
  state.conversationLoading = true;
  if (append) renderSessions();
  const offset = append ? state.conversations.length : 0;
  const parameters = new URLSearchParams({limit:String(CONVERSATION_PAGE_SIZE),offset:String(offset)});
  if (query) parameters.set('q',query);
  try {
    const data = await api(`/api/gui/conversations?${parameters}`);
    if (request !== state.conversationRequest) return;
    state.conversationQuery = query;
    if (append) {
      const known = new Set(state.conversations.map(item=>item.id));
      state.conversations.push(...data.data.filter(item=>!known.has(item.id)));
    } else state.conversations = data.data;
    state.conversationTotal = Number(data.total ?? state.conversations.length);
    state.conversationHasMore = Boolean(data.has_more);
  } finally {
    if (request === state.conversationRequest) {
      state.conversationLoading = false;
      renderSessions();
    }
  }
  if (state.current) state.current = state.conversations.find(item => item.id === state.current.id) || state.current;
  renderSessions(); fillMemoryEditor();
}

function searchConversations() {
  clearTimeout(searchConversations.timer);
  searchConversations.timer = setTimeout(
    ()=>refreshConversations().catch(error=>toast(`搜索失败：${error.message}`,'error')),
    180,
  );
}

function autoResize() { const input=$('#messageInput'); input.style.height='auto'; input.style.height=Math.min(input.scrollHeight,180)+'px'; $('#charCount').textContent=input.value.length ? `${input.value.length} 字` : ''; }

async function processImage(file) {
  if (!file.type.startsWith('image/')) throw new Error(`${file.name} 不是图片`);
  const original = await new Promise((resolve,reject) => { const reader=new FileReader(); reader.onload=()=>resolve(reader.result); reader.onerror=reject; reader.readAsDataURL(file); });
  const image = await new Promise((resolve,reject) => { const img=new Image(); img.onload=()=>resolve(img); img.onerror=reject; img.src=original; });
  const maxDimension = 1800; const scale = Math.min(1, maxDimension / Math.max(image.width,image.height));
  if (scale === 1 && file.size < 3_000_000) return {name:file.name, data:original, size:file.size};
  const canvas=document.createElement('canvas'); canvas.width=Math.round(image.width*scale); canvas.height=Math.round(image.height*scale); canvas.getContext('2d').drawImage(image,0,0,canvas.width,canvas.height);
  const mime=file.type==='image/png'?'image/png':'image/jpeg'; const data=canvas.toDataURL(mime,.88);
  return {name:file.name, data, size:Math.round(data.length*.75)};
}

async function addFiles(files) {
  const candidates=[...files].filter(file=>file.type.startsWith('image/')).slice(0,4-state.attachments.length);
  if (!candidates.length) return toast('请选择图片文件', 'error');
  setStatus('正在处理图片…');
  try { for (const file of candidates) state.attachments.push(await processImage(file)); renderAttachments(); }
  catch(error){ toast(error.message,'error'); }
}
function renderAttachments() { const strip=$('#attachmentStrip'); strip.classList.toggle('hidden',!state.attachments.length); strip.innerHTML=state.attachments.map((item,index)=>`<div class="attachment-preview"><img src="${item.data}" alt="${escapeHTML(item.name)}"><button data-remove="${index}">×</button></div>`).join(''); $$('[data-remove]').forEach(btn=>btn.onclick=()=>{state.attachments.splice(Number(btn.dataset.remove),1);renderAttachments();}); }
function clearAttachments(){ state.attachments=[]; renderAttachments(); $('#imageInput').value=''; }

function openSettings(tab='general') { $('#drawerBackdrop').classList.remove('hidden'); $('#settingsDrawer').classList.add('open'); switchTab(tab); refreshSettings(); }
function closeSettings(){ $('#drawerBackdrop').classList.add('hidden'); $('#settingsDrawer').classList.remove('open'); if(state.loungePoll){clearInterval(state.loungePoll);state.loungePoll=null;} }
function switchTab(tab){
  $$('.settings-tabs button').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
  $$('.settings-page').forEach(p=>p.classList.toggle('active',p.dataset.page===tab));
  $('.settings-content').scrollTop=0;
  if(state.loungePoll){clearInterval(state.loungePoll);state.loungePoll=null;}
  if(tab==='lounge'){
    setLoungeView(state.loungeView,false);
    refreshLounge().catch(e=>toast(e.message,'error'));
    state.loungePoll=setInterval(()=>refreshLounge(true).catch(()=>{}),15000);
  }
}
async function refreshSettings(){
  nativeBridge('getPreferences');
  fillMemoryEditor();
  try {
    const [storage, personaData, usage] = await Promise.all([api('/api/gui/storage'), api('/v1/personas'), api('/api/gui/ultimate-usage')]);
    updateStorage(storage); state.personas = personaData.data; state.ultimateUsage=usage; updateUltimateUsage(usage); fillMemoryEditor(); renderPersonaControls();
  } catch (_) {}
  renderModelGrid();
}
function fillMemoryEditor(){
  const persona=selectedPersona(); if(!persona)return;
  $('#memoryPersonaAvatar').src=avatarURL(persona.id);
  $('#memoryPersonaAvatar').alt=persona.name;
  $('#memoryIdentityCard').classList.toggle('shaya',persona.id==='shaya');
  $('#memoryPersonaTitle').textContent=`${persona.name}的跨会话记忆`;
  $('#memoryStatus').textContent=persona.pending_messages ? `${persona.pending_messages} 条新消息正在等待后台整理` : `四个档位共享 · 最近更新 ${formatTime(persona.updated_at)}`;
  $('#personaProfileInput').value=persona.profile||'';
  $('#personaSystemPromptInput').value=persona.core_prompt||persona.system_prompt||'';
  $('#personaMemoryInput').value=persona.memory||'';
  $('#sessionMemoryBlock').classList.toggle('hidden',!state.current);
  if(state.current){$('#conversationTitleInput').value=state.current.title||'';$('#systemPromptInput').value=state.current.system_prompt||'';$('#summaryInput').value=state.current.summary||'';}
}

function renderModelGrid(){
  const chatCards=state.models.map(model=>`<div class="model-card ${model.recommended?'recommended':''} ${model.cloud?'ultimate-model-card':''}"><div class="model-card-head"><h3>${escapeHTML(model.persona_name)} · ${escapeHTML(model.tier)}</h3><span class="status-dot ${(model.loaded||(model.cloud&&model.installed))?'online':''}"></span></div><p>${model.cloud?'智慧加速 · 本地视觉中继 · 不占本机模型内存':escapeHTML(model.id)}</p><div class="model-card-tags"><span>${Math.round(model.context/1024)}K</span><span>${model.cloud?'本地读图':'视觉'}</span><span>工具</span><span>思考</span>${model.cloud?`<span>${model.installed?'已就绪':'未配置'}</span>`:`<span>${escapeHTML(model.parameter_size||'')} · ${escapeHTML(model.quantization||'')}</span>`}${model.loaded?`<span>已载入 ${escapeHTML(model.loaded_size_text||'')}</span>`:''}${model.recommended?'<span>推荐档</span>':''}</div>${model.loaded?`<button class="secondary-action" data-unload="${escapeHTML(model.id)}">释放内存</button>`:''}</div>`).join('');
  const toolCards=state.toolModels.map(model=>`<div class="model-card tool-model"><div class="model-card-head"><h3>${escapeHTML(model.label)}</h3><span class="status-dot ${model.loaded?'online':''}"></span></div><p>${escapeHTML(model.description)}</p><div class="model-card-tags"><span>${escapeHTML(model.role||'幕后工具')}</span><span>${model.installed?'已安装':'未安装'}</span><span>${model.size_text||'—'}</span></div>${model.loaded?`<button class="secondary-action" data-unload="${escapeHTML(model.id)}">释放内存</button>`:''}</div>`).join('');
  $('#modelGrid').innerHTML=chatCards+toolCards;
  $$('[data-unload]').forEach(button=>button.onclick=async()=>{try{await api('/api/gui/model/unload',{method:'POST',body:JSON.stringify({model:button.dataset.unload})});toast('已释放模型内存','success');await bootstrap();openSettings('models');}catch(e){toast(e.message,'error');}});
}
function updateUltimateUsage(usage){
  if(!usage||!$('#ultimateTodayTokens'))return;
  const background=usage.background||{};const today=usage.today||{};const all=usage.all_time||{};
  const number=value=>Number(value||0).toLocaleString('zh-CN');
  const money=value=>`¥${Number(value||0).toFixed(6)}`;
  $('#ultimateTodayTokens').textContent=number(background.used);
  $('#ultimateRemaining').textContent=number(background.remaining);
  $('#ultimateCacheRate').textContent=`${Math.round(Number(today.cache_hit_rate||0)*100)}%`;
  $('#ultimateTodayCost').textContent=money(today.estimated_cost_cny);
  $('#ultimateTotalCost').textContent=money(all.estimated_cost_cny);
  $('#ultimateQuotaBar').style.width=`${Math.min(100,Number(background.percent||0))}%`;
  $('#ultimateUsageNote').textContent=!usage.available?'究极未就绪，后台保持本地逻辑。':background.exhausted?'今日后台额度已用完，已自动回落本地；明日自动恢复。':'额度内后台任务优先使用究极；价格按当前公布基础价估算。';
}
function updateStorage(storage){ if(!storage)return; $('#storageTotal').textContent=storage.total_size;$('#storageSessions').textContent=storage.sessions;$('#storageMessages').textContent=storage.messages;$('#storageUploads').textContent=storage.uploads_size;$('#storageDatabase').textContent=storage.database_size;$('#storagePath').textContent=storage.data_path; }

function formatDuration(seconds){
  const value=Math.max(0,Number(seconds)||0);
  if(value<60)return `${Math.round(value)} 秒`;
  if(value<3600)return `${Math.floor(value/60)} 分钟`;
  return `${Math.floor(value/3600)} 小时 ${Math.floor(value%3600/60)} 分`;
}
function setLoungeView(view,scrollToTop=true){
  const allowed=['overview','screen','history'];
  state.loungeView=allowed.includes(view)?view:'overview';
  localStorage.setItem('localai.loungeView',state.loungeView);
  $$('[data-lounge-view]').forEach(button=>{
    const active=button.dataset.loungeView===state.loungeView;
    button.classList.toggle('active',active);
    button.setAttribute('aria-selected',String(active));
  });
  $$('[data-lounge-panel]').forEach(panel=>panel.classList.toggle('active',panel.dataset.loungePanel===state.loungeView));
  if(scrollToTop)$('.settings-content').scrollTop=0;
}
function loungeStatusLabel(status){return ({completed:'已完成',running:'交流中',interrupted:'已让出资源',failed:'本轮失败'})[status]||status;}
function renderScreenWatchLog(history=[]){
  const log=$('#screenWatchLog');if(!log)return;
  if(!history.length){log.innerHTML='<div class="lounge-empty">尚无屏幕观察记录。图片不会保存，这里只显示两个人格留下的文字经历。</div>';return;}
  log.innerHTML=history.map(item=>{
    const complete=item.status==='completed';const retained=item.image_retained?'异常：仍有图像':'截图已释放 · 仅文字入池';
    const observers=complete?`<div class="screen-observer aili"><img src="${avatarURL('aili')}" alt=""><p>${escapeHTML(item.aili_observation||'')}</p></div><div class="screen-observer shaya"><img src="${avatarURL('shaya')}" alt=""><p>${escapeHTML(item.shaya_observation||'')}</p></div>`:`<div class="screen-error">${escapeHTML(item.error||'观察尚未完成')}</div>`;
    const discussion=item.metadata?.discussion||{};const discussionText=discussion.session_id?` · 已转入茶话室讨论 ${Number(discussion.messages||0)} 条${discussion.completed?'':'（中断）'}`:'';
    const displayCount=Number(item.metadata?.display_count||item.metadata?.displays?.length||1);
    return `<article class="screen-log-card"><div class="screen-log-head"><b>${escapeHTML(formatTime(item.captured_at))} · 全屏视觉9B→质量讨论${escapeHTML(String(item.model_tier||'9b').toUpperCase())}</b><small>${escapeHTML(`${displayCount} 个显示器 · `+retained+discussionText)}</small></div>${observers}</article>`;
  }).join('');
}
function renderMemoryPoolCard(persona,pool={}){
  const prefix=persona==='aili'?'aili':'shaya';const sources=pool.sources||{};
  $(`#${prefix}PoolCount`).textContent=`${pool.total_items??0} 条经历`;
  const observed=Number(sources.lounge_conversation||0)+Number(sources.lounge_message||0)+Number(sources.file_observation||0)+Number(sources.screen_observation||0)+Number(sources.screen_daily_digest||0);
  $(`#${prefix}PoolDetail`).textContent=`对话 ${pool.chat_messages??0} · 自主经历 ${observed} · 已索引 ${pool.indexed_items??0}${Number(pool.quarantined_items||0)?` · 隔离 ${pool.quarantined_items}`:''}`;
}
function renderLoungeLog(history=[]){
  const log=$('#loungeLog');
  if(!history.length){log.innerHTML='<div class="lounge-empty">尚无茶话室记录。达到空闲条件后，她们会在这里留下记录。</div>';return;}
  log.innerHTML=history.map(session=>{
    const observed=session.observations?.length?`<div class="lounge-observations"><b>本轮只读观察 ${session.observations.length} 个文件</b>${session.observations.map(item=>`<div class="lounge-file"><span>⌘</span><code>${escapeHTML(item.path)}</code><small>修改 ${escapeHTML(formatTime(item.modified_at))} · 观察 ${escapeHTML(formatTime(item.observed_at))}${item.error?` · ${escapeHTML(item.error)}`:''}</small></div>`).join('')}</div>`:'';
    const dialogue=session.messages?.length?`<div class="lounge-dialogue">${session.messages.map(item=>`<div class="lounge-line ${item.speaker}"><img src="${avatarURL(item.speaker)}" alt=""><div><p>${escapeHTML(item.content)}</p><small>${escapeHTML(formatTime(item.created_at))} · ${item.served_by==='ultimate'?'究极服务生成':escapeHTML(item.model)}${item.served_by==='ultimate'?` · 名义档位 ${escapeHTML(item.model)}`:''}${item.metadata?.fallback?' · 同人格备用模型':''}</small></div></div>`).join('')}</div>`:'';
    const resources=session.resources||{};
    const trigger=session.trigger_type==='manual'?'手动启动':(session.trigger_type==='screen'?'屏幕观察触发':'自主空闲');
    const topicMode=session.topic_mode==='free'?'随性闲聊':(session.topic_mode==='memory'?'回忆旧话题':(session.topic_mode==='resume'?'续接被打断的话题':(session.topic_mode==='screen'?'共同看屏幕':'随手发现')));
    const ending=session.termination_reason?`<div class="lounge-summary">${session.status==='interrupted'?'中断原因':'结束方式'}：${escapeHTML(session.termination_reason)}</div>`:'';
    const quarantine=session.quality_status==='quarantined'?`<div class="lounge-summary">旧错误记录已隔离，不再进入记忆检索：${escapeHTML(session.quality_reason||'身份或事实错位')}</div>`:'';
    return `<article class="lounge-log-card"><div class="lounge-log-head"><div><b>${escapeHTML(formatTime(session.started_at))} · ${trigger} · ${topicMode} · ${escapeHTML(String(session.model_tier).toUpperCase())}</b><small>${session.messages?.length||0} 条发言 · 启动时内存 ${escapeHTML(resources.memory_free_percent??'--')}% · CPU ${Math.round(Number(resources.load_ratio||0)*100)}%</small></div><span class="lounge-status-tag ${escapeHTML(session.quality_status==='quarantined'?'failed':session.status)}">${escapeHTML(session.quality_status==='quarantined'?'已隔离':loungeStatusLabel(session.status))}</span></div>${observed}${dialogue}${quarantine}${ending}${session.summary&&session.summary!==session.termination_reason?`<div class="lounge-summary">共同回忆：${escapeHTML(session.summary)}</div>`:''}</article>`;
  }).join('');
}
function renderLounge(payload,preserveForm=false){
  state.lounge=payload;const config=payload.config||{};const resources=payload.resources||{};
  const running=Boolean(payload.running);const screenRunning=Boolean(payload.screen_running);const status=running?'艾莉和沙雅正在后台交流':(screenRunning?'艾莉和沙雅正在观察屏幕':(config.last_status||payload.eligibility_reason||'等待空闲'));
  $('#loungeStatus').textContent=status;$('#loungePulse').className=`lounge-pulse ${running?'running':(payload.eligible?'ready':'')}`;
  $('#loungeTier').textContent=running?String(payload.active_tier||payload.selected_tier||'auto').toUpperCase():(config.model_strategy==='auto'?`AUTO→${String(payload.selected_tier||'4b').toUpperCase()}`:String(config.model_strategy||'4b').toUpperCase());
  $('#loungeMemory').textContent=`${resources.memory_free_percent??'--'}%`;$('#loungeLoad').textContent=`${Math.round(Number(resources.load_ratio||0)*100)}%`;$('#loungeIdle').textContent=formatDuration(Math.min(Number(resources.system_idle_seconds||0),Number(resources.app_idle_seconds||0)));
  const today=new Date().toDateString();const todayCount=(payload.history||[]).filter(item=>item.trigger_type!=='screen'&&new Date(item.started_at).toDateString()===today&&['completed','interrupted'].includes(item.status)).length;$('#loungeToday').textContent=`${todayCount} / ${config.max_daily_rounds||4}`;
  $('#loungeNextRun').textContent=config.next_run_after?`下轮最早 ${new Date(config.next_run_after).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'})}`:payload.eligibility_reason;
  $('#runLoungeButton').disabled=running||screenRunning;$('#runLoungeButton').textContent=running?'正在交流…':'现在聊一轮';
  $('#screenWatchStatus').textContent=config.screen_last_status||'等待下一次屏幕观察';
  $('#watchScreenButton').disabled=running||screenRunning||Boolean(payload.screen_pending);$('#watchScreenButton').textContent=screenRunning?'正在观察…':(payload.screen_pending?'正在等待截图…':'现在看一眼');
  renderScreenDiagnostic(config);
  $('#checkScreenCaptureButton').disabled=state.screenDiagnosticRunning||Boolean(payload.screen_pending);
  $('#checkScreenCaptureButton').textContent=state.screenDiagnosticRunning?'正在检测…':'检测权限与截图';
  renderMemoryPoolCard('aili',payload.memory_pools?.aili||{});renderMemoryPoolCard('shaya',payload.memory_pools?.shaya||{});
  const editing=preserveForm&&['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName)&&document.activeElement?.closest('[data-page="lounge"]');
  if(!editing){$('#loungeEnabled').checked=Boolean(config.enabled);$('#loungeIdleSelect').value=String(config.idle_minutes||15);$('#loungeModelSelect').value=config.model_strategy||'auto';$('#loungeMinInterval').value=String(config.min_interval_minutes||180);$('#loungeMaxInterval').value=String(config.max_interval_minutes||360);$('#loungeDailyMax').value=String(config.max_daily_rounds||4);$('#loungeInspectFiles').checked=Boolean(config.inspect_files);$('#loungeRoots').value=(config.scan_roots||payload.default_roots||[]).join('\n');$('#screenWatchEnabled').checked=Boolean(config.screen_watch_enabled);$('#screenMinInterval').value=String(config.screen_min_interval_minutes||60);$('#screenMaxInterval').value=String(config.screen_max_interval_minutes||180);$('#screenDailyMax').value=String(config.screen_max_daily||6);}
  renderLoungeLog(payload.history||[]);
  renderScreenWatchLog(payload.screen_history||[]);
}
async function refreshLounge(preserveForm=false){const payload=await api('/api/gui/lounge');renderLounge(payload,preserveForm);return payload;}
function renderScreenDiagnostic(config={}){
  const panel=$('#screenDiagnostic');if(!panel)return;
  const status=String(config.screen_diagnostic_status||'idle');
  const titles={idle:'尚未检测',waiting:'等待原生客户端',capturing:'正在读取权限并截图',success:'检测通过',failed:'检测未通过',timeout:'检测超时'};
  const icons={idle:'?',waiting:'…',capturing:'◎',success:'✓',failed:'!',timeout:'!'};
  const fallback='点击检测后，只读取 macOS 权限并截取一帧；不调用模型、不保存截图。';
  panel.className=`screen-diagnostic ${status in titles?status:'idle'}`;
  $('#screenDiagnosticIcon').textContent=icons[status]||'?';
  $('#screenDiagnosticTitle').textContent=titles[status]||titles.idle;
  const rawDetail=config.screen_diagnostic_detail||fallback;
  $('#screenDiagnosticDetail').textContent=status==='failed'&&/\u6743\u9650|\u672a\u6388\u6743/.test(rawDetail)
    ? `${rawDetail} 如果系统列表里已经开启，请先关闭再开启“星语茶话屋”，然后完全退出并重新打开应用。`
    : rawDetail;
}
async function saveLounge(){
  const minimum=Number($('#loungeMinInterval').value),maximum=Number($('#loungeMaxInterval').value);if(maximum<minimum)throw new Error('最长间隔不能小于最短间隔');
  const screenMinimum=Number($('#screenMinInterval').value),screenMaximum=Number($('#screenMaxInterval').value);if(screenMaximum<screenMinimum)throw new Error('屏幕观察最长间隔不能小于最短间隔');
  await api('/api/gui/lounge/config',{method:'POST',body:JSON.stringify({enabled:$('#loungeEnabled').checked,idle_minutes:Number($('#loungeIdleSelect').value),min_interval_minutes:minimum,max_interval_minutes:maximum,max_daily_rounds:Number($('#loungeDailyMax').value),model_strategy:$('#loungeModelSelect').value,inspect_files:$('#loungeInspectFiles').checked,scan_roots:$('#loungeRoots').value,screen_watch_enabled:$('#screenWatchEnabled').checked,screen_min_interval_minutes:screenMinimum,screen_max_interval_minutes:screenMaximum,screen_max_daily:Number($('#screenDailyMax').value)})});toast('人格记忆与观察策略已保存','success');await refreshLounge();
}
async function runLounge(){const result=await api('/api/gui/lounge/run',{method:'POST',body:'{}'});toast(`已启动 ${String(result.tier).toUpperCase()} 后台交流`,'success');setTimeout(()=>refreshLounge().catch(()=>{}),500);}
async function watchScreenNow(){await api('/api/gui/screen-watch/request',{method:'POST',body:'{}'});toast('已请求原生客户端观察；截图只在内存中停留','success');setTimeout(()=>refreshLounge().catch(()=>{}),800);}
async function checkScreenCapture(){
  if(state.screenDiagnosticRunning)return;
  state.screenDiagnosticRunning=true;
  const button=$('#checkScreenCaptureButton');button.disabled=true;button.textContent='正在检测…';
  try{
    await api('/api/gui/screen-watch/diagnose',{method:'POST',body:'{}'});
    await refreshLounge();
    const deadline=Date.now()+45000;
    while(Date.now()<deadline){
      await new Promise(resolve=>setTimeout(resolve,1000));
      const payload=await refreshLounge();
      const status=String(payload.config?.screen_diagnostic_status||'');
      if(status==='success'){toast('屏幕权限与截图链路正常','success');return;}
      if(status==='failed'){toast('检测未通过，已显示 macOS 返回的原因','error');return;}
    }
    const panel=$('#screenDiagnostic');panel.className='screen-diagnostic timeout';
    $('#screenDiagnosticIcon').textContent='!';$('#screenDiagnosticTitle').textContent='检测超时';
    $('#screenDiagnosticDetail').textContent='原生客户端 45 秒内没有交回结果。请确认菜单栏的星语茶话屋仍在运行，然后重试。';
    toast('屏幕检测超时','error');
  }finally{
    state.screenDiagnosticRunning=false;button.disabled=false;button.textContent='检测权限与截图';
  }
}
async function clearScreenWatch(){if(!await askConfirm('清空屏幕观察文字？','只删除艾莉和沙雅由屏幕观察形成的文字经历与向量。系统从未保存截图，普通对话和茶话记忆不受影响。','清空'))return;await api('/api/gui/screen-watch/clear',{method:'POST',body:JSON.stringify({confirm:'DELETE'})});toast('屏幕观察文字与向量已清空','success');await refreshLounge();}
async function clearLounge(){if(!await askConfirm('清空茶话室日志？','将删除两个人格的后台交流、文件观察和共享学习记忆。正常对话与两套用户档案不受影响。','清空'))return;await api('/api/gui/lounge/clear',{method:'POST',body:JSON.stringify({confirm:'DELETE'})});toast('茶话室日志已清空','success');await refreshLounge();}
function pingUserActivity(){const now=Date.now();if(now-state.lastActivityPing<30000)return;state.lastActivityPing=now;fetch('/api/gui/lounge/activity',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).catch(()=>{});}

function askConfirm(title,text,confirmText='确认'){ return new Promise(resolve=>{ $('#modalTitle').textContent=title;$('#modalText').textContent=text;$('#modalConfirm').textContent=confirmText;$('#modalBackdrop').classList.remove('hidden');const finish=value=>{$('#modalBackdrop').classList.add('hidden');$('#modalConfirm').onclick=null;$('#modalCancel').onclick=null;resolve(value);};$('#modalConfirm').onclick=()=>finish(true);$('#modalCancel').onclick=()=>finish(false); }); }

async function deleteCurrent(){ if(!state.current)return; if(!await askConfirm('删除这个会话？','原始消息和所属图片会被永久删除，并立即回收数据库空间。','删除'))return; await api(`/api/gui/conversations/${state.current.id}`,{method:'DELETE',headers:{'X-Confirm-Delete':'DELETE'}});state.current=null;await refreshConversations();if(state.conversations.length)await loadConversation(state.conversations[0].id);else showWelcome();toast('已删除会话','success'); }
async function renameCurrent(){ if(!state.current)return;openSettings('memory');setTimeout(()=>$('#conversationTitleInput').focus(),250); }
async function saveMemory(){
  const persona=selectedPersona(); if(!persona)return;
  const updatedPersona=await api(`/api/gui/personas/${persona.id}`,{method:'PATCH',body:JSON.stringify({profile:$('#personaProfileInput').value,memory:$('#personaMemoryInput').value})});
  state.personas=state.personas.map(item=>item.id===persona.id?{...item,...updatedPersona}:item);
  if(state.current){const updated=await api(`/api/gui/conversations/${state.current.id}`,{method:'PATCH',body:JSON.stringify({title:$('#conversationTitleInput').value,system_prompt:$('#systemPromptInput').value,summary:$('#summaryInput').value})});state.current={...state.current,...updated};await refreshConversations();}
  fillMemoryEditor();toast(`${persona.name}的个人简介与长期记忆已保存`,'success');
}
async function rebuildMemory(){
  const persona=selectedPersona();if(!persona)return;
  if(!await askConfirm(`重建${persona.name}的长期记忆？`,'将清空当前整理结果，再只根据该人格所有现存对话原文重新构造。另一人格不受影响。','开始重建'))return;
  await api(`/api/gui/personas/${persona.id}/rebuild`,{method:'POST',body:'{}'});toast(`${persona.name}已在后台重建记忆`,'success');await refreshSettings();
}
async function cleanup(all=false){ const text=all?'将删除全部对话原文和图片附件，但保留艾莉与沙雅的共享长期记忆。模型不会删除。':'将删除 30 天未使用的对话和所属图片，保留两套人格记忆。';if(!await askConfirm(all?'清空全部对话？':'清理过期历史？',text,'立即清理'))return;const body=all?{all:true,confirm:'DELETE'}:{older_than_days:30,confirm:'DELETE'};const result=await api('/api/gui/cleanup',{method:'POST',body:JSON.stringify(body)});toast(`已清理 ${result.deleted} 个会话`,'success');state.current=null;await bootstrap();openSettings('storage'); }

function copyText(text){ navigator.clipboard.writeText(text).then(()=>toast('已复制到剪贴板','success')).catch(()=>toast('复制失败','error')); }

function nativeBridge(action, values={}){
  const handler=window.webkit?.messageHandlers?.nativeBridge;
  if(!handler){
    $('#nativeSettingsStatus').textContent='当前在浏览器中预览；自启动设置仅在 macOS 客户端内可用。';
    return false;
  }
  handler.postMessage({action,...values});return true;
}
function applyNativePreferences(detail={}){
  $('#launchAtLoginToggle').checked=Boolean(detail.launchAtLogin);
  $('#openMainAtLaunchToggle').checked=Boolean(detail.openMainAtLaunch);
  $('#nativeSettingsStatus').textContent=detail.message||`菜单栏常驻 · 登录自启${detail.launchAtLogin?'已开启':'未开启'}`;
}
function syncPersonaFromStorage(){
  const persona=localStorage.getItem('localai.persona');
  if(!persona||persona===state.settings.persona||state.generating||!state.personas.some(item=>item.id===persona))return;
  switchPersona(persona).catch(error=>toast(error.message,'error'));
}
window.addEventListener('local-ai-native-preferences',event=>applyNativePreferences(event.detail||{}));
window.addEventListener('local-ai-native-result',event=>{
  applyNativePreferences(event.detail||{});
  if(event.detail?.error)toast(event.detail.error,'error');
  else if(event.detail?.message)toast(event.detail.message,'success');
});
window.addEventListener('storage',event=>{if(event.key==='localai.persona')syncPersonaFromStorage();});
window.addEventListener('focus',syncPersonaFromStorage);
document.addEventListener('visibilitychange',()=>{document.body.classList.toggle('page-hidden',document.hidden);if(!document.hidden)syncPersonaFromStorage();});

$('#newChatButton').onclick=()=>createConversation().catch(e=>toast(e.message,'error'));
$('#sessionSearch').oninput=searchConversations;
$('#loadMoreSessions').onclick=()=>refreshConversations({append:true,query:state.conversationQuery}).catch(error=>toast(`加载失败：${error.message}`,'error'));
$('#tierSelect').onchange=event=>switchTier(event.target.value).catch(error=>toast(error.message,'error'));
$('#messageInput').oninput=autoResize;
$('#messageInput').addEventListener('compositionstart',()=>{state.composing=true;});
$('#messageInput').addEventListener('compositionend',()=>{state.composing=false;autoResize();});
$('#messageInput').onkeydown=event=>{if(event.isComposing||state.composing||event.keyCode===229)return;if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage();}};
$('#sendButton').onclick=sendMessage; $('#attachButton').onclick=()=>$('#imageInput').click(); $('#imageInput').onchange=event=>addFiles(event.target.files);
$('#starterImage').onclick=()=>$('#imageInput').click(); $$('.starter-grid [data-prompt]').forEach(button=>button.onclick=()=>{$('#messageInput').value=button.dataset.prompt;autoResize();$('#messageInput').focus();});
$('#loadOlderButton').onclick=loadOlder; $('#renameButton').onclick=renameCurrent; $('#deleteButton').onclick=deleteCurrent;
$('#exportButton').onclick=()=>{if(state.current)window.location.href=`/api/gui/conversations/${state.current.id}/export`;};
$('#settingsButton').onclick=()=>openSettings('general'); $('#apiButton').onclick=()=>openSettings('api'); $('#closeSettings').onclick=closeSettings; $('#drawerBackdrop').onclick=closeSettings;
$$('.settings-tabs button').forEach(button=>button.onclick=()=>switchTab(button.dataset.tab));
$$('[data-lounge-view]').forEach(button=>button.onclick=()=>setLoungeView(button.dataset.loungeView));
$$('#qualitySwitch button').forEach(button=>button.onclick=()=>{if(state.generating||state.settings.tier==='ultimate')return;state.settings.qualityMode=button.dataset.quality;localStorage.setItem('localai.qualityMode',state.settings.qualityMode);updateQualityControls();toast(`已切换到${qualityLabel(state.settings.qualityMode)}模式`,'success');});
$('#temperatureInput').value=state.settings.temperature;$('#temperatureValue').textContent=state.settings.temperature.toFixed(1);$('#temperatureInput').oninput=event=>{state.settings.temperature=Number(event.target.value);$('#temperatureValue').textContent=state.settings.temperature.toFixed(1);localStorage.setItem('localai.temperature',state.settings.temperature);};
$('#topPInput').value=state.settings.topP;$('#topPValue').textContent=state.settings.topP.toFixed(2);$('#topPInput').oninput=event=>{state.settings.topP=Number(event.target.value);$('#topPValue').textContent=state.settings.topP.toFixed(2);localStorage.setItem('localai.topP',state.settings.topP);};
$('#repeatPenaltyInput').value=state.settings.repeatPenalty;$('#repeatPenaltyValue').textContent=state.settings.repeatPenalty.toFixed(2);$('#repeatPenaltyInput').oninput=event=>{state.settings.repeatPenalty=Number(event.target.value);$('#repeatPenaltyValue').textContent=state.settings.repeatPenalty.toFixed(2);localStorage.setItem('localai.repeatPenalty',state.settings.repeatPenalty);};
$('#seedInput').value=String(state.settings.seed);$('#seedInput').onchange=event=>{state.settings.seed=Math.max(0,Number(event.target.value)||0);localStorage.setItem('localai.seed',state.settings.seed);};
$('#keepAliveSelect').value='1m';localStorage.setItem('localai.keepAlive','1m');
$('#maxTokensSelect').value=String(state.settings.maxTokens);$('#maxTokensSelect').onchange=event=>{state.settings.maxTokens=Number(event.target.value);localStorage.setItem('localai.maxTokens',state.settings.maxTokens);};
$('#launchAtLoginToggle').onchange=event=>nativeBridge('setLaunchAtLogin',{enabled:event.target.checked});
$('#openMainAtLaunchToggle').onchange=event=>nativeBridge('setOpenMainAtLaunch',{enabled:event.target.checked});
$('#openScreenPrivacyButton').onclick=()=>nativeBridge('openScreenPrivacy');
$('#saveMemoryButton').onclick=()=>saveMemory().catch(e=>toast(e.message,'error'));$('#rebuildMemoryButton').onclick=()=>rebuildMemory().catch(e=>toast(e.message,'error'));
$('#saveLoungeButton').onclick=()=>saveLounge().catch(e=>toast(e.message,'error'));$('#runLoungeButton').onclick=()=>runLounge().catch(e=>toast(e.message,'error'));$('#watchScreenButton').onclick=()=>watchScreenNow().catch(e=>toast(e.message,'error'));$('#checkScreenCaptureButton').onclick=()=>checkScreenCapture().catch(e=>toast(e.message,'error'));$('#clearScreenButton').onclick=()=>clearScreenWatch().catch(e=>toast(e.message,'error'));$('#refreshLoungeButton').onclick=()=>refreshLounge().catch(e=>toast(e.message,'error'));$('#clearLoungeButton').onclick=()=>clearLounge().catch(e=>toast(e.message,'error'));
$('#cleanup30Button').onclick=()=>cleanup(false).catch(e=>toast(e.message,'error'));$('#cleanupAllButton').onclick=()=>cleanup(true).catch(e=>toast(e.message,'error'));$('#revealDataButton').onclick=()=>api('/api/gui/reveal-data',{method:'POST',body:'{}'});
$$('[data-copy]').forEach(button=>button.onclick=()=>copyText(button.dataset.copy==='native'?$('#nativeApiValue').textContent:$('#memoryApiValue').textContent));$('#copyCodeButton').onclick=()=>copyText($('#apiCodeSample').textContent);
$('#closeLightbox').onclick=()=>$('#imageLightbox').classList.add('hidden');$('#imageLightbox').onclick=event=>{if(event.target===$('#imageLightbox'))$('#imageLightbox').classList.add('hidden');};
document.addEventListener('paste',event=>{const files=[...event.clipboardData.items].filter(item=>item.kind==='file').map(item=>item.getAsFile()).filter(Boolean);if(files.length)addFiles(files);});
['pointerdown','keydown','wheel','touchstart'].forEach(name=>document.addEventListener(name,pingUserActivity,{passive:true}));
let dragDepth=0;document.addEventListener('dragenter',event=>{event.preventDefault();dragDepth++;$('#dropOverlay').classList.remove('hidden');});document.addEventListener('dragleave',event=>{event.preventDefault();dragDepth--;if(dragDepth<=0){dragDepth=0;$('#dropOverlay').classList.add('hidden');}});document.addEventListener('dragover',event=>event.preventDefault());document.addEventListener('drop',event=>{event.preventDefault();dragDepth=0;$('#dropOverlay').classList.add('hidden');addFiles(event.dataTransfer.files);});
document.addEventListener('keydown',event=>{if(event.metaKey&&event.key.toLowerCase()==='n'){event.preventDefault();createConversation();}if(event.key==='Escape'){closeSettings();$('#imageLightbox').classList.add('hidden');}});

nativeBridge('getPreferences');
setLoungeView(state.loungeView,false);
bootstrap();
