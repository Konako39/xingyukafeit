const $ = selector => document.querySelector(selector);

const state = {
  personas: [], models: [], conversations: [],
  persona: localStorage.getItem('localai.persona') || 'aili',
  tier: localStorage.getItem('localai.tier') || '9b',
  quality: localStorage.getItem('localai.qualityMode') || 'balanced',
  current: null, messages: [], busy: false, controller: null, activeRun: null,
  composing: false,
  viewVersion: 0, loadVersion: 0,
  views: {
    aili: {current: null, messages: []},
    shaya: {current: null, messages: []},
  },
};

const avatar = id => `/app/assets/${id === 'shaya' ? 'shaya' : 'aili'}-avatar.png`;
const portrait = id => `/app/assets/${id === 'shaya' ? 'shaya' : 'aili'}-full.png`;
const tierLabel = tier => ({'4b':'极速·4B','9b':'日用·9B','27b':'高级·27B','ultimate':'究极'})[String(tier).toLowerCase()] || String(tier);
const esc = value => String(value ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const personaCopy = id => id === 'shaya'
  ? '我会认真帮你理好的。所、所以别敷衍就好。'
  : '有啥就说啥呗，聊到哪算哪～别那么拘谨💗';

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {'Content-Type':'application/json', ...(options.headers || {})},
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message = (await response.json()).error?.message || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function selectedPersona(id = state.persona) { return state.personas.find(item => item.id === id); }
function model(persona = state.persona, tier = state.tier) { return selectedPersona(persona)?.models?.[tier] || ''; }
function toast(message) { const node=$('#toast');node.textContent=message;node.classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>node.classList.remove('show'),2600); }
function setStatus(message, thinking = false) { $('#quickStatus').textContent=message;$('#quickStatus').classList.toggle('thinking',thinking); }

function ensureView(persona) {
  if (!state.views[persona]) state.views[persona] = {current:null, messages:[]};
  return state.views[persona];
}

function saveActiveView() {
  const view = ensureView(state.persona);
  view.current = state.current;
  view.messages = state.messages;
}

function restoreView(persona) {
  const view = ensureView(persona);
  state.current = view.current;
  state.messages = view.messages;
}

function renderPersona() {
  document.body.classList.toggle('aili',state.persona==='aili');
  document.body.classList.toggle('shaya',state.persona==='shaya');
  document.body.classList.toggle('ultimate',state.tier==='ultimate');
  $('#personaPicker').innerHTML = state.personas.map(item => {
    // 有成长心境时显示实时心境，让人格看起来是"活的"；没有则退回固定副标题。
    const mood = item.growth?.mood || '';
    return `
    <button class="persona-card ${item.id === state.persona ? 'active' : ''}" data-persona="${item.id}" aria-pressed="${item.id === state.persona}" title="${esc(mood || item.subtitle)}">
      <img src="${avatar(item.id)}" alt="${esc(item.name)}">
      <span><b>${esc(item.name)}</b><small>${esc(mood || item.subtitle)}</small></span>
      <i aria-hidden="true">✦</i>
    </button>`;
  }).join('');
  document.querySelectorAll('[data-persona]').forEach(button => {
    button.onclick = () => switchPersona(button.dataset.persona);
  });
  const persona = selectedPersona();
  const emptyPortrait = $('#emptyPortrait');
  const emptyTitle = $('#emptyTitle');
  if (emptyPortrait) {
    emptyPortrait.src = portrait(state.persona);
    emptyPortrait.alt = persona?.name || '';
  }
  if (emptyTitle) emptyTitle.textContent = `${persona?.name || ''}在这儿`;
  $('#tierSelect').value = state.tier;
  const ultimateOption=$('#tierSelect').querySelector('option[value="ultimate"]');
  if(ultimateOption){
    const ready=Boolean(persona?.models?.ultimate);
    ultimateOption.disabled=!ready;
    ultimateOption.textContent=ready?'究极':'究极 · 等待服务重载';
  }
  renderQuality();
  renderHistory();
}

function renderQuality() {
  $('#qualityPicker').classList.toggle('hidden',state.tier==='ultimate');
  document.querySelectorAll('[data-quality]').forEach(button => button.classList.toggle('active',button.dataset.quality===state.quality));
}

function renderHistory() {
  const mine = state.conversations.filter(item => item.persona === state.persona);
  $('#historySelect').innerHTML = '<option value="">新对话</option>' + mine.map(item => `<option value="${item.id}">${esc(item.title)} · ${tierLabel(item.tier)}</option>`).join('');
  $('#historySelect').value = state.current?.persona === state.persona ? String(state.current.id) : '';
}

function renderMessages() {
  if (!state.messages.length) {
    const persona = selectedPersona();
    $('#messages').innerHTML = `<div class="empty-state"><img id="emptyPortrait" src="${portrait(state.persona)}" alt="${esc(persona?.name || '')}"><div><span class="empty-kicker">✦ ${state.persona === 'shaya' ? '班长值日中' : '元气上线'} ✦</span><b id="emptyTitle">${esc(persona?.name || '')}在这儿</b><p>${esc(personaCopy(state.persona))}</p></div></div>`;
    return;
  }
  const recent = state.messages.slice(-30);
  $('#messages').innerHTML = recent.map(item => {
    if (item.role === 'user') return `<div class="bubble-row user"><div class="bubble">${esc(item.content)}</div></div>`;
    const messagePersona = item.persona || state.current?.persona || state.persona;
    return `<div class="bubble-row"><img src="${avatar(messagePersona)}" alt="${esc(selectedPersona(messagePersona)?.name || '')}"><div class="bubble">${esc(item.content)}</div></div>`;
  }).join('');
  $('#messages').scrollTop = $('#messages').scrollHeight;
}

// 流式输出只改最后一个气泡的文本，不整页重建：
// 不打断用户选中文字，也允许往回翻看时不被强行拉到底部。
function updateStreamingBubble(placeholder) {
  const container = $('#messages');
  const bubbles = container.querySelectorAll('.bubble-row:not(.user) .bubble');
  const last = bubbles[bubbles.length - 1];
  if (!last) { renderMessages(); return; }
  const stick = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
  last.textContent = placeholder.content;
  if (stick) container.scrollTop = container.scrollHeight;
}

function cancelActiveRun() {
  if (!state.activeRun) return false;
  state.activeRun.controller.abort();
  state.activeRun = null;
  state.controller = null;
  state.busy = false;
  $('#sendButton span').textContent = '↑';
  return true;
}

function switchPersona(persona, {persist = true, source = 'picker'} = {}) {
  if (!selectedPersona(persona) || persona === state.persona) return;
  const stopped = cancelActiveRun();
  saveActiveView();
  state.viewVersion += 1;
  state.loadVersion += 1;
  state.persona = persona;
  if (persist) localStorage.setItem('localai.persona', persona);
  restoreView(persona);
  renderPersona();
  renderMessages();
  const sourceText = source === 'external' ? '已与主面板同步' : '已切换人格';
  setStatus(`${sourceText}：${selectedPersona()?.name || ''}${stopped ? '（已停止上一个回答）' : ''}`);
}

async function loadConversation(id) {
  if (!id) {
    state.loadVersion += 1;
    state.current = null;
    state.messages = [];
    saveActiveView();
    renderHistory();
    renderMessages();
    setStatus(`已为${selectedPersona()?.name || ''}开启新对话`);
    return;
  }
  cancelActiveRun();
  const requestVersion = ++state.loadVersion;
  const data = await api(`/api/gui/conversations/${id}?limit=80`);
  if (requestVersion !== state.loadVersion) return;
  const conversationPersona = data.conversation.persona;
  if (conversationPersona !== state.persona) {
    saveActiveView();
    state.persona = conversationPersona;
    localStorage.setItem('localai.persona', state.persona);
  }
  state.current = data.conversation;
  state.tier = state.current.tier;
  localStorage.setItem('localai.tier', state.tier);
  state.messages = data.messages.map(message => ({...message, persona:conversationPersona}));
  saveActiveView();
  renderPersona();
  renderMessages();
  setStatus(`已载入「${state.current.title}」`);
}

async function createConversation(persona, tier) {
  const created = await api('/api/gui/conversations', {
    method:'POST',
    body:JSON.stringify({persona, model:model(persona,tier), title:'快速对话'}),
  });
  state.conversations.unshift({...created,message_count:0,preview:''});
  const view = ensureView(persona);
  view.current = created;
  view.messages = [];
  if (state.persona === persona) {
    state.current = created;
    state.messages = view.messages;
    renderHistory();
  }
  return created;
}

async function switchTier(tier) {
  if (state.busy) return toast('回答完成后再切换档位');
  const previous = state.tier;
  if(!model(state.persona,tier)){
    $('#tierSelect').value=previous;
    toast(tier==='ultimate'?'当前还是旧后端，等你方便时重新打开软件即可使用究极':'该档位尚未就绪');
    focusComposer();
    return;
  }
  state.tier = tier;
  localStorage.setItem('localai.tier',tier);
  renderPersona();
  try {
    if (state.current) {
      const updated = await api(`/api/gui/conversations/${state.current.id}`,{method:'PATCH',body:JSON.stringify({model:model()})});
      state.current = {...state.current,...updated};
      state.conversations = state.conversations.map(item=>item.id===updated.id?{...item,...updated}:item);
      saveActiveView();
      renderHistory();
    }
    setStatus(`已切换到 ${tierLabel(tier)}，当前对话继续`);
    focusComposer();
  } catch (error) {
    state.tier = previous;
    localStorage.setItem('localai.tier',previous);
    $('#tierSelect').value = previous;
    renderPersona();
    toast(error.message);
    focusComposer();
  }
}

async function send() {
  if (state.busy) { state.controller?.abort(); return; }
  if(state.composing)return;
  const input = $('#messageInput');
  const text = input.value.trim();
  if (!text) return;

  const runPersona = state.persona;
  const runTier = state.tier;
  const runViewVersion = state.viewVersion;
  let conversation = state.current;
  if (!conversation) conversation = await createConversation(runPersona, runTier);
  if (runViewVersion !== state.viewVersion || runPersona !== state.persona) return;

  const userMessage = {role:'user',content:text,persona:runPersona};
  const placeholder = {role:'assistant',content:'',persona:runPersona};
  state.messages.push(userMessage, placeholder);
  saveActiveView();
  input.value = '';
  resize();
  renderMessages();

  const controller = new AbortController();
  const run = {persona:runPersona, conversationId:conversation.id, controller, placeholder};
  state.activeRun = run;
  state.busy = true;
  state.controller = controller;
  $('#sendButton span').textContent = '■';
  setStatus(runTier==='ultimate'?'正在唤醒究极智慧…':`${selectedPersona(runPersona)?.name || ''}正在回想与思考…`,true);
  let done = false;

  const isActive = () => state.activeRun === run && state.persona === runPersona;
  try {
    const response = await fetch('/api/gui/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      signal:controller.signal,
      body:JSON.stringify({conversation_id:conversation.id,persona:runPersona,model:model(runPersona,runTier),message:text,quality_mode:runTier==='ultimate'?'ultimate':state.quality,temperature:Number(localStorage.getItem('localai.temperature')||.7),top_p:Number(localStorage.getItem('localai.topP')||.95),repeat_penalty:Number(localStorage.getItem('localai.repeatPenalty')||1.1),seed:Number(localStorage.getItem('localai.seed')||0),keep_alive:'1m',max_tokens:Number(localStorage.getItem('localai.maxTokens')||0)||undefined,client_surface:'menubar_quick'}),
    });
    if (!response.ok) throw new Error((await response.json()).error?.message||`HTTP ${response.status}`);
    const reader=response.body.getReader(), decoder=new TextDecoder();
    let buffer='';
    while (true) {
      const next=await reader.read();
      if (next.done) break;
      buffer += decoder.decode(next.value,{stream:true});
      const blocks=buffer.split('\n\n');
      buffer=blocks.pop()||'';
      for (const block of blocks) {
        const line=block.split('\n').find(value=>value.startsWith('data: '));
        if (!line) continue;
        const event=JSON.parse(line.slice(6));
        if (!isActive()) continue;
        if (event.type==='meta') userMessage.id=event.user_message_id;
        if (event.type==='status') setStatus(event.message,true);
        if (event.type==='memory_recall') setStatus(event.count?`已召回 ${event.count} 条相关记忆`:'本轮无需旧记忆',true);
        if (event.type==='delta') { placeholder.content+=event.content;updateStreamingBubble(placeholder); }
        if (event.type==='done') {
          done=true;
          placeholder.id=event.assistant_message_id;
          const cost=event.metrics?.cost_cny!==undefined?` · 本轮 ¥${Number(event.metrics.cost_cny).toFixed(6)}`:'';
          const agent=event.metrics?.agent_performed
            ?' · 已真实执行本地操作 ✓'
            :(event.metrics?.agent_intent?' · 本地工具已响应':'');
          setStatus((event.finish_reason==='length'?'已达本轮长度上限，可发送“继续”':'回答完成')+agent+cost);
        }
        if (event.type==='error') throw new Error(event.message);
      }
    }
    if (!done && isActive()) throw new Error('回答流意外中断');
    const updated=await api('/api/gui/conversations');
    state.conversations=updated.data;
    if (isActive()) renderHistory();
  } catch (error) {
    if (!isActive()) return;
    if (error.name==='AbortError') {
      placeholder.content=placeholder.content||'[生成已停止]';
      setStatus('已停止生成');
    } else {
      placeholder.content=placeholder.content||`[请求失败] ${error.message}`;
      setStatus('本轮失败');
      toast(error.message);
    }
    renderMessages();
  } finally {
    if (state.activeRun === run) {
      state.activeRun=null;
      state.busy=false;
      state.controller=null;
      $('#sendButton span').textContent='↑';
      saveActiveView();
    }
  }
}

function resize() { const input=$('#messageInput');input.style.height='auto';input.style.height=Math.min(input.scrollHeight,92)+'px'; }
function focusComposer(){requestAnimationFrame(()=>$('#messageInput')?.focus({preventScroll:true}));}

// 快捷动作芯片：一键把常用请求发出去（GUI 入口，代替手打指令）。
function sendPreset(text) {
  if (state.busy) { toast('等这轮说完再点'); return; }
  const input = $('#messageInput');
  input.value = text;
  resize();
  send();
}

// /命令面板：输入 / 弹出可点选列表；命令走后端确定性解析，零模型成本。
// 与后端 persona_agent.QUICK_COMMANDS 保持一致。
const QUICK_COMMANDS = [
  {cmd:'日程', usage:'/日程 [今天|明天|本周|关键词]', hint:'查询日历日程'},
  {cmd:'加日程', usage:'/加日程 标题 时间', hint:'添加日程（如：牙医 明天下午3点）'},
  {cmd:'删日程', usage:'/删日程 关键词', hint:'取消日程（唯一匹配才删）'},
  {cmd:'待办', usage:'/待办 [关键词]', hint:'查询未完成提醒'},
  {cmd:'提醒', usage:'/提醒 内容 [时间]', hint:'添加提醒事项'},
  {cmd:'完成', usage:'/完成 关键词', hint:'勾掉一条提醒'},
  {cmd:'删提醒', usage:'/删提醒 关键词', hint:'删除提醒（唯一匹配才删）'},
  {cmd:'找', usage:'/找 文件关键词', hint:'聚焦搜索找文件'},
  {cmd:'打开', usage:'/打开 应用名或网址', hint:'打开应用/网址/文件'},
  {cmd:'记', usage:'/记 内容', hint:'记到系统备忘录'},
  {cmd:'音乐', usage:'/音乐 播放|暂停|下一首', hint:'控制音乐'},
];
let paletteIndex = 0;

function paletteVisible() { return !$('#commandPalette').classList.contains('hidden'); }

function paletteItems() {
  const value = $('#messageInput').value;
  if (!value.startsWith('/') || value.includes('\n')) return [];
  const typed = value.slice(1).split(/\s/)[0];
  // 已经在填参数（出现空格）就收起面板，不挡输入。
  if (value.includes(' ') && QUICK_COMMANDS.some(c => c.cmd === typed)) return [];
  return QUICK_COMMANDS.filter(c => !typed || c.cmd.startsWith(typed));
}

function renderPalette() {
  const box = $('#commandPalette');
  const items = paletteItems();
  if (!items.length) { box.classList.add('hidden'); return; }
  paletteIndex = Math.max(0, Math.min(paletteIndex, items.length - 1));
  box.innerHTML = items.map((c, i) => `
    <button class="palette-item ${i === paletteIndex ? 'active' : ''}" data-cmd="${c.cmd}">
      <b>${esc(c.usage)}</b><small>${esc(c.hint)}</small>
    </button>`).join('');
  box.classList.remove('hidden');
  box.querySelectorAll('[data-cmd]').forEach(button => {
    button.onclick = () => pickCommand(button.dataset.cmd);
  });
}

function pickCommand(cmd) {
  const input = $('#messageInput');
  input.value = '/' + cmd + ' ';
  $('#commandPalette').classList.add('hidden');
  input.focus();
  resize();
}

// 心境是后台学习循环慢慢演化的，面板获得焦点时轻量刷新一次（30 秒节流）。
let moodRefreshedAt = 0;
async function refreshPersonaMood() {
  if (Date.now() - moodRefreshedAt < 30_000) return;
  moodRefreshedAt = Date.now();
  try {
    const data = await api('/v1/personas');
    if (Array.isArray(data.data) && data.data.length) {
      state.personas = data.data;
      renderPersona();
    }
  } catch (_) {}
}

function syncPersonaFromStorage() {
  const persona = localStorage.getItem('localai.persona');
  if (persona && persona !== state.persona && selectedPersona(persona)) switchPersona(persona,{persist:false,source:'external'});
}

async function boot() {
  try {
    const data=await api('/api/gui/bootstrap');
    state.personas=data.personas;
    state.models=data.models;
    state.conversations=data.conversations;
    const ultimate=Boolean(data.ultimate_usage?.available);
    let legacyBackend=false;
    $('#serviceDot').classList.toggle('online',data.ollama_online||ultimate);
    $('#serviceText').textContent=data.ollama_online?(ultimate?'本地与究极已就绪':'本地服务已就绪'):(ultimate?'究极已就绪':'Ollama 未连接');
    if (!state.personas.some(item=>item.id===state.persona)) state.persona='aili';
    if(!selectedPersona()?.models?.[state.tier]){
      legacyBackend=state.tier==='ultimate';
      state.tier='9b';
      localStorage.setItem('localai.tier',state.tier);
    }
    restoreView(state.persona);
    renderPersona();
    renderQuality();
    renderMessages();
    setStatus(legacyBackend?'已临时回到日用档；当前运行的还是旧后端':'准备就绪');
    focusComposer();
  } catch (error) {
    $('#serviceText').textContent='服务启动中';
    setStatus('等待本地服务…');
    setTimeout(boot,800);
  }
}

$('#historySelect').onchange=event=>loadConversation(Number(event.target.value)||0).catch(error=>toast(error.message));
$('#tierSelect').onchange=event=>switchTier(event.target.value);
document.querySelectorAll('[data-quality]').forEach(button=>button.onclick=()=>{if(state.tier==='ultimate')return;state.quality=button.dataset.quality;localStorage.setItem('localai.qualityMode',state.quality);renderQuality();});
$('#messageInput').oninput=()=>{resize();renderPalette();};
$('#messageInput').addEventListener('compositionstart',()=>{state.composing=true;});
$('#messageInput').addEventListener('compositionend',()=>{state.composing=false;resize();renderPalette();});
$('#messageInput').onkeydown=event=>{
  // Safari/WebKit 在中文输入法确认候选词时也会发 Enter。
  // 不拦这一层会立即发送并清空组字，表现成“输入不了字”。
  if(event.isComposing||state.composing||event.keyCode===229)return;
  if(paletteVisible()){
    const items=paletteItems();
    if(event.key==='ArrowDown'){event.preventDefault();paletteIndex=(paletteIndex+1)%items.length;renderPalette();return;}
    if(event.key==='ArrowUp'){event.preventDefault();paletteIndex=(paletteIndex-1+items.length)%items.length;renderPalette();return;}
    if((event.key==='Enter'&&!event.shiftKey)||event.key==='Tab'){
      event.preventDefault();
      if(items[paletteIndex])pickCommand(items[paletteIndex].cmd);
      return;
    }
    if(event.key==='Escape'){event.preventDefault();$('#commandPalette').classList.add('hidden');return;}
  }
  if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send();}
};
$('#sendButton').onclick=send;
$('#openMainButton').onclick=()=>window.webkit?.messageHandlers?.nativeBridge?.postMessage({action:'openMainWindow'});
document.querySelectorAll('#quickActions [data-preset]').forEach(button=>{
  button.onclick=()=>sendPreset(button.dataset.preset);
});
$('#newChatButton').onclick=()=>loadConversation(0).catch(error=>toast(error.message));
$('#slashButton').onclick=()=>{const input=$('#messageInput');input.value='/';input.focus();resize();renderPalette();};
document.addEventListener('keydown',event=>{
  // Esc 关闭弹窗（命令面板打开时优先只收面板，由输入框处理器负责）。
  if(event.key==='Escape'&&!state.composing&&!paletteVisible()){
    window.webkit?.messageHandlers?.nativeBridge?.postMessage({action:'closeQuickPanel'});
  }
});
window.addEventListener('storage',event=>{if(event.key==='localai.persona')syncPersonaFromStorage();});
window.addEventListener('focus',()=>{syncPersonaFromStorage();focusComposer();refreshPersonaMood();});
document.addEventListener('visibilitychange',()=>{document.body.classList.toggle('page-hidden',document.hidden);if(!document.hidden){syncPersonaFromStorage();focusComposer();}});
boot();
