import { initRag, retrieve } from './rag.js'
import { buildMessages, chatStream, judgeAnswer } from './ollama.js'

const $ = (id) => document.getElementById(id)
const statusEl = $('status')
const chatEl = $('chat')
const inputEl = $('q')
const sendEl = $('send')
const stopEl = $('stop')
const relatedEl = $('related')

let ready = false
let controller = null

const VERDICT_LABEL = {
  grounded: '근거 충분',
  partial: '근거 일부',
  ungrounded: '근거 없음(주의)',
  refusal: '거절(근거 없음)',
  unknown: '판정 보류',
}

const escapeHtml = (s) =>
  String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))

function addMsg(role, text) {
  const d = document.createElement('div')
  d.className = 'msg ' + role
  d.textContent = text
  chatEl.appendChild(d)
  chatEl.scrollTop = chatEl.scrollHeight
  return d
}

// 답변 아래 메타 행: [✍️ 생성] [🔎 판정] [👍][👎]  — 생성 자연스러움 ≠ 근거 있음을 구분
function addMeta(question) {
  const wrap = document.createElement('div')
  wrap.className = 'meta'
  const gen = document.createElement('span')
  gen.className = 'chip gen'
  gen.textContent = '✍️ 생성 답변'
  gen.title = '문장이 매끄러워도 근거가 있다는 뜻은 아닙니다'
  const judge = document.createElement('span')
  judge.className = 'chip judging'
  judge.textContent = '🔎 판정 중…'
  const fb = document.createElement('span')
  fb.className = 'fb'
  const up = document.createElement('button')
  up.textContent = '👍'
  const down = document.createElement('button')
  down.textContent = '👎'
  fb.append(up, down)
  wrap.append(gen, judge, fb)
  chatEl.appendChild(wrap)

  const key = 'fb:' + question
  const saved = loadFb(key)
  if (saved === 'up') up.classList.add('on')
  if (saved === 'down') down.classList.add('on')
  up.onclick = () => { setFb(key, 'up'); up.classList.add('on'); down.classList.remove('on') }
  down.onclick = () => { setFb(key, 'down'); down.classList.add('on'); up.classList.remove('on') }
  return { judge }
}

function setJudge(el, verdict, reason) {
  el.className = 'chip ' + (verdict || 'unknown')
  el.textContent = '🔎 근거 판정: ' + (VERDICT_LABEL[verdict] || verdict)
  if (reason) el.title = reason
}

const loadFb = (k) => { try { return localStorage.getItem(k) } catch { return null } }
const setFb = (k, v) => { try { localStorage.setItem(k, v) } catch {} }

async function boot() {
  try {
    const { count } = await initRag((s) => (statusEl.textContent = s))
    ready = true
    sendEl.disabled = false
    statusEl.textContent = `준비 완료 · 근거 청크 ${count}개 · (답변 생성엔 로컬 Ollama 필요)`
  } catch (e) {
    statusEl.textContent = '초기화 실패: ' + e.message
  }
}

function setGenerating(on) {
  stopEl.style.display = on ? 'inline-block' : 'none'
  sendEl.disabled = on
}

async function ask() {
  const q = inputEl.value.trim()
  if (!q || !ready) return
  inputEl.value = ''
  addMsg('user', q)

  const hits = await retrieve(q)
  relatedEl.innerHTML = hits.length
    ? hits
        .map(
          (h, i) =>
            `<div class="src"><b>${i + 1}. [출처 ${escapeHtml(h.chunk.source)} · p${h.chunk.page}]</b> · 유사도 ${h.score.toFixed(3)}<br>${escapeHtml(h.chunk.text)}</div>`
        )
        .join('')
    : '<i>연관 근거 없음</i>'

  // 폐쇄형: 근거 없으면 생성 안 하고 거절 + 판정도 refusal
  if (!hits.length) {
    addMsg('bot', '매뉴얼 근거에서 확인되지 않습니다.')
    const meta = addMeta(q)
    setJudge(meta.judge, 'refusal', '검색 결과 없음(자료 밖)')
    return
  }

  const botDiv = addMsg('bot', '생성 중…')
  const meta = addMeta(q)
  controller = new AbortController()
  setGenerating(true)
  let acc = ''
  try {
    for await (const piece of chatStream(buildMessages(q, hits), controller.signal)) {
      acc += piece
      botDiv.textContent = acc
      chatEl.scrollTop = chatEl.scrollHeight
    }
    if (!acc.trim()) botDiv.textContent = '매뉴얼 근거에서 확인되지 않습니다.'
  } catch (e) {
    setGenerating(false)
    if (e.name === 'AbortError') {
      botDiv.textContent = (acc || '') + '  …(중지됨)'
      meta.judge.className = 'chip unknown'
      meta.judge.textContent = '🔎 판정 생략(중지)'
      return
    }
    botDiv.textContent = '⚠️ ' + e.message
    botDiv.classList.add('err')
    meta.judge.className = 'chip unknown'
    meta.judge.textContent = '🔎 판정 불가'
    return
  }
  setGenerating(false)

  // LLM-as-a-Judge: 근거성 자동 판정
  const v = await judgeAnswer(q, hits, botDiv.textContent)
  setJudge(meta.judge, v.verdict, v.reason)
}

sendEl.addEventListener('click', ask)
stopEl.addEventListener('click', () => { if (controller) controller.abort() })
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    ask()
  }
})
boot()
