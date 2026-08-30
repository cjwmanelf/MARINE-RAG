import { initRag, retrieve } from './rag.js'
import { buildMessages, chatStream } from './ollama.js'

const $ = (id) => document.getElementById(id)
const statusEl = $('status')
const chatEl = $('chat')
const inputEl = $('q')
const sendEl = $('send')
const relatedEl = $('related')

let ready = false

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))
}

function addMsg(role, text) {
  const div = document.createElement('div')
  div.className = 'msg ' + role
  div.textContent = text
  chatEl.appendChild(div)
  chatEl.scrollTop = chatEl.scrollHeight
  return div
}

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

  // 폐쇄형: 근거가 없으면 생성하지 않는다
  if (!hits.length) {
    addMsg('bot', '매뉴얼 근거에서 확인되지 않습니다.')
    return
  }

  const botDiv = addMsg('bot', '생성 중…')
  let acc = ''
  try {
    for await (const piece of chatStream(buildMessages(q, hits))) {
      acc += piece
      botDiv.textContent = acc
      chatEl.scrollTop = chatEl.scrollHeight
    }
    if (!acc.trim()) botDiv.textContent = '매뉴얼 근거에서 확인되지 않습니다.'
  } catch (e) {
    botDiv.textContent = '⚠️ ' + e.message
    botDiv.classList.add('err')
  }
}

sendEl.addEventListener('click', ask)
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    ask()
  }
})
boot()
