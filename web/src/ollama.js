// 로컬 Ollama 호출 (사용자 브라우저 → localhost)
export const OLLAMA_DIRECT = 'http://localhost:11434' // Ollama 기본
export const OLLAMA_PROXY = 'http://localhost:11435' // ollama-proxy.mjs (배포 HTTPS→로컬 PNA 우회)
export const LLM_MODEL = 'qwen3.5:2b'
// 배포(HTTPS)면 프록시 우선(직접은 PNA로 막힘), 로컬(HTTP)이면 직접 우선(프록시 불필요)
const HOSTS =
  typeof location !== 'undefined' && location.protocol === 'https:'
    ? [OLLAMA_PROXY, OLLAMA_DIRECT]
    : [OLLAMA_DIRECT, OLLAMA_PROXY]

const SYSTEM_PROMPT =
  "당신은 선박기기 매뉴얼 안내 도우미입니다. 아래 '근거'에 있는 내용만 사용해 한국어로 답하세요.\n" +
  '규칙:\n' +
  "1) 근거에 없는 내용은 지어내지 마세요. 근거로 답할 수 없으면 정확히 '매뉴얼 근거에서 확인되지 않습니다.' 라고만 답하세요.\n" +
  '2) 답에 사용한 근거의 출처를 문장 끝에 [출처 파일명 · p쪽] 형식으로 붙이세요(근거에 제공된 태그를 그대로 사용).\n' +
  '3) 원문에 없는 평가·추측·수치·일반지식을 더하지 마세요.\n' +
  '4) 안전·정비 관련은 최종 판단을 담당 기관사와 원문 페이지 확인에 맡기도록 안내하세요.'

export function buildMessages(query, hits) {
  const ctx = hits
    .map((h, i) => `(${i + 1}) [출처 ${h.chunk.source} · p${h.chunk.page}] ${h.chunk.text}`)
    .join('\n')
  return [
    { role: 'system', content: SYSTEM_PROMPT },
    {
      role: 'user',
      content:
        `[근거]\n${ctx}\n\n[질문]\n${query}\n\n` +
        '[답변] 위 근거만 사용해 한국어로 답하고, 각 문장 끝에 해당 출처 태그를 붙이세요.',
    },
  ]
}

// think:false → qwen3.5 추론 모델이 답을 thinking에만 쏟고 content를 비우는 문제 방지
async function openChat(messages) {
  const body = JSON.stringify({ model: LLM_MODEL, messages, stream: true, think: false })
  let lastErr
  for (const host of HOSTS) {
    try {
      const res = await fetch(host + '/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      })
      if (res.ok) return res
      if (res.status === 404) throw new Error(`MODEL: 모델을 찾을 수 없습니다. \`ollama pull ${LLM_MODEL}\``)
      lastErr = new Error(`Ollama ${res.status}`)
    } catch (e) {
      if (String(e.message || '').startsWith('MODEL')) throw e
      lastErr = e
    }
  }
  throw new Error(
    'CONN: 로컬 Ollama에 연결하지 못했습니다.\n' +
      '· 로컬(HTTP) 실행: Ollama 실행 + `ollama pull qwen3.5:2b`\n' +
      '· 배포(HTTPS) 사이트: 위에 더해 `node web/ollama-proxy.mjs` 실행 필요(브라우저 PNA 우회, README 참고)'
  )
}

export async function* chatStream(messages) {
  const res = await openChat(messages)
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let nl
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim()
      buf = buf.slice(nl + 1)
      if (!line) continue
      let obj
      try {
        obj = JSON.parse(line)
      } catch {
        continue
      }
      const piece = obj?.message?.content || ''
      if (piece) yield piece
      if (obj?.done) return
    }
  }
}
