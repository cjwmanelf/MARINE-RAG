// 로컬 Ollama 직접 호출 (사용자 브라우저 → localhost:11434)
export const OLLAMA_HOST = 'http://localhost:11434'
export const LLM_MODEL = 'qwen3.5:2b'

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
export async function* chatStream(messages) {
  let res
  try {
    res = await fetch(OLLAMA_HOST + '/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: LLM_MODEL, messages, stream: true, think: false }),
    })
  } catch (e) {
    throw new Error(
      'CONN: 로컬 Ollama에 연결하지 못했습니다. Ollama 실행 + `ollama pull qwen3.5:2b` + ' +
        '이 페이지 출처를 OLLAMA_ORIGINS에 허용했는지 확인하세요.'
    )
  }
  if (!res.ok) {
    const t = await res.text().catch(() => '')
    if (res.status === 404) throw new Error(`MODEL: 모델을 찾을 수 없습니다. \`ollama pull ${LLM_MODEL}\``)
    throw new Error(`Ollama ${res.status}: ${t.slice(0, 200)}`)
  }
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
