// 로컬 Ollama 호출 (스트리밍 생성 + 취소 + LLM-as-Judge)
export const OLLAMA_DIRECT = 'http://localhost:11434'
export const OLLAMA_PROXY = 'http://localhost:11435' // ollama-proxy.mjs (배포 HTTPS→로컬 PNA 우회)
export const LLM_MODEL = 'qwen3.5:2b'
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

const JUDGE_PROMPT =
  "당신은 RAG 답변의 '근거성'을 판정하는 심판입니다. 아래 [근거]와 [답변]만 보고 판정하세요.\n" +
  '- grounded: 답변의 핵심 내용이 근거로 확인됨\n' +
  '- partial: 일부만 근거로 확인됨\n' +
  '- ungrounded: 근거에 없는 내용을 말함\n' +
  "- refusal: 답변이 '확인되지 않습니다' 류의 거절\n" +
  '반드시 JSON 한 줄로만: {"verdict":"grounded|partial|ungrounded|refusal","reason":"한 줄 이유"}'

function evidenceBlock(hits) {
  return hits
    .map((h, i) => {
      const c = h.chunk
      const sec = c.section && c.section !== '[검증 필요]' ? ` · ${c.section}` : ''
      return `(${i + 1}) [출처 ${c.source} · p${c.page}${sec}] ${c.text}`
    })
    .join('\n')
}

export function buildMessages(query, hits) {
  return [
    { role: 'system', content: SYSTEM_PROMPT },
    {
      role: 'user',
      content:
        `[근거]\n${evidenceBlock(hits)}\n\n[질문]\n${query}\n\n` +
        '[답변] 위 근거만 사용해 한국어로 답하고, 각 문장 끝에 해당 출처 태그를 붙이세요.',
    },
  ]
}

async function ollamaFetch(payloadObj, signal) {
  const body = JSON.stringify(payloadObj)
  let lastErr
  for (const host of HOSTS) {
    try {
      const res = await fetch(host + '/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal,
      })
      if (res.ok) return res
      if (res.status === 404) throw new Error(`MODEL: 모델을 찾을 수 없습니다. \`ollama pull ${LLM_MODEL}\``)
      lastErr = new Error(`Ollama ${res.status}`)
    } catch (e) {
      if (e.name === 'AbortError') throw e
      if (String(e.message || '').startsWith('MODEL')) throw e
      lastErr = e
    }
  }
  throw new Error(
    'CONN: 로컬 Ollama에 연결하지 못했습니다.\n' +
      '· 로컬(HTTP) 실행: Ollama 실행 + `ollama pull qwen3.5:2b`\n' +
      '· 배포(HTTPS) 사이트: 위에 더해 `node web/ollama-proxy.mjs` 실행(브라우저 PNA 우회, README 참고)'
  )
}

// think:false → 추론 모델이 thinking에만 답을 쏟고 content를 비우는 문제 방지. signal로 취소 가능.
export async function* chatStream(messages, signal) {
  const res = await ollamaFetch({ model: LLM_MODEL, messages, stream: true, think: false }, signal)
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

// LLM-as-a-Judge: 답변이 근거에 뒷받침되는지 자동 판정. 반환 {verdict, reason}.
export async function judgeAnswer(query, hits, answer) {
  const msgs = [
    { role: 'system', content: JUDGE_PROMPT },
    {
      role: 'user',
      content: `[근거]\n${evidenceBlock(hits)}\n\n[질문]\n${query}\n\n[답변]\n${answer}\n\nJSON 한 줄로만 판정:`,
    },
  ]
  let txt = ''
  try {
    const res = await ollamaFetch({ model: LLM_MODEL, messages: msgs, stream: false, think: false })
    const data = await res.json()
    txt = data?.message?.content || ''
  } catch {
    return { verdict: 'unknown', reason: '판정 실패(연결)' }
  }
  const m = txt.match(/\{[\s\S]*\}/)
  if (m) {
    try {
      const o = JSON.parse(m[0])
      if (o.verdict) return { verdict: String(o.verdict).toLowerCase(), reason: o.reason || '' }
    } catch {
      /* fall through */
    }
  }
  const k = (txt.toLowerCase().match(/grounded|partial|ungrounded|refusal/) || ['unknown'])[0]
  return { verdict: k, reason: txt.slice(0, 80) }
}
