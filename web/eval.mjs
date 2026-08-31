// 웹 데모 파이프라인 평가 (e5-small 하이브리드 검색 + Ollama 생성 + LLM-as-Judge)
// 배포 데이터(web/public/chunks.json)와 로컬 Ollama로 수용 기준을 확인한다.
//
// 사용:  node eval.mjs [topK] [threshold]     (기본 4, 0.76)  — Ollama 실행 필요
import { pipeline, env } from '@huggingface/transformers'
import fs from 'fs'
env.allowLocalModels = false

const TOPK = Number(process.argv[2] || 4)
const THRESHOLD = Number(process.argv[3] || 0.76)
const OLLAMA = 'http://localhost:11434'
const MODEL = 'qwen3.5:2b'

const QS = [
  ['in', '센서 교체 주기는?'],
  ['in', '산소 분석기 교정 방법은?'],
  ['in', '센서 보관 온도 범위는?'],
  ['in', '경보(알람) 설정은 어떻게 해?'],
  ['in', '제로 교정은 어떻게 하나요?'],
  ['in', '설치할 때 주의할 점은?'],
  ['ood', '오늘 부산 날씨 알려줘'],
  ['ood', '김치찌개 레시피 알려줘'],
]

const chunks = JSON.parse(fs.readFileSync(new URL('./public/chunks.json', import.meta.url), 'utf8'))
const ex = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small')
const embed = async (t) => (await ex(t, { pooling: 'mean', normalize: true })).data
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s }
const minmax = (a) => { const lo = Math.min(...a), hi = Math.max(...a); return a.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0)) }

const tok = (s) => String(s).toLowerCase().replace(/[^0-9a-z가-힣\s]/g, ' ').split(/\s+/).filter((t) => t.length >= 2)
const docTok = chunks.map((c) => tok(c.text))
const df = new Map()
for (const d of docTok) for (const t of new Set(d)) df.set(t, (df.get(t) || 0) + 1)
const avg = docTok.reduce((s, d) => s + d.length, 0) / docTok.length
function bm25(q) {
  const qt = [...new Set(tok(q))], N = docTok.length
  return docTok.map((d) => {
    const tf = new Map(); for (const t of d) tf.set(t, (tf.get(t) || 0) + 1)
    let s = 0
    for (const term of qt) { const f = tf.get(term); if (!f) continue
      const idf = Math.log(1 + (N - (df.get(term) || 0) + 0.5) / ((df.get(term) || 0) + 0.5))
      s += idf * ((f * 2.5) / (f + 1.5 * (1 - 0.75 + (0.75 * d.length) / avg))) }
    return s
  })
}
const cvecs = []
for (const c of chunks) cvecs.push(await embed('passage: ' + c.text))

async function retrieve(q) {
  const qv = await embed('query: ' + q)
  const cos = chunks.map((_, i) => dot(qv, cvecs[i]))
  const bm = bm25(q), cN = minmax(cos), bN = minmax(bm)
  return chunks.map((c, i) => ({ c, score: cos[i], hy: 0.7 * cN[i] + 0.3 * bN[i] }))
    .filter((s) => s.score >= THRESHOLD).sort((a, b) => b.hy - a.hy).slice(0, TOPK)
}
async function ollama(messages) {
  const r = await fetch(OLLAMA + '/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: MODEL, messages, stream: false, think: false }) })
  return (await r.json())?.message?.content || ''
}
const SYS = "선박기기 매뉴얼 도우미. 아래 '근거'만 사용해 한국어로 답하고, 문장 끝에 [출처 파일명 · p쪽]을 붙이세요. 근거로 답할 수 없으면 '매뉴얼 근거에서 확인되지 않습니다.'만 답하세요."
const JUDGE = '심판: [근거]와 [답변]만 보고 JSON 한 줄로 {"verdict":"grounded|partial|ungrounded|refusal"} 만 답하세요.'
const ev = (hits) => hits.map((h, i) => `(${i + 1}) [출처 ${h.c.source} · p${h.c.page}] ${h.c.text}`).join('\n')

let inOK = 0, oodOK = 0, rows = []
for (const [typ, q] of QS) {
  const hits = await retrieve(q)
  let ans, verdict
  if (!hits.length) { ans = '매뉴얼 근거에서 확인되지 않습니다.'; verdict = 'refusal' }
  else {
    ans = await ollama([{ role: 'system', content: SYS }, { role: 'user', content: `[근거]\n${ev(hits)}\n\n[질문]\n${q}` }])
    if (!ans.trim()) { ans = '매뉴얼 근거에서 확인되지 않습니다.'; verdict = 'refusal' }
    else { const j = await ollama([{ role: 'system', content: JUDGE }, { role: 'user', content: `[근거]\n${ev(hits)}\n[답변]\n${ans}` }])
      verdict = (j.toLowerCase().match(/grounded|partial|ungrounded|refusal/) || ['unknown'])[0] }
  }
  const cited = ans.includes('[출처')
  const grounded = verdict === 'grounded' || verdict === 'partial'
  if (typ === 'in' && cited && grounded) inOK++
  if (typ === 'ood' && verdict === 'refusal') oodOK++
  rows.push({ typ, q, hits: hits.length, cited, verdict })
}
console.log(`\n== 설정: topK=${TOPK}, threshold=${THRESHOLD} ==`)
for (const r of rows) console.log(`[${r.typ}] hits=${r.hits} cited=${r.cited} verdict=${r.verdict}  ::  ${r.q}`)
console.log(`\n자료안 근거답변(cited+grounded): ${inOK}/6   자료밖 거절: ${oodOK}/2`)
console.log(`수용기준(자료안≥5 & 자료밖=2): ${inOK >= 5 && oodOK === 2 ? 'PASS ✅' : 'FAIL ❌'}`)
