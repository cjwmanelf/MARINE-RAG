// 브라우저 임베딩 + 하이브리드 검색(코사인 + BM25). 완전 클라이언트 사이드.
import { pipeline, env } from '@huggingface/transformers'

env.allowLocalModels = false
const MODEL_ID = 'Xenova/multilingual-e5-small' // 다국어 소형 임베딩(교차언어)

let extractor = null
let chunks = []
let chunkVecs = []

// --- BM25 색인 ---
let docTokens = [] // 청크별 토큰 배열
let df = new Map() // 토큰 → 문서빈도
let avgLen = 0
const K1 = 1.5
const B = 0.75

function tokenize(s) {
  return String(s)
    .toLowerCase()
    .replace(/[^0-9a-z가-힣\s]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length >= 2)
}

function buildBM25() {
  docTokens = chunks.map((c) => tokenize(c.text))
  df = new Map()
  let total = 0
  for (const toks of docTokens) {
    total += toks.length
    for (const t of new Set(toks)) df.set(t, (df.get(t) || 0) + 1)
  }
  avgLen = total / Math.max(1, docTokens.length)
}

function bm25Scores(query) {
  const q = [...new Set(tokenize(query))]
  const N = docTokens.length
  return docTokens.map((toks) => {
    const len = toks.length
    const tf = new Map()
    for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1)
    let s = 0
    for (const term of q) {
      const f = tf.get(term)
      if (!f) continue
      const idf = Math.log(1 + (N - (df.get(term) || 0) + 0.5) / ((df.get(term) || 0) + 0.5))
      s += idf * ((f * (K1 + 1)) / (f + K1 * (1 - B + (B * len) / avgLen)))
    }
    return s
  })
}

export async function initRag(onStatus) {
  onStatus?.('임베딩 모델 로딩 중… (첫 방문은 수백 MB 다운로드)')
  extractor = await pipeline('feature-extraction', MODEL_ID, {
    progress_callback: (p) => {
      if (p && p.status === 'progress' && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100)
        onStatus?.(`모델 다운로드 ${pct}%  ${p.file ? '· ' + p.file : ''}`)
      }
    },
  })

  onStatus?.('근거 데이터(청크) 불러오는 중…')
  const res = await fetch(import.meta.env.BASE_URL + 'chunks.json')
  chunks = await res.json()

  chunkVecs = []
  for (let i = 0; i < chunks.length; i++) {
    chunkVecs.push(await embed('passage: ' + chunks[i].text))
    if (i % 5 === 0 || i === chunks.length - 1) {
      onStatus?.(`청크 임베딩 ${i + 1}/${chunks.length}`)
    }
  }
  buildBM25()

  onStatus?.(`준비 완료 · 근거 청크 ${chunks.length}개`)
  return { count: chunks.length }
}

async function embed(text) {
  const out = await extractor(text, { pooling: 'mean', normalize: true })
  return out.data
}

function dot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

const minmax = (arr) => {
  const lo = Math.min(...arr), hi = Math.max(...arr)
  return arr.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0))
}

// 하이브리드: 코사인(의미) + BM25(키워드). 근거 게이트는 '코사인 임계값'으로(자료밖 거부 유지).
export async function retrieve(query, { topK = 4, threshold = 0.76, wSem = 0.7 } = {}) {
  if (!extractor) throw new Error('RAG 미초기화')
  const q = await embed('query: ' + query)
  const cos = chunks.map((_, i) => dot(q, chunkVecs[i]))
  const bm = bm25Scores(query)
  const cosN = minmax(cos)
  const bmN = minmax(bm)
  const scored = chunks.map((c, i) => ({
    chunk: c,
    score: cos[i], // 표시·게이트용 코사인
    hybrid: wSem * cosN[i] + (1 - wSem) * bmN[i], // 순위용
  }))
  // 코사인 임계값 통과분만(= 의미적으로 관련) → 하이브리드로 재정렬
  return scored
    .filter((s) => s.score >= threshold)
    .sort((a, b) => b.hybrid - a.hybrid)
    .slice(0, topK)
}
