// 브라우저 임베딩 + 코사인 검색 (transformers.js, 완전 클라이언트 사이드)
import { pipeline, env } from '@huggingface/transformers'

// 모델 가중치는 HF Hub에서 최초 1회 다운로드 후 브라우저 캐시에 저장
env.allowLocalModels = false

const MODEL_ID = 'Xenova/multilingual-e5-small' // 다국어 소형 임베딩(교차언어 검색)

let extractor = null
let chunks = []
let chunkVecs = []

export async function initRag(onStatus) {
  onStatus?.('임베딩 모델 로딩 중… (첫 방문은 수백 MB 다운로드, 잠시 기다려 주세요)')
  extractor = await pipeline('feature-extraction', MODEL_ID)

  onStatus?.('근거 데이터(청크)를 불러오는 중…')
  const res = await fetch(import.meta.env.BASE_URL + 'chunks.json')
  chunks = await res.json()

  onStatus?.(`청크 ${chunks.length}개 임베딩 계산 중…`)
  chunkVecs = []
  for (const c of chunks) chunkVecs.push(await embed('passage: ' + c.text))

  onStatus?.('준비 완료 — 질문을 입력하세요.')
  return { count: chunks.length }
}

async function embed(text) {
  const out = await extractor(text, { pooling: 'mean', normalize: true })
  return out.data // 정규화된 Float32Array
}

function dot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

// e5 는 질의에 'query:' 접두사 사용. 반환: [{chunk, score}]
// threshold 0.76: e5-small 로 인스코프(0.77+)/자료밖(0.76-) 분리 지점(측정값). LLM 폐쇄형이 백스톱.
export async function retrieve(query, { topK = 4, threshold = 0.76 } = {}) {
  if (!extractor) throw new Error('RAG 미초기화')
  const q = await embed('query: ' + query)
  const scored = chunks
    .map((c, i) => ({ chunk: c, score: dot(q, chunkVecs[i]) }))
    .sort((a, b) => b.score - a.score)
  return scored.filter((s) => s.score >= threshold).slice(0, topK)
}
