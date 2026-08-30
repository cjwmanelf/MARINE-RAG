// Ollama CORS + Private Network Access(PNA) 프록시 (의존성 0, Node 18+)
//
// 왜 필요한가: 배포된 HTTPS 페이지(GitHub Pages)가 http://localhost:11434(Ollama)를
// 직접 부르면 브라우저의 PNA 정책이 막는다(Ollama가 PNA 허용 헤더를 안 보냄).
// 이 프록시는 브라우저 요청에 CORS + `Access-Control-Allow-Private-Network: true`를
// 붙여 응답하고, 서버-사이드로 Ollama에 그대로 전달한다(그 구간은 CORS 무관).
//
// 실행:  node ollama-proxy.mjs      (Ollama가 11434에서 실행 중이어야 함)
// 그러면 배포 사이트가 http://localhost:11435 를 통해 답변 생성 가능.

import http from 'node:http'

const TARGET = process.env.OLLAMA_TARGET || 'http://localhost:11434'
const PORT = Number(process.env.PROXY_PORT || 11435)

function corsHeaders(req) {
  return {
    'Access-Control-Allow-Origin': req.headers.origin || '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': req.headers['access-control-request-headers'] || 'Content-Type',
    'Access-Control-Allow-Private-Network': 'true',
    Vary: 'Origin',
  }
}

http
  .createServer((req, res) => {
    const cors = corsHeaders(req)
    if (req.method === 'OPTIONS') {
      res.writeHead(204, cors)
      res.end()
      return
    }
    const chunks = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', async () => {
      try {
        const upstream = await fetch(TARGET + req.url, {
          method: req.method,
          headers: { 'Content-Type': 'application/json' },
          body: req.method === 'POST' ? Buffer.concat(chunks) : undefined,
        })
        res.writeHead(upstream.status, {
          ...cors,
          'Content-Type': upstream.headers.get('content-type') || 'application/json',
        })
        const reader = upstream.body.getReader()
        const pump = async () => {
          const { value, done } = await reader.read()
          if (done) return res.end()
          res.write(Buffer.from(value))
          return pump()
        }
        await pump()
      } catch (e) {
        res.writeHead(502, cors)
        res.end(JSON.stringify({ error: String(e) }))
      }
    })
  })
  .listen(PORT, () => {
    console.log(`[proxy] Ollama CORS+PNA 프록시: http://localhost:${PORT}  →  ${TARGET}`)
    console.log('[proxy] 배포(HTTPS) 사이트에서 이 포트로 답변 생성이 가능합니다. (Ollama도 함께 실행)')
  })
