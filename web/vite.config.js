import { defineConfig } from 'vite'

// base:'./' → GitHub Pages 하위 경로(/<repo>/)에서도 자산을 상대경로로 로드
export default defineConfig({
  base: './',
  build: { target: 'es2022', outDir: 'dist' },
  // transformers.js(onnxruntime-web)는 사전 번들 최적화에서 제외해야 wasm 로딩이 안정적
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
})
