// postinstall: node_modules/onnxruntime-web/dist/ → www/ort/ 로 필수 파일 복사.
// (브라우저/Capacitor/Electron 모두 같은 경로에서 wasm 을 로드)
const fs = require('fs');
const path = require('path');

const SRC = path.resolve(__dirname, '..', 'node_modules', 'onnxruntime-web', 'dist');
const DST = path.resolve(__dirname, '..', 'www', 'ort');

if (!fs.existsSync(SRC)) {
  console.warn(`⚠️  onnxruntime-web not installed yet at ${SRC} — skip.`);
  process.exit(0);
}
fs.mkdirSync(DST, { recursive: true });

// JS 한 개 + 우리가 실제로 쓰는 wasm 만 복사 (전체는 100MB 가까이 됨)
const NEEDED = [
  'ort.min.js',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
];
let copied = 0;
for (const f of NEEDED) {
  const src = path.join(SRC, f);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, path.join(DST, f));
    copied++;
  } else {
    // 버전이 다르면 폴백 — wasm 전부 복사
    console.warn(`(skip missing) ${f}`);
  }
}

// 폴백: 위 NEEDED 중 빠진 게 있으면 dist 의 모든 .wasm 과 .mjs 를 복사
if (copied < NEEDED.length) {
  for (const f of fs.readdirSync(SRC)) {
    if (f.endsWith('.wasm') || f.endsWith('.mjs') || f === 'ort.min.js') {
      fs.copyFileSync(path.join(SRC, f), path.join(DST, f));
    }
  }
}

console.log(`✅ ort assets → ${path.relative(process.cwd(), DST)}`);
