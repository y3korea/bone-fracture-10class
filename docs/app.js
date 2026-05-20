/*
 * Bone Fracture Classifier — on-device ONNX inference (브라우저/Capacitor/Electron 공용).
 *
 * 입력: <input type=file> 또는 카메라 캡쳐 이미지
 * 처리: HTMLImageElement → <canvas> resize(IMG_SIZE) → Float32Array (1,3,H,W) ImageNet normalize
 * 추론: ort.InferenceSession.run({input}) → logits → softmax
 */
(() => {
  'use strict';

  /* ── DOM ─────────────────────────────────────────────── */
  const $ = (s) => document.querySelector(s);
  const dropEl  = $('#drop');
  const fileEl  = $('#fileInput');
  const prevEl  = $('#preview');
  const runEl   = $('#runBtn');
  const statEl  = $('#status');
  const resEl   = $('#result');
  const ttaEl   = $('#ttaToggle');
  const badgeEl = $('#modelBadge');
  const infoEl  = $('#modelInfo');

  /* ── 상태 ─────────────────────────────────────────────── */
  let META = null;        // metadata.json
  let SESSION = null;     // ort.InferenceSession
  let selectedFile = null;
  let lastBitmap = null;  // ImageBitmap (drawImage 효율)

  /* ── 초기화 ──────────────────────────────────────────── */
  async function init() {
    try {
      // ort 로딩 확인
      if (typeof ort === 'undefined') throw new Error('onnxruntime-web 미로드 — ort/ort.min.js 확인');

      // WASM 위치 — onnxruntime-web 내부에서 ort.min.js 의 URL 기준으로 .mjs 를 받고,
      // .mjs 가 다시 .wasm 을 부르는데 그 .wasm 은 document URL 기준이라 경로가 어긋난다.
      // → 문서 기준으로 ./ort/ 의 *절대 URL* 을 직접 지정하면 Capacitor(https://localhost/),
      //   Electron(file://), serve(http://localhost:5173) 모두에서 동일하게 동작.
      ort.env.wasm.wasmPaths = new URL('./ort/', document.baseURI).href;
      ort.env.wasm.numThreads = 1;       // 모바일 WebView 호환성 우선 (SharedArrayBuffer 없는 환경)
      ort.env.wasm.simd = true;

      const t0 = performance.now();
      const metaRes = await fetch('./metadata.json', { cache: 'no-cache' });
      if (!metaRes.ok) throw new Error(`metadata.json 로드 실패 (${metaRes.status})`);
      META = await metaRes.json();

      SESSION = await ort.InferenceSession.create('./model.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      const dt = performance.now() - t0;

      // 입출력 이름 확인
      const inName  = SESSION.inputNames[0]  || 'input';
      const outName = SESSION.outputNames[0] || 'logits';
      META._inName  = inName;
      META._outName = outName;

      badgeEl.textContent =
        `${META.model_label || META.model_name}  ·  ${META.img_size}px  ·  ` +
        `${META.class_names.length} classes  ·  loaded ${(dt/1000).toFixed(1)}s`;
      renderInfo();
      statEl.textContent = '이미지를 선택하면 예측이 활성화됩니다.';
    } catch (e) {
      console.error(e);
      badgeEl.innerHTML = `<span class="err">모델 로딩 실패: ${e.message}</span>`;
      statEl.innerHTML  = `<span class="err">${e.message}</span>`;
    }
  }

  /* ── 모델 정보 카드 ──────────────────────────────────── */
  function renderInfo() {
    const m = META;
    const perClass = m.per_class_acc || {};
    const rows = Object.entries(perClass).map(([cls, acc]) =>
      `<tr>
        <td style="padding:3px 8px;">${cls}</td>
        <td style="padding:3px 8px; text-align:right; font-variant-numeric:tabular-nums;">${Number(acc).toFixed(1)}%</td>
      </tr>`
    ).join('');
    infoEl.innerHTML = `
      <div class="chips">
        <span class="chip">model: ${m.model_name}</span>
        <span class="chip">IMG ${m.img_size}</span>
        <span class="chip">classes: ${m.class_names.length}</span>
        <span class="chip">runtime: onnxruntime-web (wasm)</span>
        ${m.source_run ? `<span class="chip">run: ${m.source_run}</span>` : ''}
      </div>
      <div style="margin-top:10px; display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:8px;">
        ${m.test_acc_simple != null
          ? `<div><div style="color:var(--muted); font-size:10px;">Test Acc</div>
             <div style="font-size:16px; font-weight:600;">${Number(m.test_acc_simple).toFixed(2)}%</div></div>`
          : ''}
        ${m.best_val_acc != null
          ? `<div><div style="color:var(--muted); font-size:10px;">Best Val Acc</div>
             <div style="font-size:16px; font-weight:600;">${Number(m.best_val_acc).toFixed(2)}%</div></div>`
          : ''}
      </div>
      ${rows ? `<details style="margin-top:10px;">
        <summary style="cursor:pointer; color:var(--muted); font-size:11px;">클래스별 정확도 보기</summary>
        <table style="width:100%; margin-top:8px; font-size:11px; border-collapse:collapse;">${rows}</table>
      </details>` : ''}
    `;
  }

  /* ── 파일 선택 + 드래그 ───────────────────────────────── */
  ['dragenter', 'dragover'].forEach(ev =>
    dropEl.addEventListener(ev, e => { e.preventDefault(); dropEl.classList.add('dragover'); })
  );
  ['dragleave', 'drop'].forEach(ev =>
    dropEl.addEventListener(ev, e => { e.preventDefault(); dropEl.classList.remove('dragover'); })
  );
  dropEl.addEventListener('drop', e => {
    const f = e.dataTransfer?.files?.[0]; if (f) handleFile(f);
  });
  fileEl.addEventListener('change', e => {
    const f = e.target.files?.[0]; if (f) handleFile(f);
  });

  async function handleFile(f) {
    if (!f.type || !f.type.startsWith('image/')) {
      setError('이미지 파일만 업로드 가능합니다.'); return;
    }
    selectedFile = f;
    statEl.classList.remove('err');
    statEl.textContent = `${f.name} (${(f.size/1024).toFixed(1)} KB)`;
    const url = URL.createObjectURL(f);
    prevEl.src = url; prevEl.style.display = 'block';
    try {
      lastBitmap = await createImageBitmap(f);
    } catch (e) {
      // Safari/old WebView 폴백
      lastBitmap = await new Promise((ok, ng) => {
        const img = new Image();
        img.onload = () => ok(img);
        img.onerror = () => ng(new Error('이미지 디코딩 실패'));
        img.src = url;
      });
    }
    runEl.disabled = !SESSION;
  }

  /* ── 전처리 ──────────────────────────────────────────── */
  function imageToTensor(bitmap, size, mean, std, transforms = null) {
    // transforms: { hflip?:bool, rotateDeg?:number, scale?:number(>1 = upscale+center-crop) }
    const t = transforms || {};
    const canvas = document.createElement('canvas');
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    // 변환은 캔버스 좌표계에서 한 번에 적용
    ctx.save();
    ctx.translate(size / 2, size / 2);
    if (t.rotateDeg) ctx.rotate(t.rotateDeg * Math.PI / 180);
    const drawScale = t.scale || 1;
    if (t.hflip) ctx.scale(-drawScale, drawScale); else ctx.scale(drawScale, drawScale);
    ctx.drawImage(bitmap, -size / 2, -size / 2, size, size);
    ctx.restore();

    const { data } = ctx.getImageData(0, 0, size, size);  // RGBA HWC uint8
    const N = size * size;
    const out = new Float32Array(3 * N);
    // NCHW + (x/255 - mean) / std
    for (let i = 0; i < N; i++) {
      const r = data[i*4    ] / 255;
      const g = data[i*4 + 1] / 255;
      const b = data[i*4 + 2] / 255;
      out[i        ] = (r - mean[0]) / std[0];
      out[i + N    ] = (g - mean[1]) / std[1];
      out[i + 2*N  ] = (b - mean[2]) / std[2];
    }
    return out;
  }

  function softmax(arr) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    const exps = new Float32Array(arr.length);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) { exps[i] = Math.exp(arr[i] - max); sum += exps[i]; }
    for (let i = 0; i < arr.length; i++) exps[i] /= sum;
    return exps;
  }

  /* ── 추론 ────────────────────────────────────────────── */
  async function runInference() {
    if (!SESSION || !lastBitmap) return;
    runEl.disabled = true;
    statEl.classList.remove('err');
    statEl.innerHTML = '<span class="loader"></span> 추론 중…';
    resEl.innerHTML = '<em style="color:var(--muted); font-size:12px;">처리 중…</em>';

    const size = META.img_size;
    const mean = META.normalize_mean;
    const std  = META.normalize_std;
    const inName  = META._inName;
    const numClasses = META.class_names.length;

    const transformsList = ttaEl.checked
      ? [
          {},                                // 원본
          { hflip: true },                   // 좌우반전
          { scale: 1.15 },                   // 10% 확대 + center crop (캔버스가 자동 crop)
          { rotateDeg: 10 },
          { rotateDeg: -10 },
        ]
      : [{}];

    const t0 = performance.now();
    try {
      const probsAccum = new Float32Array(numClasses);
      for (const tf of transformsList) {
        const inputArr = imageToTensor(lastBitmap, size, mean, std, tf);
        const tensor = new ort.Tensor('float32', inputArr, [1, 3, size, size]);
        const out = await SESSION.run({ [inName]: tensor });
        const outName = Object.keys(out)[0];
        const logits = out[outName].data;
        const probs = softmax(logits);
        for (let i = 0; i < numClasses; i++) probsAccum[i] += probs[i];
      }
      for (let i = 0; i < numClasses; i++) probsAccum[i] /= transformsList.length;

      const dt = performance.now() - t0;
      const order = Array.from(probsAccum.keys()).sort((a, b) => probsAccum[b] - probsAccum[a]);
      renderResult(order, probsAccum, dt, transformsList.length > 1);
      statEl.textContent = `✅ 완료 (${dt.toFixed(0)}ms · ${transformsList.length}× 추론)`;
    } catch (e) {
      console.error(e);
      setError('추론 실패: ' + e.message);
      resEl.innerHTML = '<span class="err">예측 실패</span>';
    } finally {
      runEl.disabled = false;
    }
  }

  /* ── 결과 렌더 ───────────────────────────────────────── */
  function renderResult(order, probs, ms, ttaUsed) {
    const names = META.class_names;
    const topI = order[0];
    const topPct = probs[topI] * 100;
    const color =
      topPct > 70 ? 'var(--good)' :
      topPct > 40 ? 'var(--warn)' : 'var(--bad)';
    const confLabel =
      topPct <= 40 ? '— 불확실 (저신뢰)' :
      topPct >= 70 ? '— 높음' : '— 중간';

    let html = `
      <div class="top-pred">
        <div class="label">🏆 TOP-1${ttaUsed ? ' · TTA' : ''}</div>
        <div class="value">${escapeHtml(names[topI])}</div>
        <div class="conf" style="color:${color};">
          신뢰도 ${topPct.toFixed(1)}% ${confLabel}
        </div>
      </div>
      <div class="bars">`;
    const maxPct = Math.max(...order.map(i => probs[i])) * 100 || 1;
    order.forEach((i, rank) => {
      const pct = probs[i] * 100;
      const w = (pct / maxPct * 100).toFixed(2);
      html += `
        <div class="bar ${rank === 0 ? 'top' : ''}">
          <div class="name">${escapeHtml(names[i])}</div>
          <div class="track"><div class="fill" style="width:${w}%"></div></div>
          <div class="pct">${pct.toFixed(1)}%</div>
        </div>`;
    });
    html += '</div>';
    resEl.innerHTML = html;
  }

  /* ── 헬퍼 ────────────────────────────────────────────── */
  function setError(msg) {
    statEl.classList.add('err');
    statEl.textContent = msg;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
  }

  runEl.addEventListener('click', runInference);
  document.addEventListener('DOMContentLoaded', init);
  if (document.readyState !== 'loading') init();
})();
