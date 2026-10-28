// =============================================================
// app.js — 把控件、训练循环、可视化连起来
// =============================================================

// ---------- 状态 ----------
const state = {
  dataset: 'circle',
  noise: 0.05,
  ratio: 0.5,
  features: { x1: true, y1: true, x2: false, y2: false, xy: false, sinx: false, siny: false, cosx: false, cosy: false },
  hidden: [4, 2],
  activation: 'tanh',
  lr: 0.03,
  l2: 0,
  batch: 10,
  speed: 5,
  net: null,
  data: [],
  train: [],
  test: [],
  steps: 0,
  history: [],   // {step, trainLoss, testLoss, trainAcc, testAcc}
  running: false,
};

// ---------- DOM ----------
const $ = id => document.getElementById(id);
const bCanvas = $('boundary'), bCtx = bCanvas.getContext('2d');
const cCanvas = $('curve'), cCtx = cCanvas.getContext('2d');
const nCanvas = $('net'), nCtx = nCanvas.getContext('2d');
// 决策边界用离屏低分辨率计算，再放大，省 CPU
const off = document.createElement('canvas'); off.width = 160; off.height = 160;
const offCtx = off.getContext('2d');

// ---------- 网络 / 数据 构建 ----------
function countFeatures() {
  const opts = state.features;
  return Object.values(opts).filter(Boolean).length;
}

function rebuildNetwork() {
  const inDim = Math.max(1, countFeatures());
  const sizes = [inDim, ...state.hidden, 1];
  state.net = new Network(sizes, state.activation);
  state.steps = 0;
  state.history = [];
  renderBoundary();
  renderNetwork();
  updateStats();
}

function generateData() {
  state.data = makeData(state.dataset, 240, state.noise);
  // 打乱后按比例切分训练/测试
  const shuffled = state.data.slice().sort(() => Math.random() - 0.5);
  const nTrain = Math.floor(shuffled.length * state.ratio);
  state.train = shuffled.slice(0, nTrain);
  state.test = shuffled.slice(nTrain);
}

// ---------- 一步训练 ----------
function trainStep() {
  const net = state.net;
  const X = [], Y = [];
  for (let i = 0; i < state.batch; i++) {
    const p = state.train[Math.floor(Math.random() * state.train.length)];
    X.push(features(p, state.features));
    Y.push(p.label);
  }
  net.trainBatch(X, Y, state.lr, state.l2);
  state.steps++;

  // 每 10 步评估一次损失/准确率（评估本身有开销，不必每步都算）
  if (state.steps % 10 === 0) {
    const tr = evalSet(state.train);
    const te = evalSet(state.test);
    state.history.push({ step: state.steps, trainLoss: tr.loss, testLoss: te.loss, trainAcc: tr.acc, testAcc: te.acc });
    if (state.history.length > 400) state.history.shift();
  }
}

function evalSet(set) {
  let loss = 0, correct = 0;
  for (const p of set) {
    const f = features(p, state.features);
    const out = state.net.forward(f).activations.at(-1)[0];
    const y = p.label;
    const pc = Math.min(Math.max(out, 1e-7), 1 - 1e-7);
    loss += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
    if ((out > 0.5 ? 1 : 0) === y) correct++;
  }
  return { loss: loss / set.length, acc: correct / set.length };
}

// ---------- 可视化：决策边界 ----------
function renderBoundary() {
  const W = off.width, H = off.height;
  const img = offCtx.createImageData(W, H);
  for (let py = 0; py < H; py++) {
    const y = 1 - (py + 0.5) / H * 2;     // 像素 → [-1,1]，y 向上
    for (let px = 0; px < W; px++) {
      const x = (px + 0.5) / W * 2 - 1;
      const out = state.net.forward(features({ x, y }, state.features)).activations.at(-1)[0];
      const t = Math.max(0, Math.min(1, out));
      const conf = Math.abs(t - 0.5) * 2;  // 越接近 0.5 越白（不确定）
      // 类别0偏红，类别1偏蓝
      let R = 230 + (110 - 230) * t;
      let G = 120 + (160 - 120) * t;
      let B = 120 + (235 - 120) * t;
      R = 255 + (R - 255) * conf;
      G = 255 + (G - 255) * conf;
      B = 255 + (B - 255) * conf;
      const idx = (py * W + px) * 4;
      img.data[idx] = R; img.data[idx + 1] = G; img.data[idx + 2] = B; img.data[idx + 3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);
  // 放大到主画布
  bCtx.imageSmoothingEnabled = true;
  bCtx.clearRect(0, 0, bCanvas.width, bCanvas.height);
  bCtx.drawImage(off, 0, 0, bCanvas.width, bCanvas.height);
  drawPoints();
}

function drawPoints() {
  const W = bCanvas.width, H = bCanvas.height;
  const toPx = x => (x + 1) / 2 * W;
  const toPy = y => (1 - (y + 1) / 2) * H;
  // 测试集：空心
  for (const p of state.test) {
    bCtx.beginPath();
    bCtx.arc(toPx(p.x), toPy(p.y), 3, 0, Math.PI * 2);
    bCtx.strokeStyle = p.label ? '#2b6cb0' : '#c53030';
    bCtx.lineWidth = 1.5; bCtx.stroke();
  }
  // 训练集：实心
  for (const p of state.train) {
    bCtx.beginPath();
    bCtx.arc(toPx(p.x), toPy(p.y), 3.5, 0, Math.PI * 2);
    bCtx.fillStyle = p.label ? '#2b6cb0' : '#c53030';
    bCtx.fill();
  }
}

// ---------- 可视化：损失曲线 ----------
function renderCurve() {
  const W = cCanvas.width, H = cCanvas.height;
  cCtx.clearRect(0, 0, W, H);
  cCtx.fillStyle = '#fafafa'; cCtx.fillRect(0, 0, W, H);
  if (state.history.length < 2) return;
  const pad = 30;
  let maxL = 0;
  for (const h of state.history) maxL = Math.max(maxL, h.trainLoss, h.testLoss);
  maxL = maxL || 1;
  const n = state.history.length;
  const xAt = i => pad + (W - 2 * pad) * i / (n - 1);
  const yAt = l => H - pad - (H - 2 * pad) * (l / maxL);

  // 坐标轴
  cCtx.strokeStyle = '#d1d5db'; cCtx.lineWidth = 1;
  cCtx.beginPath(); cCtx.moveTo(pad, pad); cCtx.lineTo(pad, H - pad); cCtx.lineTo(W - pad, H - pad); cCtx.stroke();

  const line = (key, color) => {
    cCtx.strokeStyle = color; cCtx.lineWidth = 2; cCtx.beginPath();
    state.history.forEach((h, i) => {
      const X = xAt(i), Y = yAt(h[key]);
      i === 0 ? cCtx.moveTo(X, Y) : cCtx.lineTo(X, Y);
    });
    cCtx.stroke();
  };
  line('trainLoss', '#c53030');
  line('testLoss', '#2b6cb0');
  // 图例
  cCtx.font = '11px sans-serif';
  cCtx.fillStyle = '#c53030'; cCtx.fillText('训练损失', W - 80, 16);
  cCtx.fillStyle = '#2b6cb0'; cCtx.fillText('测试损失', W - 80, 30);
}

// ---------- 可视化：网络结构 ----------
function renderNetwork() {
  const W = nCanvas.width, H = nCanvas.height;
  nCtx.clearRect(0, 0, W, H);
  nCtx.fillStyle = '#fafafa'; nCtx.fillRect(0, 0, W, H);
  const sizes = state.net.sizes;
  const L = sizes.length;
  const xs = sizes.map((_, l) => (L === 1 ? W / 2 : 40 + (W - 80) * l / (L - 1)));
  const nodeY = layer => {
    const cnt = sizes[layer];
    return Array.from({ length: cnt }, (_, i) => cnt === 1 ? H / 2 : 30 + (H - 60) * i / (cnt - 1));
  };

  // 先画连线（权重）
  for (let l = 0; l < L - 1; l++) {
    const ysL = nodeY(l), ysR = nodeY(l + 1);
    const Wm = state.net.weights[l];
    for (let i = 0; i < Wm.length; i++) {
      for (let j = 0; j < Wm[i].length; j++) {
        const w = Wm[i][j];
        nCtx.beginPath();
        nCtx.moveTo(xs[l], ysL[j]);
        nCtx.lineTo(xs[l + 1], ysR[i]);
        nCtx.strokeStyle = w >= 0 ? 'rgba(43,108,176,0.55)' : 'rgba(197,48,48,0.55)';
        nCtx.lineWidth = Math.min(4, Math.max(0.4, Math.abs(w) * 1.5));
        nCtx.stroke();
      }
    }
  }
  // 再画节点
  for (let l = 0; l < L; l++) {
    const ys = nodeY(l);
    for (const y of ys) {
      nCtx.beginPath();
      nCtx.arc(xs[l], y, 9, 0, Math.PI * 2);
      nCtx.fillStyle = l === L - 1 ? '#4f46e5' : '#94a3b8';
      nCtx.fill();
      nCtx.strokeStyle = '#fff'; nCtx.lineWidth = 1.5; nCtx.stroke();
    }
  }
  // 层标签
  nCtx.fillStyle = '#6b7280'; nCtx.font = '11px sans-serif';
  nCtx.fillText('输入', xs[0] - 12, H - 8);
  nCtx.fillText('输出', xs[L - 1] - 12, H - 8);
}

// ---------- 统计数字 ----------
function updateStats() {
  const last = state.history[state.history.length - 1];
  const fmt = v => (v === undefined ? '—' : v.toFixed(3));
  $('stats').textContent =
    `步数 ${state.steps} · 训练损失 ${fmt(last?.trainLoss)} · 测试损失 ${fmt(last?.testLoss)} · ` +
    `训练准确率 ${fmt(last?.trainAcc)} · 测试准确率 ${fmt(last?.testAcc)}`;
}

// ---------- 主循环 ----------
let frame = 0;
function tick() {
  if (state.running) {
    for (let i = 0; i < state.speed; i++) trainStep();
    renderBoundary();
    if (frame % 5 === 0) renderCurve();
    if (frame % 8 === 0) renderNetwork();
    if (frame % 5 === 0) updateStats();
  }
  frame++;
  requestAnimationFrame(tick);
}

// ---------- 控件绑定 ----------
function readFeatures() {
  document.querySelectorAll('.feat').forEach(cb => {
    state.features[cb.value] = cb.checked;
  });
}
function parseHidden() {
  const raw = $('hidden').value.split(',').map(s => parseInt(s.trim(), 10)).filter(n => n > 0);
  state.hidden = raw.length ? raw : [4, 2];
}

$('dataset').onchange = e => { state.dataset = e.target.value; generateData(); rebuildNetwork(); };
$('noise').oninput = e => { state.noise = +e.target.value; $('noiseVal').textContent = state.noise.toFixed(2); generateData(); rebuildNetwork(); };
$('ratio').oninput = e => { state.ratio = +e.target.value / 100; $('ratioVal').textContent = e.target.value + '%'; generateData(); rebuildNetwork(); };
$('regen').onclick = () => { generateData(); rebuildNetwork(); };

document.querySelectorAll('.feat').forEach(cb => cb.onchange = () => { readFeatures(); rebuildNetwork(); });
$('hidden').onchange = () => { parseHidden(); rebuildNetwork(); };
$('activation').onchange = e => { state.activation = e.target.value; rebuildNetwork(); };

$('lr').oninput = e => { state.lr = +e.target.value; $('lrVal').textContent = state.lr.toFixed(3); };
$('l2').oninput = e => { state.l2 = +e.target.value; $('l2Val').textContent = state.l2.toFixed(3); };
$('batch').oninput = e => { state.batch = Math.max(1, +e.target.value || 1); };
$('speed').oninput = e => { state.speed = +e.target.value; $('speedVal').textContent = state.speed; };

$('run').onclick = () => {
  state.running = !state.running;
  $('run').textContent = state.running ? '⏸ 暂停' : '▶ 开始';
};
$('step').onclick = () => { trainStep(); renderBoundary(); renderCurve(); renderNetwork(); updateStats(); };
$('reset').onclick = () => { rebuildNetwork(); };

// ---------- 启动 ----------
parseHidden();
readFeatures();
generateData();
rebuildNetwork();
tick();
