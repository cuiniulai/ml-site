// =============================================================
// datasets.js — 玩具数据集生成 + 特征工程
// 四类经典非线性二分类数据：circle / xor / gaussian / spiral
// =============================================================

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// 生成数据集。type: 类型；n: 样本数；noise: 噪声强度(0~0.3)
function makeData(type, n = 200, noise = 0.0) {
  const pts = [];
  const half = Math.floor(n / 2);

  if (type === 'circle') {
    // 同心圆：内圈一类、外环一类。一条直线分不开，必须靠网络把边界“弯”成环形
    for (let i = 0; i < n; i++) {
      const inner = i < half;
      const r = inner ? 0.30 + randn() * 0.04 : 0.70 + randn() * 0.04;
      const t = Math.random() * Math.PI * 2;
      pts.push({ x: r * Math.cos(t), y: r * Math.sin(t), label: inner ? 0 : 1 });
    }
  } else if (type === 'xor') {
    // 异或：对角相同为一类，不同为一类（经典需要隐藏层才能解决）
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1;
      const y = Math.random() * 2 - 1;
      pts.push({ x, y, label: (x > 0) === (y > 0) ? 0 : 1 });
    }
  } else if (type === 'gaussian') {
    // 两团高斯分布
    for (let i = 0; i < n; i++) {
      const c = i < half ? [-0.5, -0.5] : [0.5, 0.5];
      pts.push({ x: c[0] + randn() * 0.15, y: c[1] + randn() * 0.15, label: i < half ? 0 : 1 });
    }
  } else if (type === 'spiral') {
    // 两条交缠的螺旋线（最难，需要较深的网络和更多步数）
    const turns = 3.0;
    for (let i = 0; i < half; i++) {
      const t = (i / half) * turns + randn() * 0.1;
      const r = 0.1 + (i / half) * 0.85 + randn() * 0.02;
      // 类别 0
      pts.push({ x: r * Math.cos(t), y: r * Math.sin(t), label: 0 });
      // 类别 1：旋转 π，和类别 0 交错
      pts.push({ x: r * Math.cos(t + Math.PI), y: r * Math.sin(t + Math.PI), label: 1 });
    }
  }

  // 加噪声（对所有点随机抖动）
  if (noise > 0) {
    for (const p of pts) {
      p.x += randn() * noise;
      p.y += randn() * noise;
    }
  }
  return pts;
}

// 特征工程：把 (x, y) 扩展成更高维的特征向量。
// 只选 X1/X2 时网络是“线性”的，分不了圆环；
// 勾上 X1²、Y1² 后，网络就有了弯曲边界的能力。
function features(p, opts) {
  const x = p.x, y = p.y;
  const f = [];
  if (opts.x1) f.push(x);
  if (opts.y1) f.push(y);
  if (opts.x2) f.push(x * x);
  if (opts.y2) f.push(y * y);
  if (opts.xy) f.push(x * y);
  if (opts.sinx) f.push(Math.sin(x));
  if (opts.siny) f.push(Math.sin(y));
  if (opts.cosx) f.push(Math.cos(x));
  if (opts.cosy) f.push(Math.cos(y));
  return f;
}

if (typeof window !== 'undefined') {
  window.makeData = makeData;
  window.features = features;
}
