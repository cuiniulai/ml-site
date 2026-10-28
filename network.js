// =============================================================
// network.js — 从零实现的前馈神经网络（含小批量反向传播）
// 不依赖任何第三方库，纯 JS，方便你在作业里讲解原理。
// =============================================================

// ---------- 工具：随机数（正态分布） ----------
// 用 Box-Muller 变换把均匀分布转成标准正态分布，
// 这样初始化权重时不会全为 0（全 0 会导致对称权重无法打破）。
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- 激活函数及其导数 ----------
// 反向传播需要用到导数。注意：导数里我们用 a（激活后的输出）而不是 z，
// 因为像 tanh 的导数 1-a^2 用 a 表示最方便。
const ACT = {
  tanh:    { f: z => Math.tanh(z),            df: (z, a) => 1 - a * a },
  relu:    { f: z => Math.max(0, z),          df: (z, a) => (z > 0 ? 1 : 0) },
  sigmoid: { f: z => 1 / (1 + Math.exp(-z)),  df: (z, a) => a * (1 - a) },
  linear:  { f: z => z,                       df: (z, a) => 1 },
};

class Network {
  // layerSizes: 例如 [2, 4, 2, 1] 表示 输入2 → 隐藏层4 → 隐藏层2 → 输出1
  // activation: 隐藏层使用的激活函数名（输出层固定用 sigmoid 做二分类）
  constructor(layerSizes, activation, initZero = false) {
    this.sizes = layerSizes;
    this.activation = activation;
    this.weights = []; // weights[l][i][j]：第 l 层，第 i 个输出神经元，连接第 j 个输入
    this.biases = [];

    for (let l = 1; l < layerSizes.length; l++) {
      const inDim = layerSizes[l - 1];
      const outDim = layerSizes[l];
      const W = [];
      for (let i = 0; i < outDim; i++) {
        const row = [];
        for (let j = 0; j < inDim; j++) {
          // He 初始化：用输入维度开方来缩放，帮助深层网络稳定训练
          row.push(initZero ? 0 : randn() * Math.sqrt(2 / inDim));
        }
        W.push(row);
      }
      this.weights.push(W);
      this.biases.push(new Array(outDim).fill(0));
    }
  }

  // ---------- 前向传播 ----------
  // 输入 input 是一维数组（单个样本的特征向量）
  // 返回每一层的 z（线性结果）和 a（激活后结果），方便反向传播使用
  forward(input) {
    const activations = [input];
    const zs = [];
    let a = input;
    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const outDim = W.length;
      const z = new Array(outDim);
      const aNext = new Array(outDim);
      for (let i = 0; i < outDim; i++) {
        let sum = b[i];
        for (let j = 0; j < a.length; j++) sum += W[i][j] * a[j];
        z[i] = sum;
        // 输出层（最后一层）用 sigmoid；其余隐藏层用用户选的激活
        const isOutput = l === this.weights.length - 1;
        aNext[i] = isOutput ? ACT.sigmoid.f(sum) : ACT[this.activation].f(sum);
      }
      zs.push(z);
      activations.push(aNext);
      a = aNext;
    }
    return { activations, zs };
  }

  // ---------- 小批量训练一步 ----------
  // X: 一批样本的特征向量数组；Y: 对应标签（0 或 1）
  // lr: 学习率；l2: L2 正则系数（防止过拟合）
  trainBatch(X, Y, lr, l2) {
    const L = this.weights.length;
    // 梯度累加器，结构和 weights/biases 一致，初始全 0
    const gW = this.weights.map(W => W.map(row => row.map(() => 0)));
    const gB = this.biases.map(b => b.map(() => 0));

    for (let s = 0; s < X.length; s++) {
      const { activations, zs } = this.forward(X[s]);
      const out = activations[L];           // 最终输出（长度=输出维度，这里为 1）
      const y = Y[s];
      // 输出层误差信号（sigmoid + 交叉熵时，导数恰好是 a - y，非常简洁）
      let delta = out.map((o, i) => o - (Array.isArray(y) ? y[i] : y));

      // 从输出层往输入层逐层回传
      for (let l = L - 1; l >= 0; l--) {
        const aPrev = activations[l];        // 本层输入（上一层激活）
        const outDim = this.weights[l].length;
        const inDim = aPrev.length;

        // 累加本层梯度
        for (let i = 0; i < outDim; i++) {
          for (let j = 0; j < inDim; j++) gW[l][i][j] += delta[i] * aPrev[j];
          gB[l][i] += delta[i];
        }

        // 计算上一层的 delta（用于继续往前传）
        if (l > 0) {
          const newDelta = new Array(inDim).fill(0);
          const zPrev = zs[l - 1];
          for (let j = 0; j < inDim; j++) {
            let s2 = 0;
            for (let i = 0; i < outDim; i++) s2 += delta[i] * this.weights[l][i][j];
            const ap = activations[l][j];    // 上一层激活值
            newDelta[j] = s2 * ACT[this.activation].df(zPrev[j], ap);
          }
          delta = newDelta;
        }
      }
    }

    // 用平均梯度（+ 正则项）更新参数
    const m = X.length;
    for (let l = 0; l < L; l++) {
      const outDim = this.weights[l].length;
      const inDim = this.weights[l][0].length;
      for (let i = 0; i < outDim; i++) {
        for (let j = 0; j < inDim; j++) {
          this.weights[l][i][j] -= lr * (gW[l][i][j] / m + l2 * this.weights[l][i][j]);
        }
        this.biases[l][i] -= lr * (gB[l][i] / m);
      }
    }
  }
}

// 暴露到全局，供 app.js 使用
if (typeof window !== 'undefined') window.Network = Network;
if (typeof window !== 'undefined') window.ACT = ACT;
