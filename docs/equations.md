# 数式一覧

## functions.py

### Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

### Softmax

$$
y_k = \frac{e^{x_k}}{\sum_{i} e^{x_i}}
$$

数値安定化のため、入力から最大値を引いて計算する。

$$
y_k = \frac{e^{x_k - \max(x)}}{\sum_{i} e^{x_i - \max(x)}}
$$

### Cross Entropy Error

$$
L = -\frac{1}{N} \sum_{n=1}^{N} \log y_{n, t_n}
$$

- $N$: バッチサイズ
- $t_n$: サンプル $n$ の正解ラベルのインデックス
- $y_{n, t_n}$: サンプル $n$ の正解クラスに対するSoftmax出力

ゼロ除算を防ぐため、$\log(y + \epsilon)$（$\epsilon = 10^{-7}$）を使用する。

---

## optimizer.py

### SGD（確率的勾配降下法）

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$

- $\eta$: 学習率（デフォルト: 0.01）
- $\frac{\partial L}{\partial W}$: 損失 $L$ のパラメータ $W$ に対する勾配

### Adam

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{\eta}_t = \eta \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
$$

$$
W \leftarrow W - \hat{\eta}_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

- $\eta$: 学習率（デフォルト: 0.001）
- $\beta_1$: 1次モーメントの減衰率（デフォルト: 0.9）
- $\beta_2$: 2次モーメントの減衰率（デフォルト: 0.999）
- $\epsilon = 10^{-7}$

---

## layers.py

### MatMul

順伝搬:

$$
\mathbf{y} = \mathbf{x} W
$$

逆伝搬:

$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} W^\top, \quad
\frac{\partial L}{\partial W} = \mathbf{x}^\top \frac{\partial L}{\partial \mathbf{y}}
$$

### Sigmoid レイヤ

順伝搬:

$$
y = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

逆伝搬:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot y(1 - y)
$$

### Affine

順伝搬:

$$
\mathbf{y} = \mathbf{x} W + \mathbf{b}
$$

逆伝搬:

$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} W^\top, \quad
\frac{\partial L}{\partial W} = \mathbf{x}^\top \frac{\partial L}{\partial \mathbf{y}}, \quad
\frac{\partial L}{\partial \mathbf{b}} = \sum_n \frac{\partial L}{\partial \mathbf{y}_n}
$$

### Embedding

順伝搬: 重み行列 $W$ から $\text{idx}$ 番目の行ベクトルを取り出す。

$$
\mathbf{y} = W[\text{idx}]
$$

逆伝搬: 該当行に勾配を加算する（同一インデックスが重複する場合は累積加算）。

### SoftmaxWithLoss

Softmax + Cross Entropy Error を統合したレイヤ。

逆伝搬:

$$
\frac{\partial L}{\partial x_k} = \frac{y_k - t_k}{N}
$$

- $y_k$: Softmax出力
- $t_k$: 正解ラベル（one-hot）
- $N$: バッチサイズ

### SigmoidWithLoss

Sigmoid + Cross Entropy Error を統合したレイヤ（二値分類用）。

順伝搬:

$$
y = \sigma(x), \quad L = -\bigl[ t \log y + (1 - t) \log (1 - y) \bigr]
$$

逆伝搬:

$$
\frac{\partial L}{\partial x} = \frac{y - t}{N}
$$

---

## util.py

### コサイン類似度

$$
\text{similarity}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \, \|\mathbf{y}\|}
$$

数値安定化のため、分母に $\epsilon = 10^{-8}$ を加算する。

### PPMI（正の相互情報量）

$$
\text{PPMI}(i, j) = \max\!\left(0, \, \log_2 \frac{C(i, j) \cdot N}{S(i) \cdot S(j)}\right)
$$

- $C(i, j)$: 共起行列の要素
- $N = \sum_{i,j} C(i,j)$: 共起の総数
- $S(i) = \sum_j C(i, j)$: 単語 $i$ の出現回数

### 勾配クリッピング

$$
\|\mathbf{g}\| = \sqrt{\sum_i g_i^2}
$$

$$
\text{if} \; \|\mathbf{g}\| > \theta: \quad \mathbf{g} \leftarrow \frac{\theta}{\|\mathbf{g}\|} \mathbf{g}
$$

- $\theta$: 勾配の最大ノルム

---

## ch04/negative_sampling_layer.py

### EmbeddingDot

$$
\text{out}_n = \sum_h W[\text{idx}_n]_h \cdot h_{n,h} = W[\text{idx}_n] \cdot \mathbf{h}_n
$$

Embedding で取り出したベクトルと隠れ層ベクトルの内積を計算する。

### UnigramSampler（負例サンプリング）

単語の出現確率に 0.75 乗を適用して、低頻度語がサンプリングされやすくなるよう補正する。

$$
P'(w_i) = \frac{P(w_i)^{0.75}}{\sum_j P(w_j)^{0.75}}
$$

### NegativeSamplingLoss

正例と負例それぞれに対してSigmoidWithLossを適用し、損失を合算する。

$$
L = -\log \sigma(\mathbf{h} \cdot \mathbf{w}_{\text{pos}}) - \sum_{k=1}^{K} \log \sigma(-\mathbf{h} \cdot \mathbf{w}_{\text{neg}_k})
$$

- $\mathbf{h}$: 隠れ層のベクトル（コンテキストの平均）
- $\mathbf{w}_{\text{pos}}$: 正例の単語ベクトル
- $\mathbf{w}_{\text{neg}_k}$: $k$ 番目の負例の単語ベクトル
- $K$: 負例のサンプル数（デフォルト: 5）
