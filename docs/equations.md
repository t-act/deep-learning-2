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
