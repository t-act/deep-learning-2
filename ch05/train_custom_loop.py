import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# ハイパーパラメータの設定
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 100

# 学習データの読み込み（データセットを小さくする）
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

"""
モデルは「私」を見て「は」を、「は」を見て「猫」を予測するように学習する
corpus = [「私」「は」「猫」「が」「好き」]
xs = [「私」「は」「猫」「が」]       ← 入力
ts = [「は」「猫」「が」「好き」]     ← 正解
"""
xs = corpus[:-1]  # input
ts = corpus[1:]   # output (教師ラベル)
data_size = int(max(corpus) + 1)
print(f"corpus size: {corpus_size}, vocabulary size: {vocab_size}")

# 学習時に使用する変数
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []  # perplexity

# モデルの生成
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 1. ミニバッチの各サンプルの読み込み開始位置を計算
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 2. ミニバッチの取得
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

         # 勾配を求めパラメータの更新
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 2

    # 3. エポックごとにperplexityの評価
    ppl = np.exp(total_loss / loss_count)
    print(f"| epoch: {epoch+1} | perplexity: {ppl:.2f} |")
    ppl_list.append([int(epoch+1), float(ppl)])
    total_loss, loss_count = 0, 0

# plot
ppl_list = np.array(ppl_list)
plt.plot(ppl_list[:, 0], ppl_list[:, 1])
plt.xlabel("epoch")
plt.ylabe("perplexity")
plt.show()