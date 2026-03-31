import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.optimizer import SGD
from dataset import spiral
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

# 1. ハイパーパラメータの設定
max_epoch = 300  # epoch -> 学習の単位(1epoch=学習データをすべて見たとき)
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 2. データの読み込み、モデルとオプティマイザの生成
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# 学習で使用する変数
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # 3. データのシャッフル
    idx = np.random.permutation(data_size)  # 既存の配列をシャッフル・ランダム並び替え
    x = x[idx]  # シャッフル順に並び替え
    t = t[idx]

    for iters in range(max_iters):
        """
        iters=0: x[0*30 : 1*30] = x[0:30]    → 1〜30件目
        iters=1: x[1*30 : 2*30] = x[30:60]   → 31〜60件目
        iters=2: x[2*30 : 3*30] = x[60:90]   → 61〜90件目
        """
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]
        
        # 4. 勾配を求めデータを更新（学習）
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 5. 定期的に損失を可視化
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d |  iter %d / %d | loss %.2f'
                  % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

# ↓ 参考コードをそのまま移植
# 学習結果のプロット
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()

# 境界領域のプロット
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# データ点のプロット
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()
