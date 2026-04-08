import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.time_layers import *
from common.optimizer import SGD
from common.util import eval_perplexity
from common.trainer import RnnlmTrainer
from dataset import ptb
from rnnlm import Rnnlm

# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 100
hidden_size = 100  # RNNの隠れ状態ベクトルの要素数
time_size = 35     # RNNを展開するサイズ
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 学習データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_test, _, _ = ptb.load_data("test")
vaocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# モデルの生成
model = Rnnlm(vaocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 1. 勾配クリッピングを適用して学習
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0, 800))

# 2. テストデータで評価
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print(f"test perplexity: {ppl_test}")

# 3. パラメータの保存
model.save_params()