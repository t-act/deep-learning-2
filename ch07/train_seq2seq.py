import argparse
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

parser = argparse.ArgumentParser()
parser.add_argument("--reverse", action="store_true")
parser.add_argument("--peeky", action="store_true")
args = parser.parse_args()

# データセットの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
if args.reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
char_to_id, id_to_char = sequence.get_vocab()

# ハイパーパラメータの設定
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# モデル / オプティマイザ / トレーナーの生成
if args.peeky:
    model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
else:
    model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose)

    acc = float(correct_num) / len(x_test)
    acc_list.append([int(epoch), acc])
    print(f"val acc: {acc*100:.3f}")
    
# save
acc_list = np.array(acc_list)
suffix = ("_reverse" if args.reverse else "") + ("_peeky" if args.peeky else "")
file_name = f"acc{suffix or '_normal'}.npy"
file_path = os.path.join(os.path.dirname(__file__), file_name)
np.save(file_path, acc_list)
print(f"saved: {file_name}")