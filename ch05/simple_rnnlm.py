import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.time_layers import *

class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 重みの初期化
            # Xavierの初期化
            # 層を通しても値の大きさが変わらないようにするため
        embed_W = (rn(V, D) / 100).astype("f")
        rnn_Wx = (rn(V, D) / np.aqrt(D)).astype("f")
        rnn_Wh = (rn(D, H) / np.aqrt(H)).astype("f")
        rnn_b = np.zeros(H).astype("f")
        affine_W = (rn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1] # TimeRNNを使用

        # すべての重みと勾配をリストにまとめる
        self.params, self.gards = [], []
        for layer in self.layers:
            self.params += layer.params
            self.gards += layer.grads
    
    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()