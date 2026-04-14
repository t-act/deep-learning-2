# ゼロから作るDeep Learning 2 学習用

「ゼロから作るDeep Learning 2 ―自然言語処理編」の学習用リポジトリ。
書籍の内容に沿って、各章のコードを自分で実装しています。

## セットアップ

```bash
uv sync
```

## ディレクトリ構成

```
ch01/  ニューラルネットワークの復習
ch02/  自然言語と単語の分散表現
ch03/  word2vec
ch04/  word2vecの高速化
ch05/  RNN
ch06/  ゲート付きRNN (LSTM)
ch07/  Seq2seq
ch08/  Attention

common/  共通レイヤー・ユーティリティ
dataset/ データセット (PTB, spiral, addition, date)
```

## 各章の内容

### ch01: ニューラルネットワークの復習
- 2層ニューラルネットワークの実装とspiral datasetでの学習

### ch02: 自然言語と単語の分散表現
- 共起行列、PPMI、SVDによる単語の分散表現
- 単語の類似度計算

### ch03: word2vec
- CBOWモデルの実装と学習

### ch04: word2vecの高速化
- Negative Samplingによる高速化
- PTBデータセットでの学習

### ch05: RNN
- SimpleRnnlmの実装と学習

### ch06: ゲート付きRNN (LSTM)
- LSTM言語モデル (BetterRnnlm) の実装
- 勾配クリッピング、ドロップアウト

### ch07: Seq2seq
- Seq2seq、PeekySeq2seqの実装
- 加算データセットでの学習・精度比較

### ch08: Attention
- Attention機構 (WeightSum, AttentionWeight, Attention) の実装
- AttentionSeq2seqの実装と加算データセットでの実験
