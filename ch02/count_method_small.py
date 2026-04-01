import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.util import *
import numpy as np
import matplotlib.pyplot as plt

text = "You say goodbaye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

# 「2次元」とは全単語を2つの軸で近似的に表すことであり、単語を2つに絞るわけではない
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id,0], U[word_id,1]))
plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()