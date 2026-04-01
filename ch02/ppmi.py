import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.util import *

text = "You say goodbaye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)
print("convmatrix")
print(C)
print("ppmimatrix")
print(W)
