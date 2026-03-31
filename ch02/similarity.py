import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.util import *

text = "You say goodbaye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
# print(corpus, word_to_id, id_to_word)

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id["you"]]  # "you"の単語ベクトル
c1 = C[word_to_id["i"]]    # "i"の単語ベクトル
print(cos_similarity(c0, c1))