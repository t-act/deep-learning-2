import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data("train")

print(f"corpus size = {len(corpus)}")
print(f"corpus[:30] = {corpus[:30]}")
print()
print(f"id_to_word[0] = {id_to_word[0]}")
print(f"id_to_word[1] = {id_to_word[1]}")