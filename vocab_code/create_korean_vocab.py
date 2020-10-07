""" Creates a vocabulary from a tsv file.
"""

import codecs
import example_helper
from deepmoji.create_vocab import VocabBuilder,MasterVocab
from deepmoji.word_generator import TweetWordGenerator

with codecs.open('../data/korean/train_set.tsv', 'rU', 'utf-8') as stream:
    wg = TweetWordGenerator(stream, allow_unicode_text=True)
    vb = VocabBuilder(wg)
    vb.count_all_words()
    vb.save_vocab(path="temp_vocab.npz")
    mv = MasterVocab()
    mv.populate_master_vocab(vocab_path='temp_vocab')
    mv.save_vocab(path_count = "new_count.npz", path_vocab = "new_vocab.json")

# mv = MasterVocab()
# mv.populate_master_vocab(vocab_path ='../data/korean/4e86ac8f-d32f-444d-a462-be2e3d0471f7.npz')
# mv.save_vocab(path_count, path_vocab = './kor_vocab.json', word_limit=100000)
