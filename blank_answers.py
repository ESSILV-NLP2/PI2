import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import fr_core_news_sm
from dateutil.parser import parse
import spacy
from spacy import displacy
from collections import Counter
import random


avoid_tags = ["ADV_", "ADP_", "VERB", "PRON", "PUNC"]

filepath_wac_fr_data = "data/frWac_no_postag_no_phrase_700_skip_cut50.bin"
nlp_fr = fr_core_news_sm.load()



def modify_answer_to_blank(answer, nlp_fr):
    answer = answer.lower()
    doc_fr = nlp_fr(answer)
    important_words_answer = [X for X in doc_fr if X.tag_[:4] not in avoid_tags] # 4 == taille des avoid_tags
    important_word_answer = important_words_answer[random.randint(0, len(important_words_answer) - 1)]
    answer = answer.replace(important_word_answer.text, "___")

    return answer



if __name__ == "__main__":
    answer = "Chez Chirac nous mangeons bien."
    answer_modified = modify_answer_to_blank(answer, nlp_fr)
    print(answer_modified)