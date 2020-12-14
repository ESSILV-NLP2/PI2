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


avoid_tags = ["ADV_", "ADP_", "VERB", "PRON"]

filepath_wac_fr_data = "data/frWac_no_postag_no_phrase_700_skip_cut50.bin"
nlp_fr = fr_core_news_sm.load()
french_model = KeyedVectors.load_word2vec_format(filepath_wac_fr_data, binary=True, unicode_errors="ignore")


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def modify_date(answer_date):
    int_confidence_year = 3
    sooner_random_answer_tuple = (answer_date.year - int_confidence, answer_date.month - random.randint(0, answer_date.month), answer_date.day - random.randint(0, answer_date.day))
    new_answer_year = answer_date.year - random.randint(0, int_confidence_year)
    new_answer_month = answer_date.month - random.randint(0, answer_date.month - 1)
    new_answer_day = answer_date.day - random.randint(0, answer_date.day - 1)
    return datetime.datetime(new_answer_year, new_answer_month, new_answer_day)



def modify_answer_to_incorrect(answer, nlp_fr, model_fr):
    answer = answer.lower()
    doc_fr = nlp_fr(answer)
    important_words_answer = [X for X in doc_fr if X.tag_[:4] not in avoid_tags] # 4 == taille des avoid_tags
    for important_word_answer in important_words_answer:
        try:
            if is_date(important_word_answer.text):
                answer_date = parse(important_word_answer.text)
                new_date = modify_date(answer_date)
                answer = answer.replace(important_word_answer.text, str(new_date))
            else:
                similar_words = french_model.most_similar(important_word_answer.text)
                random_position_answer = random.randint(0, len(similar_words) - 1)
                other_answer = similar_words[random_position_answer][0] 
                answer = answer.replace(important_word_answer.text, other_answer)
        except KeyError:
            pass
    return answer



if __name__ == "__main__":
    answer = "Chez Chirac nous mangeons bien."
    answer_modified = modify_answer_to_incorrect(answer, nlp_fr, french_model)
    print(answer_modified)