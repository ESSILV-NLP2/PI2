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
french_model = KeyedVectors.load_word2vec_format(filepath_wac_fr_data, binary=True, unicode_errors="ignore")


def is_date(string, fuzzy=False):
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



def modify_answer_to_blank(answer, nlp_fr, number_possible_answers=3):
    answer = answer.lower()
    possible_answers = []
    doc_fr = nlp_fr(answer)
    important_words_answer = [X for X in doc_fr if X.tag_[:4] not in avoid_tags] # 4 == taille des avoid_tags
    important_word_answer = important_words_answer[random.randint(0, len(important_words_answer) - 1)]
    blank_answer = answer.replace(important_word_answer.text, "___")
    if is_date(important_word_answer.text):
        for _ in range(number_possible_answers):
            answer_date = parse(important_word_answer.text)
            new_date = modify_date(answer_date)
            possible_answers.append(new_date)
    else:
        for _ in range(number_possible_answers):
            similar_words = french_model.most_similar(important_word_answer.text)
            random_position_answer = random.randint(0, len(similar_words) - 1)
            other_answer = similar_words[random_position_answer][0] 
            possible_answers.append(other_answer)
        
    return blank_answer, possible_answers



if __name__ == "__main__":
    answer = "Chez Chirac nous mangeons bien."
    answer_modified, possible_answers = modify_answer_to_blank(answer, nlp_fr)
    print(answer_modified)
    print(possible_answers)