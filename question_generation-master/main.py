

#import nltk
#nltk.download('punkt')


text3 = "Lionel Messi is 32 years old."


from pipelines import pipeline


nlp = pipeline("question-generation")


#ceci est un test de mathieu


nlp(text3)
nlp(text3)

nlp(text3)
nlp(text3)

nlp(text3)


import os

cwd = os.getcwd()
print(cwd)
test2 = os.path.join(cwd, "PI2/question_generation-master/data/squad_multitask/test.json")

files = os.listdir(test2)

import nltk
import json

with open(test2) as f:
    squad = json.load(f)
    for article in squad["data"]:
        title = article.get("title", "").strip()
        for paragraph in article["paragraphs"]:
            """print(paragraph)
            print(paragraph['qas'])"""
            for qa in paragraph['qas']:
                if len(qa['answers']) != 0:
                    print(qa)
                    print(qa['answers'])
                    print(qa['answers'][0])
            """on recupere les contextes"""
            context = paragraph["context"].strip()
            """print(context)"""
            # split into sentences
            sents = nltk.sent_tokenize(context)
            """print(sents)"""


            # get positions of the sentences
            positions = []
            for i, sent in enumerate(sents):
                print(i)
                print(sent)
                if i == 0:
                    start, end = 0, len(sent)
                else:
                    start, end = (prev_end + 1), (prev_end + len(sent) + 1)
                prev_end = end
                positions.append({'start': start, 'end': end})

            # get answers
            """ Out of range car il y a des paragraphes avec answers et d'autres avec plausible_answers"""

            answers = [qa['answers'][0] for qa in paragraph['qas']]