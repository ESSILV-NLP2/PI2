
import torch

#import nltk
#nltk.download('punkt')


text3 = "Lionel Messi is 32 years old."


from pipelines import pipeline


nlp = pipeline("question-generation")


#ceci est un test de mathieu


nlp(text3)



