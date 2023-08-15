import nltk
import numpy as np
import random
import string

f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# küçük harflere dönüştürme

nltk.download('punkt') # sadece ilk seferde kullanılır.
nltk.download('wordnet') # sadece ilk seferde kullanılır.

sent_tokens = nltk.sent_tokenize(raw)# cümleler dizisine dönüştürme
word_tokens = nltk.word_tokenize(raw)# kelimeler dizisine dönüştür

sent_tokens[:2]
word_tokens[:2]
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("merhaba", "selaam", "merhabalar", "selamlar", "selamun aleykum", "hey")
GREETING_RESPONSES = ["merhaba", "selam", "merhabalar", "selamlar", "aleykum selam", "hey"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

greeting("selam")
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"Üzgünüm, seni anlamıyorum !"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response
flag=True
print("Chatbot: Benim adım Chatbot. Senin chatbotlar hakkındaki sorgularını cevaplıyacağım. Çıkmak istiyorsan bay yazabilirsin.")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bay'):
        if(user_response=='teşekkürler' or user_response=='teşekkür ederim' ):
            flag=False
            print("Chatbot: Rica ederim.")
        else:
            if(greeting(user_response)!=None):
                print("Chatbot: "+greeting(user_response))
            else:
                print("Chatbot: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Chatbot: Bay! Kendine çok iyi bak, tekrar görüşmek üzere..")

