import pandas as pd
import numpy as np
from nltk import flatten
from googletrans import Translator
np.set_printoptions(threshold=np.inf)
from langdetect import detect, LangDetectException
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import re
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer

df3 = pd.read_csv("reviews.csv")
listing_id = df3["listing_id"]
comments = df3.iloc[:,5]
import json


#read comments from csv and process it and save it to json
def read_comment_fromID(listing_id):
    reviews = {}
    bag_list = []
    for index,value in listing_id.items():
        print("comment"+str(index))
        reviews.setdefault(value,[])
        comment,bag_of_word = process_comments(comments[index])
        bag_list = bag_list + bag_of_word
        reviews[value].append(comment)
    f = open('bag_all_word.txt', 'w')
    for i in bag_list:
        f.write(str(i) + '\n')
    print("success save bag_all!!!!!!!!!!")
    bag = []
    [bag.append(x) for x in bag_list if x not in bag]
    return reviews,bag


def reviews_bag(reviews,bag):
    reviews_bag = {}
    vectorizer = CountVectorizer(vocabulary=bag)
    #print(vectorizer.get_feature_names_out())
    for key,values in reviews.items():
        reviews_bag.setdefault(int(key), [])
        values = flatten(values)

        X = vectorizer.fit_transform(values)
        arr = X.toarray()
        bag_of_word_array = arr[0]
        for i in range(0, len(arr) - 1):
            bag_of_word_array = bag_of_word_array + arr[i + 1]
        reviews_bag[int(key)].append(list((bag_of_word_array)))

    return reviews_bag



def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def process_comments(comment):
    #process sentence
    if type(comment) == float or comment[0].isalpha() == False:
        comment = 'good'
    if detect(comment)!='en':
        translator = Translator()
        t = translator.translate(comment, src=detect(comment), dest='en')
        comment = t.text
    sentence = nltk.sent_tokenize(comment)
    # generate token and stemming and remove stopwords and remove Punctuation and special words
    words = []
    bag_of_word = []
    for s in sentence:
        s = s.replace('<br/>','')
        s = re.sub(r'[^a-zA-Z0-9\s]', '', string=s)
        s = s.lower()
        token = word_tokenize(s)
        tagged_s = nltk.pos_tag(token)
        lemmas=[]
        for t in tagged_s:
            wordnet_pos = get_wordnet_pos(t[1]) or wordnet.NOUN
            lemmas.append(wnl.lemmatize(t[0], pos=wordnet_pos))
        word = lemmas
        word = [word for word in word if word not in stopwords.words('english')]
        if len(word)==0:
            continue
        else:
            words.append(word)
            bag_of_word = bag_of_word + word
    return words,bag_of_word


if __name__ == '__main__':

    # s = set()
    # for i in listing_id:
    #     s.add(i)
    # print(len(s))

    # reviews,bag = read_comment_fromID(listing_id)
    #
    # with open('../FinalML/reviews.json', 'w') as f:
    #     json.dump(reviews, f)
    #
    # print("success save comments!!!!!!!!!!")
    #
    # f = open('bag.txt', 'w')
    # for i in bag:
    #     f.write(str(i) + '\n')
    # print("success save bag!!!!!!!!!!")


    import collections

    bag_all_word = []
    with open('review_all_word.csv', 'r') as f:
        for line in f:
            bag_all_word.append(list(line.strip('\n').split(',')))
    bag_all_word = flatten(bag_all_word)

    bag_all_word = collections.Counter(bag_all_word)

    bag = []
    bag_list = list(bag_all_word.most_common(50000))
    for i in range(0,50000):
       bag.append(bag_list[i][0])



    # with open("reviews.json", 'r', encoding='UTF-8') as f:
    #     reviews = json.load(f)




    # class NpEncoder(json.JSONEncoder):
    #     def default(self, obj):
    #         if isinstance(obj, np.integer):
    #             return int(obj)
    #         if isinstance(obj, np.floating):
    #             return float(obj)
    #         if isinstance(obj, np.ndarray):
    #             return obj.tolist()
    #         return super(NpEncoder, self).default(obj)
    #
    # re = reviews_bag(reviews,bag)
    #
    # with open('../FinalML/reviews_bag.json', 'w') as f:
    #     json.dump(re, f,cls=NpEncoder)












