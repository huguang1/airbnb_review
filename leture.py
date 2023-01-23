from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
import re  # 正则表达式
from bs4 import BeautifulSoup  # html标签处理

# tokenizer = CountVectorizer().build_tokenizer()
# print(tokenizer("Here's example very text, isn't it?"))
# print(WhitespaceTokenizer().tokenize("Here's example text, isn't it?"))
# print(word_tokenize("Here’s example very text, isn't it?"))
# print(tokenizer("likes liking liked"))
# print(WhitespaceTokenizer().tokenize("likes liking liked"))
# print(word_tokenize("likes liking liked"))
#
# stemmer = PorterStemmer()
# tokens = word_tokenize("Here's example text, isn't it?")
# stems = [stemmer.stem(token) for token in tokens]
# print(stems)
# tokens = word_tokenize("likes liking liked very")
# stems = [stemmer.stem(token) for token in tokens]
# print(stems)

a = """
We enjoyed our stay very much. The room was comfortable, neat and clean. There were no problems at all and the host family was very helpful and caring. They helped us planning trips or recommended sights.
The house is situated in a calm neighbourhood close the the Luas and different bus lines. 
There are no negative aspects to mention, it was a very satisfying stay. I would recommend it and stay there again whenever I am in Dublin.
"""

def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    '''
    # 用正则表达式取出符合规范的部分
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 小写化所有的词，并转成词list
    tokens = word_tokenize(review_text)
    stemmer = PorterStemmer()
    words = [stemmer.stem(token) for token in tokens]
    # 返回words
    return words


import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords




# 获取单词的词性
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


a = re.sub("[^a-zA-Z]", " ", a)
tokens = word_tokenize(a)  # 分词
tagged_sent = pos_tag(tokens)  # 获取单词词性

wnl = WordNetLemmatizer()
lemmas_sent = []
for tag in tagged_sent:
    wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
    lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
sw_nltk = stopwords.words('english')

lemmas_sent = [word for word in lemmas_sent if word.lower() not in sw_nltk]

print(len(lemmas_sent))
print(lemmas_sent)

# docs = ['This is the first document This is the first document.', 'This is the second second document.', 'And the third one.',
#         'Is this the first document?']
# from sklearn.feature_extraction.text import CountVectorizer
# import nltk
# # nltk.download('stopwords')
# vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
# print(vectorizer)
#
# # vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(docs)
# print(vectorizer.get_feature_names())
# # print(X.toarray())
