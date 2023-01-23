import pandas as pd
import numpy as np
import nltk
import re  # 正则表达式
from bs4 import BeautifulSoup  # html标签处理
import time

from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from nltk import collections
from googletrans import Translator
from sklearn.neural_network import MLPClassifier
sw_nltk = stopwords.words('english')
"""
1. 翻译为英语
2. 将句子变为单词
3. 删除低频词汇
4. 最后进行训练
"""


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


def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    '''
    # 1.用正则表达式取出符合规范的部分
    review_text = BeautifulSoup(review).get_text()
    # 2.去除标点符号
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 3.词性还原
    tokens = word_tokenize(review_text)
    tokens = [word for word in tokens if word.lower() not in sw_nltk]
    tagged_sent = pos_tag(tokens)  # 获取单词词性

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
    # 4.停用词去除
    lemmas_sent = [word for word in lemmas_sent if word.lower() not in sw_nltk]
    return lemmas_sent


def translate_english():
    review = pd.read_csv('reviews1.csv', encoding='utf-8')
    data = {'listing_id': [], 'comments': []}
    a = time.time()
    num = 0
    # for i, v in review.loc[:, 'comments'].items():
    for _, row in review.iterrows():
        try:
            v = BeautifulSoup(row['comments']).get_text()
            if detect(v) != 'en':
                num += 1
                translator = Translator()
                t = translator.translate(v, so=detect(v), dest='en')
                c = t.text
                c = re.sub("[^a-zA-Z]", " ", c)
                data['listing_id'].append(row['listing_id'])
                data['comments'].append(c)
                # review.loc[i, 'comments'] = t.text
            else:
                data['listing_id'].append(row['listing_id'])
                v = re.sub("[^a-zA-Z]", " ", v)
                data['comments'].append(v)
        except Exception as e:
            data['listing_id'].append(row['listing_id'])
            # v = re.sub("[^a-zA-Z]", " ", row['comments'])
            data['comments'].append(row['comments'])
            print(row['listing_id'])
            print(row['comments'])
    # review.to_csv('review_translate.csv')
    df = pd.DataFrame(data)
    df.to_csv('review_get_translate.csv', index=False)
    print(time.time() - a)
    print(num)


def translate_english_delete():
    review = pd.read_csv('reviews1.csv', encoding='utf-8')
    data = {'listing_id': [], 'comments': []}
    a = time.time()
    num = 0
    for i, row in review.iterrows():
        try:
            v = BeautifulSoup(row['comments']).get_text()
            if detect(v) != 'en':
                num += 1
                translator = Translator()
                t = translator.translate(v, so=detect(v), dest='en')
                c = t.text
                c = re.sub("[^a-zA-Z0-9]", " ", c)
                data['listing_id'].append(row['listing_id'])
                data['comments'].append(c)
            else:
                data['listing_id'].append(row['listing_id'])
                v = re.sub("[^a-zA-Z0-9]", " ", v)
                data['comments'].append(v)
        except Exception as e:
            print(i)
            print(row['comments'])
    # review.to_csv('review_translate.csv')
    df = pd.DataFrame(data)
    df.to_csv('review_translate_delete.csv', index=False)
    print(time.time() - a)
    print(num)


def prepare_data():
    review = pd.read_csv('reviews1.csv', encoding='utf-8')
    data_new = review.groupby(['listing_id'])['comments'].apply(list).to_frame()
    data = {'listing_id': [], 'comments': []}
    a = time.time()
    for i, row in data_new.iterrows():
        if type(row["comments"]) is not float:
            review_content = ' '.join([str(j) for j in row["comments"]])
        else:
            review_content = ''
        data['listing_id'].append(i)
        data['comments'].append(' '.join(review_to_wordlist(review_content)))
    df = pd.DataFrame(data)
    df.to_csv('review_translate_all_word.csv', index=False)
    print(time.time() - a)


def combine_each_house():
    review = pd.read_csv('review_translate_delete.csv', encoding='utf-8')
    data_new = review.groupby(['listing_id'])['comments'].apply(list).to_frame()
    data = {'listing_id': [], 'comments': []}
    a = time.time()
    for i, row in data_new.iterrows():
        if type(row["comments"]) is not float:
            review_content = ' '.join([str(j) for j in row["comments"]])
            data['listing_id'].append(i)
            data['comments'].append(review_content)
    df = pd.DataFrame(data)
    df.to_csv('review_translate_delete_combine.csv', index=False)
    print(time.time() - a)


def count_each_house():
    review = pd.read_csv('review_translate_delete.csv', encoding='utf-8')
    data_new = review.groupby(['listing_id'])['comments'].apply(list).to_frame()
    listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
    df1 = listings.drop(listings[np.isnan(listings['review_scores_rating'])].index)
    df2 = df1.drop(df1[np.isnan(df1['review_scores_accuracy'])].index)
    df3 = df2.drop(df2[np.isnan(df2['review_scores_cleanliness'])].index)
    df4 = df3.drop(df3[np.isnan(df3['review_scores_checkin'])].index)
    df5 = df4.drop(df4[np.isnan(df4['review_scores_communication'])].index)  # 最高
    df6 = df5.drop(df5[np.isnan(df5['review_scores_location'])].index)
    listings = df6.drop(df6[np.isnan(df6['review_scores_value'])].index)
    list_review = []
    list_score1 = []
    dict_score = {}
    for _, row in listings.iterrows():
        dict_score[int(row['id'])] = row['review_scores_rating']
    # 对每个房子和评论进行匹配

    data = {'listing_id': [], 'number': [], 'score': []}
    a = time.time()

    for i, row in data_new.iterrows():
        if type(row["comments"]) is not float and i in dict_score.keys():
            data['listing_id'].append(i)
            data['number'].append(len(row["comments"]))
            data['score'].append(dict_score[i])

    pccs = np.corrcoef(data['number'], data['score'])
    print(pccs)
    df = pd.DataFrame(data)
    df.to_csv('number_combine.csv', index=False)
    print(time.time() - a)


def remove_fre_stop_word(words):
    t = 1e-5  # t 值
    threshold = -1  # 剔除概率阈值
    # 统计单词频率
    int_word_counts = collections.Counter(words)
    total_count = len(words)
    # 计算单词频率
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
    # 计算被删除的概率
    prob_drop = {w: 1 - np.sqrt(t / f) for w, f in word_freqs.items()}
    # 对单词进行采样
    train_words = set()
    for w in words:
        if w and prob_drop[w] > threshold:
            train_words.add(w)
    print(train_words)
    print(len(train_words))
    return train_words


def delete_low_words():
    review = pd.read_csv('review_translate_delete_combine.csv', encoding='utf-8')
    listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
    dict_score = {}
    whole_word = []
    a = time.time()
    data = {'listing_id': [], 'comments': []}
    for _, row in listings.iterrows():
        dict_score[int(row['id'])] = row['review_scores_rating']
    # 对每个房子和评论进行匹配
    for _, row in review.iterrows():
        if type(row['comments']) is not float and row['listing_id'] in dict_score.keys():
            whole_word += row['comments'].split(" ")
    train_words = remove_fre_stop_word(whole_word)
    for i, row in review.iterrows():
        if type(row['comments']) is not float and row['listing_id'] in dict_score.keys():
            comment_list = row['comments'].split(" ")
            core_word = []
            for j in comment_list:
                if j in train_words:
                    core_word.append(j)
            data['listing_id'].append(row['listing_id'])
            data['comments'].append(' '.join(core_word))
    df = pd.DataFrame(data)
    df.to_csv('review_translate_all_delete_low_6751.csv', index=False)
    print(time.time() - a)


def create_vector():
    review = pd.read_csv('review_translate_all_delete_low_87.csv', encoding='utf-8')
    listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
    df1 = listings.drop(listings[np.isnan(listings['review_scores_rating'])].index)
    df2 = df1.drop(df1[np.isnan(df1['review_scores_accuracy'])].index)
    df3 = df2.drop(df2[np.isnan(df2['review_scores_cleanliness'])].index)
    df4 = df3.drop(df3[np.isnan(df3['review_scores_checkin'])].index)
    df5 = df4.drop(df4[np.isnan(df4['review_scores_communication'])].index)
    df6 = df5.drop(df5[np.isnan(df5['review_scores_location'])].index)
    listings = df6.drop(df6[np.isnan(df6['review_scores_value'])].index)
    a = time.time()
    list_review = []
    list_score = []
    dict_score = {}
    # 统计每个房子的得分
    for _, row in listings.iterrows():
        dict_score[int(row['id'])] = row['review_scores_communication']
    # 对每个房子和评论进行匹配
    for _, row in review.iterrows():
        if type(row['comments']) is not float and row['listing_id'] in dict_score.keys():
            list_score.append(dict_score[row['listing_id']])
            list_review.append(row['comments'])
    # 将评论写成向量
    from sklearn.feature_extraction.text import TfidfVectorizer
    # vectorizer = TfidfVectorizer(norm=None)
    # X = vectorizer.fit_transform(list_review)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list_review)
    # 对评分进行简单的操作
    # label_encoder = preprocessing.LabelEncoder()
    # list_score = label_encoder.fit_transform(list_score)
    # 使用逻辑回归进行预测
    model = LinearRegression()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X.toarray(), list_score, test_size=0.2)
    model.fit(Xtrain, ytrain)

    print(model.score(Xtest, ytest))
    print(time.time() - a)


def neural_network():
    review = pd.read_csv('review_translate_all_delete_low_2284.csv', encoding='utf-8')
    listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
    df1 = listings.drop(listings[np.isnan(listings['review_scores_rating'])].index)
    df2 = df1.drop(df1[np.isnan(df1['review_scores_accuracy'])].index)
    df3 = df2.drop(df2[np.isnan(df2['review_scores_cleanliness'])].index)
    df4 = df3.drop(df3[np.isnan(df3['review_scores_checkin'])].index)
    df5 = df4.drop(df4[np.isnan(df4['review_scores_communication'])].index)
    df6 = df5.drop(df5[np.isnan(df5['review_scores_location'])].index)
    listings = df6.drop(df6[np.isnan(df6['review_scores_value'])].index)
    a = time.time()
    list_review = []
    list_score = []
    dict_score = {}
    # 统计每个房子的得分
    for _, row in listings.iterrows():
        dict_score[int(row['id'])] = row['review_scores_communication']
    # 对每个房子和评论进行匹配
    for _, row in review.iterrows():
        if type(row['comments']) is not float and row['listing_id'] in dict_score.keys():
            list_score.append(dict_score[row['listing_id']])
            list_review.append(row['comments'])
    # 将评论写成向量
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list_review)
    # 对评分进行简单的操作
    label_encoder = preprocessing.LabelEncoder()
    list_score = label_encoder.fit_transform(list_score)
    # 使用逻辑回归进行预测
    model = MLPClassifier()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X.toarray(), list_score, test_size=0.2)
    model.fit(Xtrain, ytrain)

    # print(model.coef_)
    # print(model.intercept_)
    print(model.predict(Xtest))
    print(ytest)
    print(model.score(Xtest, ytest))
    print(time.time() - a)


if __name__ == '__main__':
    count_each_house()
