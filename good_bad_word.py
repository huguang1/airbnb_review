from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from sklearn import preprocessing
from numpy import mean
import pandas as pd
import numpy as np
import re
import time


def delete_low_words():
    review = pd.read_csv('review_translate_delete_combine.csv', encoding='utf-8')
    listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
    data_new = review.groupby(['listing_id'])['comments'].apply(list).to_frame()
    good_word_bag = []
    bad_word_bag = []

    with open("positive.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            good_word_bag.append(line)

    with open("negative.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            bad_word_bag.append(line)

    dict_score = {}
    a = time.time()
    data = {'listing_id': [], 'good_word': [], 'bad_word': []}
    for _, row in listings.iterrows():
        dict_score[int(row['id'])] = row['review_scores_rating']

    # 对每个房子和评论进行匹配
    for i, row in data_new.iterrows():
        good_word = 0
        bad_word = 0
        if type(row['comments']) is not float and int(i) in dict_score.keys():
            comments = row['comments']
            comment_str = ' '.join(comments)
            if not comment_str:
                continue
            comment_list = comment_str.split(' ')
            for j in comment_list:
                if j in good_word_bag:
                    good_word += 1
                if j in bad_word_bag:
                    bad_word += 1
            print(i)
            data['listing_id'].append(i)
            data['good_word'].append(good_word)
            data['bad_word'].append(bad_word)
    df = pd.DataFrame(data)
    df.to_csv('review_count.csv', index=False)
    print(time.time() - a)


def create_vector():
    review = pd.read_csv('review_count.csv', encoding='utf-8')
    listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
    df1 = listings.drop(listings[np.isnan(listings['review_scores_rating'])].index)
    df2 = df1.drop(df1[np.isnan(df1['review_scores_accuracy'])].index)
    df3 = df2.drop(df2[np.isnan(df2['review_scores_cleanliness'])].index)
    df4 = df3.drop(df3[np.isnan(df3['review_scores_checkin'])].index)
    df5 = df4.drop(df4[np.isnan(df4['review_scores_communication'])].index)
    df6 = df5.drop(df5[np.isnan(df5['review_scores_location'])].index)
    listings = df6.drop(df6[np.isnan(df6['review_scores_value'])].index)
    a = time.time()
    list_good = []
    list_bad = []
    list_score = []
    dict_score = {}

    # 统计每个房子的得分
    for _, row in listings.iterrows():
        dict_score[int(row['id'])] = row['review_scores_rating']
    # 对每个房子和评论进行匹配
    for i, row in review.iterrows():
        if row['listing_id'] in dict_score.keys():
            list_score.append(dict_score[row['listing_id']])
            list_good.append(row['good_word'])
            list_bad.append(row['bad_word'])

    # 对评分进行简单的操作
    # 使用逻辑回归进行预测
    features = np.column_stack((
        list_good,
        # list_bad,
    ))
    model = LinearRegression()
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, list_score, test_size=0.2)
    model.fit(features, list_score)

    print(model.score(Xtest, ytest))
    pccs = np.corrcoef([i for i in features[:, 0]], list_score)
    print(pccs)
    print(time.time() - a)


if __name__ == '__main__':
    create_vector()








