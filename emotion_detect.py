from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn import preprocessing
from numpy import mean
import pandas as pd
import numpy as np
import time


def delete_low_words():
    review = pd.read_csv('review_translate_delete.csv', encoding='utf-8')
    listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
    dict_score = {}
    a = time.time()
    data = {'listing_id': [], 'comments': []}
    for _, row in listings.iterrows():
        dict_score[int(row['id'])] = row['review_scores_rating']
    # 对每个房子和评论进行匹配
    for i, row in review.iterrows():
        if i % 1000 == 0:
            print(i)
        if type(row['comments']) is not float and int(row['listing_id']) in dict_score.keys():
            blob = TextBlob(row['comments'])
            data['listing_id'].append(row['listing_id'])
            data['comments'].append(blob.sentiment.polarity)
    df = pd.DataFrame(data)
    df.to_csv('review_emotion.csv', index=False)
    print(time.time() - a)


def create_vector():
    review = pd.read_csv('review_emotion.csv', encoding='utf-8')
    listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
    df1 = listings.drop(listings[np.isnan(listings['review_scores_rating'])].index)
    df2 = df1.drop(df1[np.isnan(df1['review_scores_accuracy'])].index)
    df3 = df2.drop(df2[np.isnan(df2['review_scores_cleanliness'])].index)
    df4 = df3.drop(df3[np.isnan(df3['review_scores_checkin'])].index)
    df5 = df4.drop(df4[np.isnan(df4['review_scores_communication'])].index)
    df6 = df5.drop(df5[np.isnan(df5['review_scores_location'])].index)
    listings = df6.drop(df6[np.isnan(df6['review_scores_value'])].index)
    data_new = review.groupby(['listing_id'])['comments'].apply(list).to_frame()
    a = time.time()
    list_review = []
    list_score1 = []
    dict_score = {}

    # 统计每个房子的得分
    for _, row in listings.iterrows():
        dict_score[int(row['id'])] = row['review_scores_rating']
    # 对每个房子和评论进行匹配
    for i, row in data_new.iterrows():
        if i in dict_score.keys():
            list_score1.append(dict_score[i])
            list_review.append(mean(row['comments']))

    # 对评分进行简单的操作
    # label_encoder = preprocessing.LabelEncoder()
    # list_score = label_encoder.fit_transform(list_score)
    # 使用逻辑回归进行预测
    features = np.column_stack((
        list_review,
    ))
    list_score = []
    for i in list_score1:
        if i >= mean(list_score1):
            list_score.append(1)
        else:
            list_score.append(0)
    model = LogisticRegression()
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, list_score, test_size=0.2)
    model.fit(features, list_score)
    y_predict = model.predict(Xtest)
    print(y_predict)
    print(model.score(features, list_score))

    # pccs = np.corrcoef([i for i in features[:, 0]], list_score)
    # print(pccs)
    # x = []
    # for i in y_predict:
    #     if i >= 5:
    #         x.append(5)
    #     elif i <= 0:
    #         x.append(0)
    #     else:
    #         x.append(i)
    # r_score = r2_score(y_true=ytest, y_pred=x)
    # print(r_score)
    # y = ytest
    # plt.xlim((0, 5))
    # plt.ylim((0, 5))
    # plt.scatter(x, y)
    # plt.show()


if __name__ == '__main__':
    create_vector()


