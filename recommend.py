import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# first 1000 articles from news dataset at https://www.kaggle.com/snapcrack/all−the−news
text = pd.read_csv('articles1.csv')
print(text.head())
x = text['content']
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.2)
X = vectorizer.fit_transform(x)
indices = np.arange(x.size)
from sklearn.model_selection import train_test_split

train, test = train_test_split(indices, test_size=0.2)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=3, metric=cosine_distances).fit(X[train])
test = [test[0]]
found = nbrs.kneighbors(X[test], return_distance=False)
test_i = 0
print('text:\n%.300s' % x[test[test_i]])
for i in found[0]:
    print('match %d:\n%.300s' % (i, x[train[i]]))
