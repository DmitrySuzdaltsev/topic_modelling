# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:32:13 2019
@author: Suzdaltsev Dmitry
"""

import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

STOP_WORDS = stopwords.words('dutch')

table = pd.read_excel(r'Extract from SPSS customer satisfaction survey Heineken NL.xlsx')

# Объединить нужные столбцы
v1 = table[~table['V3A@'].isin([99999998, 99999997, 99])][['CODERESP', 'V3A@', 'V3ACODING_01']]
v1 = v1[v1.iloc[:, 2] < 18]
v1.columns=["id", "text", "class"]

v2 = table[~table['V3B@'].isin([99999998, 99999997, 99])][['CODERESP', 'V3B@', 'V3BCODING_01']]
v2 = v2[v2.iloc[:, 2] < 18]
v2.columns=["id", "text", "class"]

v3 = table[~table['V3C@'].isin([99999998, 99999997, 99])][['CODERESP', 'V3C@', 'V3CCODING_01']]
v3 = v3[v3.iloc[:, 2] < 18]
v3.columns=["id", "text", "class"]

data = pd.concat([v1, v2, v3], ignore_index=True)
data = data.dropna()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

#text_clf = Pipeline([
#                     ('tfidf', TfidfVectorizer()),
#                     ('clf', RandomForestClassifier())
#                     ])

X = data['text'].tolist()#[:5]
y = data['class'].tolist()#[:5]

#X = ['abc, asd, abc', 'asd', 'abc', 'asd']
#Y = [1,2,1,2]


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stem = SnowballStemmer('dutch')
#stem = PorterStemmer('dutch')


count_vect = CountVectorizer(lowercase=True, stop_words=STOP_WORDS, min_df=5, ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(X)

tf_transformer = TfidfTransformer(use_idf=False, norm='l2', )
X_train_tf = tf_transformer.fit_transform(X_train_counts)
#X_train_tf = X_train_tf.todense()


# создать df из двух списков

X_to_file = pd.DataFrame(np.array(X_train_tf.todense()))
y_to_file = pd.DataFrame(y)

X_to_file.to_csv('data_X.csv', header=False, index=False)
y_to_file.to_csv('data_y.csv', header=False, index=False)

clf = RandomForestClassifier(n_estimators = 5)
clf.fit(X_train_tf, y)

plt.plot(y, 'o')
plt.plot(clf.predict(X_train_tf), '.')
plt.show()


from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB

from scipy.stats import randint as sp_randint

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# RandomForest
param_dist = {"max_depth": [2,3,4, None],
              "max_features": [1, 5, 10, 15, 20, 25, None],
              "min_samples_split": sp_randint(2, 20),
              'min_samples_leaf': [1, 2, 3, 4, 5],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

clf = RandomForestClassifier(n_estimators=10)
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=300, cv=3, verbose=10)

scores = cross_val_score(clf, X_to_file, y, cv=9)


random_search.fit(X_train_tf, y_to_file[0])

print(random_search.best_params_)

clf1 = RandomForestClassifier(n_estimators=10, **random_search.best_params_, verbose=10)
scores = cross_val_score(clf1, X_to_file, y, cv=9)
print(scores)
print(np.mean(scores))
#ypred1 = cross_val_predict(clf1, X_to_file, y, cv=9)


# KNeighborsClassifier()
param_dist = {
        "n_neighbors": np.arange(1, 5),
        'p': [1,2,3,4],
        }

clf = KNeighborsClassifier()
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, cv=3, n_iter=10, verbose=10)
random_search.fit(X_to_file, y_to_file[0])
print(random_search.best_params_)

clf2 = KNeighborsClassifier(**random_search.best_params_)
scores = cross_val_score(clf2, X_to_file, y, cv=9)
print(scores)
print(np.mean(scores))
#ypred2 = cross_val_predict(clf2, X_to_file, y, cv=9)


# MultinomialNB()
param_dist = {
        "alpha": np.arange(0.2, 0.3, 0.02),
        }
clf = MultinomialNB()

random_search = RandomizedSearchCV(clf, param_distributions=param_dist, cv=3, n_iter=1000, verbose=10)
random_search.fit(X_to_file, y_to_file[0])
print(random_search.best_params_)

clf3 = MultinomialNB(**random_search.best_params_)
scores = cross_val_score(clf3, X_to_file, y, cv=9)
print(scores)
print(np.mean(scores))
#ypred3 = cross_val_predict(clf3, X_to_file, y, cv=9)

#res1 = (ypred1 == y)
#res2 = (ypred2 == y)
#res3 = (ypred3 == y)


# SVM()

clf = SVC(kernel='rbf')
param_dist = {
        "C": np.logspace(-4, 4, 20, endpoint=True),
        "degree": [2, 3, 4], 
        'gamma': ['auto'],
        }
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, cv=3, n_iter=10, verbose=10)
random_search.fit(X_to_file, y_to_file[0])
print(random_search.best_params_)

clf4 = SVC(**random_search.best_params_, probability=True)
scores = cross_val_score(clf4, X_to_file, y, cv=9)
print(scores)
print(np.mean(scores))



from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('rf', clf1), ('kn', clf2), ('nb', clf3), ('svm', clf4)], voting='soft')

scores = cross_val_score(eclf, X_to_file, y, cv=9)
print(scores)
print(np.mean(scores))





'''
clf_list = [
        RandomForestClassifier(n_estimators = 100, min_samples_leaf=2, min_samples_split=6),
            
        #SVC(kernel='rbf', degree=2, gamma='auto'),

        KNeighborsClassifier(n_neighbors=10, p=4),

        #GaussianNB(),
        MultinomialNB(),
        #SVC(kernel="linear", C=0.025),            
        MLPClassifier(max_iter=2000, activation='relu'),
        ]


for clf in clf_list:
    scores = cross_val_score(clf, X_to_file, y, cv=9)
    print(clf)
    print(scores, np.mean(scores))
    print()
lda = LatentDirichletAllocation() 
X_to_file = lda.fit_transform(X_to_file)
for clf in clf_list:
    scores = cross_val_score(clf, X_to_file, y, cv=9)
    print(clf, 'after LDA')
    print(scores, np.mean(scores))
    print()



clf.fit(X_train_tf, y)

from sklearn.metrics import confusion_matrix
cnf_matrix = (confusion_matrix(y, clf.predict(X_train_tf)))


from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


#nltk.stem.snowball

'''