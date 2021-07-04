import numpy as np
import pandas as pd
from DataProcess import *
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

Data = pd.read_csv('pdData.csv')
X = combineSeqData(Data)
y = np.array(Data['Face'].values)

print("No Cross-validation:")
# raw
[a, w] = kNN1(X, y)
print('K=1 Raw accuracy: %f  ' % a)
[a, w] = kNN2(X, y)
print('K=3 Raw accuracy: %f  ' % a)
[a, w] = kNN3(X, y)
print('K=5 Raw accuracy: %f  ' % a)
[a, w] = kNN4(X, y)
print('K=7 Raw accuracy: %f  ' % a)

# Standardization
x_1 = standardilize(X)
[a, w] = kNN1(x_1, y)
print('K=1 Standardilize accuracy: %f   ' % a)
[a, w] = kNN2(x_1, y)
print('K=3 Standardilize accuracy: %f   ' % a)
[a, w] = kNN3(x_1, y)
print('K=5 Standardilize accuracy: %f   ' % a)
[a, w] = kNN4(x_1, y)
print('K=7 Standardilize accuracy: %f   ' % a)

# Regularization
x_2 = regularization(X)
[a, w] = kNN1(x_2, y)
print('K=1 regularization accuracy: %f   ' % a)
[a, w] = kNN2(x_2, y)
print('K=3 regularization accuracy: %f   ' % a)
[a, w] = kNN3(x_2, y)
print('K=5 regularization accuracy: %f   ' % a)
[a, w] = kNN4(x_2, y)
print('K=7 regularization accuracy: %f   ' % a)

# Binarization
x_3 = binarizer(X)
[a, w] = kNN1(x_3, y)
print('K=1 Binarizer accuracy: %f   ' % a)
[a, w] = kNN2(x_3, y)
print('K=3 Binarizer accuracy: %f   ' % a)
[a, w] = kNN3(x_3, y)
print('K=5 Binarizer accuracy: %f   ' % a)
[a, w] = kNN4(x_3, y)
print('K=7 Binarizer accuracy: %f   ' % a)

# PCA
x_4 = pca(X)
[a, w] = kNN1(x_4, y)
print('K=1 Pca accuracy: %f   ' % a)
[a, w] = kNN2(x_4, y)
print('K=3 Pca accuracy: %f   ' % a)
[a, w] = kNN3(x_4, y)
print('K=5 Pca accuracy: %f   ' % a)
[a, w] = kNN4(x_4, y)
print('K=7 Pca accuracy: %f   ' % a)

print("10 Cross-validation:")
# Standardization
x_1 = standardilize(X)
# Regularization
x_2 = regularization(X)
# Binarization
x_3 = binarizer(X)

pca1 = PCA(n_components=76, svd_solver='auto',
           whiten=True).fit(x_1)
X1 = pca1.transform(x_1)

pca2 = PCA(n_components=76, svd_solver='auto',
           whiten=True).fit(x_2)
X2 = pca2.transform(x_2)

pca3 = PCA(n_components=76, svd_solver='auto',
           whiten=True).fit(x_3)
X3 = pca3.transform(x_3)

num_folds = 10
scoring = 'accuracy'

for name, data in (["Raw", X], ["PCA+std", X1], ["PCA+reg", X2], ["PCA+bin", X3]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(
        KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto',
                             leaf_size=30, p=2, metric='minkowski', metric_params=None,
                             n_jobs=-1), data, y, cv=kfold, scoring=scoring)
    msg = "%s NB: %f" % (name, cv_results.mean())
    print(msg)
