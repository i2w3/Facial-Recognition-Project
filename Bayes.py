import numpy as np
import pandas as pd
from DataProcess import combineSeqFid, combineSeqData
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

Data = pd.read_csv('pdData.csv')
X = combineSeqData(Data)
y = np.array(Data['Face'].values)

pca = PCA(n_components=76, svd_solver='auto',
          whiten=True).fit(X)
X2 = pca.transform(X)
# 74 0.601884
# 75 0.607680
# 76 0.608690
# 77 0.609195
# 78 0.607679

num_folds = 10
scoring = 'accuracy'

kfold = model_selection.KFold(n_splits=num_folds)
cv_results = model_selection.cross_val_score(GaussianNB(), X2, y, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("NB", cv_results.mean(), cv_results.std())
print(msg)
