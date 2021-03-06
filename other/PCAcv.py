import numpy as np
import pandas as pd
from DataProcess import combineSeqData
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pca = PCA()
pipe1 = Pipeline(steps=[('pca', pca), ('NB', GaussianNB())])
pipe2 = Pipeline(steps=[('pca', pca), ('BB', BernoulliNB())])

Data = pd.read_csv('../pdData.csv')
seq = np.array(Data['Seq'], dtype=str)
target = np.array(Data['Face'].values)
target_names = list(set(Data['Face']))

X_digits = combineSeqData(Data)
y_digits = target

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': range(1, 120),
    'pca__svd_solver': ['auto'],
    'pca__whiten': [True]
}
'''
# SVD和whiten的参数：
'pca__svd_solver':['auto', 'full', 'arpack', 'randomized'],
'pca__whiten':[True, False]
'''
kfold = 10
scoring = 'accuracy'

search1 = GridSearchCV(pipe1, param_grid, n_jobs=-1, cv=kfold, scoring=scoring)
search1.fit(X_digits, y_digits)
print("GaussianNB Best parameter (CV score=%0.3f):" % search1.best_score_)
print(search1.best_params_)

search2 = GridSearchCV(pipe2, param_grid, n_jobs=-1, cv=kfold, scoring=scoring)
search2.fit(X_digits, y_digits)
print("BernoulliNB Best parameter (CV score=%0.3f):" % search2.best_score_)
print(search2.best_params_)
