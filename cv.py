import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA()
# set the tolerance to a large value to make the example faster

pipe = Pipeline(steps=[('pca', pca), ('NB', GaussianNB())])

from DataProcess import combineSeqData

Data = pd.read_csv('pdData.csv')

seq = np.array(Data['Seq'], dtype=str)
target = np.array(Data['Face'].values)
target_names = list(set(Data['Face']))

X_digits = combineSeqData(Data)

y_digits = target

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': [60],
    'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
    'pca__whiten': [True, False]
}
'''
param_grid = {
    'pca__n_components': range(40, 80),
    'pca__svd_solver':['auto', 'full', 'arpack', 'randomized'],
    'pca__whiten':[True, False]
}
'''
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_digits, y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
