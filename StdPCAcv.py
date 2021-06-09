import numpy as np
import pandas as pd
from DataProcess import combineSeqData
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('NB', GaussianNB())])

Data = pd.read_csv('pdData.csv')
seq = np.array(Data['Seq'], dtype=str)
target = np.array(Data['Face'].values)
target_names = list(set(Data['Face']))

X_digits = combineSeqData(Data)
sc = StandardScaler()
sc.fit(X_digits)
X_digits = sc.transform(X_digits)
y_digits = target

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': range(30, 90),
    'pca__svd_solver': ['auto'],#, 'full', 'arpack', 'randomized'
    'pca__whiten': [True]#, False
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_digits, y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)