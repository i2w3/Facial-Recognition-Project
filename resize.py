from DataProcess import *
from skimage.transform import resize

from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
'''


from skimage.util import img_as_ubyte

image = getFid("2412")
img = image
image_resized = resize(image, (128, 128), anti_aliasing=True)
image_resized = img_as_ubyte(image_resized)
from PIL import Image


'''
'''
img_tr = Image.fromarray(image)
img_tr.show()

img_tr1 = Image.fromarray(image_resized)
img_tr1.show()
'''
import numpy as np
import pandas as pd
from DataProcess import combineSeqFid, combineSeqData
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Data = pd.read_csv('pdData.csv')

y = np.array(Data['Face'].values)
image = combineSeqFid(Data)

num_folds = 10
scoring = 'accuracy'

'''
feature_types = ['type-2-x', 'type-2-y',
                 'type-3-x', 'type-3-y',
                 'type-4']
'''
feature_types = ['type-2-x']

for feat_t in feature_types:
    data1 = np.zeros((len(image), 128 * 128))
    for data in range(len(image)):
        coord, _ = haar_like_feature_coord(128, 128, feat_t)
        haar_feature = draw_haar_like_feature(image[data], 0, 0, 128, 128, coord, max_n_features=1,
                                              random_state=0)
        data1[data] = haar_feature.flatten()
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(GaussianNB(), data1, y, cv=kfold, scoring=scoring)
    msg = "%s NB: %f (%f)" % (feat_t, cv_results.mean(), cv_results.std())
    print(msg)
