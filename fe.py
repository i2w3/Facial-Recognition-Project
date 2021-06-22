import numpy as np
import pandas as pd
from DataProcess import combineSeqFid
from dask import delayed
from skimage.transform import integral_image
from skimage.feature import haar_like_feature


def extract_feature_image(img, feature_type, feature_coord=None):
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


Data = pd.read_csv('pdData.csv')

images = combineSeqFid(Data)  # 根据序号读取图像（128 * 128）
y = np.array(Data['Face'].values)  # 读取分类类型的标签
feature_types = ['type-2-x', 'type-2-y']

X1 = extract_feature_image(images, feature_types[0])
X2 = extract_feature_image(images, feature_types[1])