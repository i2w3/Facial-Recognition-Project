import gc
import os
import datetime
import skimage.feature
import skimage.segmentation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from skimage.feature import hog
from skimage import exposure

# 提取标签数据
DatasetPath = './face/'
name = ['faceDR', 'faceDS']  # 存放标签的文件名
error = []
labelraw = []  # 元素是英文 funny smiling serious 和 missing
labelnum = []  # 元素 funny smiling serious 和 missing 分别将标签标号为1,2,3,4
for i in name:
    with open(DatasetPath + i, 'r') as file:
        for line in file:
            num = int(line[1:5])
            if 'missing' in line:
                error.append(num)
            else:
                label1 = line.split('_face ', 1)[1].split(') (_prop')[0]  # 通过分割文本提取表情对应的文本，如smiling
                labelraw.append(label1)

# 根据表情对应的文本生成标签，元素1,2,3分别代表 'funny'，'smiling'，'serious'
for element in labelraw:
    if element == 'funny':
        labelnum.append(1)
    if element == 'smiling':
        labelnum.append(2)
    if element == 'serious':
        labelnum.append(3)

# 提取图像数据
ImgPath = './face/rawdata/'
data = []
inferior = []
filelist = os.listdir(ImgPath)
bigerror = []  # 存放尺寸过大图像的编号
rawdata = np.zeros((3991, 16384))  # 存放图像的像素

i = 0
for name1 in filelist:
    with open(ImgPath + str(name1), 'r') as file:
        array = np.fromfile(file, 'B')
        if array.size == 128 * 128:
            rawdata[i] = array
            i += 1
        else:
            bigerror.append(name1)

# 删除尺寸过大的图片
for a in bigerror:
    index = filelist.index(str(a))
    del labelnum[index]


# 提取图像hog特征
def use_hog(x):
    hogdata = np.zeros((x.shape[0], x.shape[1]))  #
    for i in range(x.shape[0]):
        image = x[i].reshape(int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hogdata[i] = hog_image_rescaled.reshape(x.shape[1])
    return hogdata


# 提取图像lbp特征
def use_lbp(x):
    lbpdata = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        image = x[i].reshape(int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))
        img_lbp = skimage.feature.local_binary_pattern(image, 8, 1.0, method='default')
        img_lbp = img_lbp.astype(np.uint8)
        lbpdata[i] = img_lbp.reshape(x.shape[1])
    return lbpdata


# PCA主成分分析降维
def usePCA(x, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(x)
    x_pca_data = pca.transform(x)
    return x_pca_data


# 标准化
def stander(x_data):
    sc = StandardScaler()
    sc.fit(x_data)
    x_all_std = sc.transform(x_data)
    return x_all_std


# 训练多层感知机分类器
def trainBPmodel(size1, size2, max_iter, learning_rate, X_train, X_test, y_train, y_test, X_all_std):
    start = datetime.datetime.now()
    clf = MLPClassifier(solver='lbfgs', learning_rate='adaptive', learning_rate_init=learning_rate,
                        hidden_layer_sizes=(size1, size2), random_state=1, max_iter=max_iter)
    clf.fit(X_train, y_train)
    end = datetime.datetime.now()
    print('spend time:', end - start, 's')
    y_pred1 = clf.predict(X_train)
    right_classified = (y_pred1 == y_train).sum()
    print("bp trainAcc: ", right_classified / y_pred1.size)
    print("bp testAcc: ", clf.score(X_test, y_test))
    scores = cross_val_score(clf, X_all_std, y, cv=5)
    print("cross_val_score Acc: ", scores.sum() / scores.size)
    cross_val_acc = scores.sum() / scores.size
    return cross_val_acc


# 训练感知机分类器
def trainppmodel(times, learning_rate, X_train, X_test, y_train, y_test, X_all_std):
    clf = Perceptron(max_iter=times, eta0=learning_rate, random_state=1)  # 40
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_train)
    right_classified = (y_pred1 == y_train).sum()
    print("pp trainAcc: ", right_classified / y_pred1.size)
    print("pp testAcc: ", clf.score(X_test, y_test))
    scores = cross_val_score(clf, X_all_std, y, cv=5)
    print("score: ", scores)
    print("cross_val_score Acc: ", scores.sum() / scores.size)


# 选择使用hog或pca处理图像
def processdata(x, y, n_components, usepca, usehog):
    if usehog == 1:
        x_data = use_hog(x)
    if usehog == 0:
        x_data = x
    if usepca == 1:
        x_pca_data = usePCA(x_data, n_components)
    if usepca == 0:
        x_pca_data = x_data
    X_all_std = stander(x_pca_data)
    X_train, X_test, y_train, y_test = train_test_split(X_all_std, y, test_size=0.3, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test, X_all_std


# 打印图像
def showimage(image_data, size1, size2):
    image = image_data.reshape(size1, size2)
    plt.imshow(image)
    plt.show()


# 截取人脸图像中嘴巴部分的图片
def cutimage(X):
    X_cut = np.zeros((X.shape[0], 40, 40))
    X_reshape = X.reshape(X.shape[0], 128, 128)
    for i in range(X.shape[0]):
        for j in range(70, 110):
            for k in range(50, 90):
                X_cut[i][j - 70][k - 50] = X_reshape[i][j][k]
    X_cut_reshape = X_cut.reshape(X.shape[0], 1600)
    return X_cut_reshape


# 用两个过滤器对图像进行简单卷积
def extraimage(x, Step):
    x_reshape = x.reshape(x.shape[0], int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))
    step = Step
    size1 = x.shape[0]  # 图像个数
    size4 = 3
    size5 = 3
    size2 = int(((x.shape[1] ** 0.5) - size4) / step + 1)  # output行数
    size3 = int(((x.shape[1] ** 0.5) - size5) / step + 1)  # output列数
    filter1 = np.array([[2, 0, 2], [0, 3, 0], [2, 0, 2]])
    filter2 = np.array([[0, 3, 0], [1, 0, 1], [0, 4, 0]])
    cutdata = np.zeros((size1, size4, size5))
    conv = [[0 for q in range(0)] for w in range(size1)]
    for i in range(size1):
        for j in range(size2):  # 过滤器竖着扫
            for k in range(size3):  # 过滤器横着扫
                for l in range(size4):
                    for h in range(size5):
                        cutdata[i][l][h] = x_reshape[i][l + j * step][h + k * step]
                num = (cutdata[i] * filter1 + cutdata[i] * filter2).sum()
                conv[i].append(num)
    conv = np.array(conv)
    return conv


X = rawdata  # X是展平后的原始图片数据，维度为（3991，16384） 即3991张图片每张有16384个像素
y = np.array(labelnum)  # y是图片的表情标签，元素有1,2,3，分别代表 'funny' ’smiling' 和 'serious'

del rawdata, labelnum, labelraw  # 清空变量，释放内存
gc.collect()

# ----------  裁剪+卷积算法  ------------
X_cut = cutimage(X)  # 截取图像中嘴部的像素，尺寸为40×40
convdata = extraimage(X_cut, 2)  # 对图像进行简单卷积操作
X_train, X_test, y_train, y_test, X_all_std = processdata(convdata, y, n_components=0, usepca=0,
                                                          usehog=0)  # 分割和测试集，不使用pca和hog
trainBPmodel(2, 3, 112, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
# 输出结果：
# spend time: 0:00:00.737491s      bp trainAcc:  0.8775510204081632
# bp testAcc: 0.837228714524207    cross_val_score Acc:  0.8211189331708552

# ----------  原始数据直接拟合  ------------
# X_train, X_test,y_train, y_test,X_all_std= processdata(X, y, n_components=0,usepca=0,usehog=0)      #不使用pca和hog，只对数据进行标准化
# trainBPmodel(2,7,100,1e-3, X_train, X_test, y_train, y_test,X_all_std)
# 输出结果：
# spend time: 0:00:30.347745s       bp trainAcc:  0.9355531686358755
# bp testAcc: 0.7487479131886478    cross_val_score Acc:  0.7253850876958808

# ----------  HOG特征  ------------
# X_train, X_test,y_train, y_test,X_all_std= processdata(X, y, n_components=256,usepca=0,usehog=1)#使用hog
# trainBPmodel(2,7,100,1e-3, X_train, X_test, y_train, y_test,X_all_std)
# 输出结果：
# spend time:  0:00:28.584067s       bp trainAcc:  0.9609738632295023
# bp testAcc: 0.7545909849749582    cross_val_score Acc:  0.7464308428195219

# ----------  LBP特征  -----------
# lbpdata=use_lbp(X)#提取lbp特征
# X_train, X_test,y_train, y_test,X_all_std= processdata(lbpdata, y, n_components=0,usepca=0,usehog=0)#不使用pca和hog
# trainBPmodel(2,7,100,1e-3, X_train, X_test, y_train, y_test,X_all_std)
# 输出结果：
# spend time: 0:00:28.584067         bp trainAcc:  0.9609738632295023
# bp testAcc: 0.7545909849749582     cross_val_score Acc:  0.7464308428195219

# ----------  PCA降维  ------------
# X_train, X_test,y_train, y_test,X_all_std= processdata(X, y, n_components=256,usepca=1,usehog=0)#使用pca
# trainBPmodel(2,7,100,1e-3, X_train, X_test, y_train, y_test,X_all_std)
# 输出结果：
# spend time: 0:00:00.536359s       bp trainAcc:  0.9155030433225922
# bp testAcc: 0.7921535893155259    cross_val_score Acc:  0.7521955769680734

# ----------  LBP+PCA  ------------
# lbpdata=use_lbp(X)#提取lbp特征
# X_train, X_test,y_train, y_test,X_all_std= processdata(lbpdata, y, n_components=256,usepca=1,usehog=0)#使用pca和lbp
# trainBPmodel(2,7,100,1e-3, X_train, X_test, y_train, y_test,X_all_std)
# 输出结果：
# spend time: 0:00:00.678457s       bp trainAcc:  0.9029717150017902
# bp testAcc: 0.7687813021702838    cross_val_score Acc:  0.7246485961906586

# ----------  HOG+PCA  ------------
# X_train, X_test,y_train, y_test,X_all_std= processdata(X, y, n_components=256,usepca=1,usehog=1)#使用pca和hog
# trainBPmodel(2,7,100,1e-3, X_train, X_test, y_train, y_test,X_all_std)
# 输出结果：
# spend time: 0:00:00.517349s       bp trainAcc:  0.9215896885069818
# bp testAcc: 0.8055091819699499    cross_val_score Acc:  0.7724853683164181
