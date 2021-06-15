import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# 定义数据路径
labelPath = 'face/'  # 标签路径
labelFile = ['faceDR', 'faceDS']  # 标签文件
rawDataPath = 'face/rawdata/'  # 图像路径
outputImgPath = 'out/'  # 输出图像路径


# 将二进制的图像数据集转为矩阵，同时负责压缩某些图像，输入DataName（序号），输出矩阵
def getFid(DataName):
    with open(rawDataPath + DataName, 'r') as file:
        array = np.fromfile(file, 'B')
        size = array.size
        reshapeSie = int(size ** 0.5)  # 128 * 128 与 512 * 512
        array = array.reshape(reshapeSie, reshapeSie)
        # 将图像从 512*512 压缩到 128*128
        if array.shape == (512, 512):
            array = cv2.resize(array, (128, 128), interpolation=cv2.INTER_LINEAR)
            print(DataName, '已经从 512 * 512 压缩到 128 * 128')
        return array


# 图像数据简单处理，过滤全黑的图像数据（灰度值全为0），返回合适图像数据的序列
def imgDataProcess():
    imgData = []  # 存放合适图像数据的序列
    skip = []
    for dataName in os.listdir(rawDataPath):
        fid = getFid(dataName)
        if np.all(fid == 0):  # 这些图像数据全为 0，保存序号到skip
            skip.append(dataName)
            continue
        imgData.append(dataName)
    print('\n图像数据为纯黑：\n', skip)
    imgData = pd.DataFrame(imgData, columns=['Seq'])
    return imgData


# 标签数据处理，将去掉缺少标签和完善道具标签栏，返回选择的序号与修改好的标签
def labelDataProcess():
    labelData = []
    miss = []
    for dataName in labelFile:
        with open(labelPath + dataName, 'r') as file:
            for line in file:
                match = re.findall(r"(\d+)|\b_missing.descriptor|\b_(?:sex|age|[rf]ace)\s+(\w+)|_prop..\(([^()]*).\)\)",
                                   line)
                labelList = list(["".join(x) for x in match])

                if len(labelList) == 2:
                    miss.append(labelList[0])
                    continue
                elif len(labelList) == 6:
                    pass
                else:
                    labelList.append('NoProp')
                # if labelList[4] == 'funny':
                # labelList[4] = 'smiling'
                labelData.append(labelList)
    print('\n标签数据丢失：\n', miss)
    labelData = pd.DataFrame(labelData, columns=['Seq', 'Sex', 'Age', 'Race', 'Face', 'Prop'])
    return labelData


# 合并 经过图像数据处理后筛选出的序号 与 经过标签处理后筛选出的序号，并保存为'pdData.csv'
def combineSeq(imgData, labelData):
    Different = list(set(imgData.iloc[:, 0]) ^ set(labelData.iloc[:, 0]))  # 找出图像数据与标签数据不同的序号
    Seq = [imgData.iloc[:, 0], labelData.iloc[:, 0]][len(imgData) > len(labelData)]  # [A,B][True]返回B

    Data = pd.DataFrame(columns=['Seq', 'Sex', 'Age', 'Race', 'Face', 'Prop'])
    for i in Seq:
        if i in Different:
            print("序号", i, "缺少图像数据或标签数据")
            continue
        a = imgData[imgData['Seq'].isin([i])]
        b = labelData[labelData['Seq'].isin([i])]
        Data = Data.append(pd.merge(a, b, how='left', on='Seq'))
    Data.index = np.arange(len(Data))
    Data.to_csv('pdData.csv', encoding='utf-8')


# 提取合并的序号对应的图像数据（128 * 128）
def combineSeqFid(DataFrame):
    img = np.zeros((len(DataFrame), 128, 128))
    k = range(len(DataFrame))
    for (i, j) in zip(DataFrame['Seq'], k):
        fid = getFid(str(i))
        img[j] = fid
    return img


# 扁平化合并序号对应的图像数据（一维向量）
def combineSeqData(DataFrame):
    data = np.zeros((len(DataFrame), 128 * 128))
    k = range(len(DataFrame))
    for (i, j) in zip(DataFrame['Seq'], k):
        fid = getFid(str(i))
        data[j] = fid.flatten()
    return data


# 将图像数据转为jpg格式的图片
def outputImages(outputImagesPath=outputImgPath):
    if not os.path.exists(outputImagesPath):  # 检查要输出图像的目录outputImagesPath是否存在
        os.mkdir(outputImagesPath)  # 不存在，创建该目录
    for Img in os.listdir(rawDataPath):
        fid = getFid(Img)
        im = Image.fromarray(fid)
        im.save(os.path.join(outputImagesPath, Img) + '.jpg')
