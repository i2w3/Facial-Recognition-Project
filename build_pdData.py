from DataProcess import *
imgData = imgDataProcess()
labelData = labelDataProcess()
combineSeq(imgData, labelData)  # 生成'pdData.csv'
Data = pd.read_csv('pdData.csv')