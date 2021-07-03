from DataProcess import *

imgData = imgDataProcess()
labelData = labelDataProcess()
combineSeq(imgData, labelData)

# 生成'pdData.csv'
Data = pd.read_csv('pdData.csv', index_col=0)
print(list(set(Data['Face'])))

# 生成'pdDataNF.csv'，该文件是将funny标签转为smiling
Data["Face"].replace("funny", "smiling", inplace=True)
print(list(set(Data['Face'])))
Data.to_csv('pdDataNF.csv', encoding='utf-8')
DataNF = pd.read_csv('pdDataNF.csv')
