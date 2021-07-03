from DataProcess import *

imgData = imgDataProcess()
labelData = labelDataProcess()
combineSeq(imgData, labelData)

# 生成'pdData.csv'
Data = pd.read_csv('pdData.csv')

# 生成'pdDataNF.csv'，该文件是将funny标签转为smiling
Data.loc[Data['Face'] == "funny"] = "smiling"
print(list(set(Data['Face'])))
Data.to_csv('pdDataNF.csv', encoding='utf-8')
