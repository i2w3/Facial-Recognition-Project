# 比较skimage、pillow和opencv压缩图像100的平均时间
import cv2
import time
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.util import img_as_ubyte


def getRawFid(DataName):
    rawDataPath = 'face/rawdata/'  # 图像路径
    with open(rawDataPath + DataName, 'r') as file:
        array = np.fromfile(file, 'B')
        size = array.size
        reshapeSie = int(size ** 0.5)  # 128 * 128 与 512 * 512
        array = array.reshape(reshapeSie, reshapeSie)
        return array


image = getRawFid("2412")
Image = Image.fromarray(np.uint8(image))


def skiamgeTime():
    skimageResize = resize(image, (128, 128))
    # skimageResize = img_as_ubyte(skimageResize)


def pilTime():
    pilResize = Image.resize((128, 128))


def opencvTime():
    opencvResize = cv2.resize(image, (128, 128))


from timeit import Timer

t1 = Timer("skiamgeTime()", "from __main__ import skiamgeTime")
t2 = Timer("pilTime()", "from __main__ import pilTime")
t3 = Timer("opencvTime()", "from __main__ import opencvTime")
print(t1.timeit(100))
print(t2.timeit(100))
print(t3.timeit(100))
