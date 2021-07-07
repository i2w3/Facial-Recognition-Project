from BP import *

# -------  打印不同训练次数的准确率曲线  ---------
X_cut = cutimage(X)  # 截取图像中嘴部的像素，尺寸为40×40
convdata = extraimage(X_cut, 2)
Acc_conv = []
Acc_pca = []
X_train, X_test, y_train, y_test, X_all_std = processdata(convdata, y, n_components=0, usepca=0, usehog=0)
for i in range(2, 150, 2):
    acc1 = trainBPmodel(2, 3, i, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
    Acc_conv.append(acc1)
X_train, X_test, y_train, y_test, X_all_std = processdata(X, y, n_components=256, usepca=1, usehog=0)
for j in range(2, 150, 2):
    acc2 = trainBPmodel(2, 3, j, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
    Acc_pca.append(acc2)
x = [i for i in range(2, 150, 2)]
y1, y2 = Acc_conv, Acc_pca
plt.title("Comparison of different algorithms")
plt.xlabel("train iteration")
plt.ylabel("cross validation accuracy")
plt.plot(x, y1, color="red", linewidth=1.5, linestyle="-", label="cut out+simple convolution")
plt.plot(x, y2, color="green", linewidth=1.5, linestyle="-", label="pca")
plt.legend(loc='lower right')
x0 = 112
y0 = 0.821
plt.scatter(x0, y0, s=60, color='g')
plt.annotate(r"corss validation accuracy= 0.821", xy=(x0, y0), xycoords='data', xytext=(-50, -30),
             textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.savefig('./plot/plotBPTrainAcc.png', bbox_inches='tight')
plt.show()
