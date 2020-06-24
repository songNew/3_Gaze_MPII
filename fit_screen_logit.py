import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# 导入线性模型和多项式特征构造模块
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import externals
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report
from matplotlib import pylab

import pickle


def plot_pr(auc_score, precision, recall, label=None):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(recall, precision, lw=1)
    pylab.show()


vector_dataFrame = pd.read_csv("gaze_vector_for_class.txt", sep='\t', header=None, index_col=False)

dataFrame = vector_dataFrame.drop([1], axis=1)
dataFrame.columns = range(9)
# print(x_dataFrame.head(3))
print(dataFrame.head(3))

X_train, X_test, y_train, y_test = train_test_split(dataFrame.iloc[:, 1:], dataFrame.iloc[:, 0], test_size=0.2,
                                                    shuffle=True)

# poly_reg = PolynomialFeatures(degree=1)
# X_train = poly_reg.fit_transform(X_train)
# X_test = poly_reg.fit_transform(X_test)

lr = linear_model.LogisticRegressionCV(multi_class="ovr", fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=5,
                                       penalty="l2", solver="lbfgs", tol=0.001, max_iter=3000)

print(X_train)
re = lr.fit(X_train, y_train)

# 模型效果获取
r = re.score(X_train, y_train)
print("R值(准确率):", r)
print("参数:", [round(x, 4) for x in re.coef_.flatten()])
print("截距:", re.intercept_)
print("稀疏化特征比率:%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
print("=========sigmoid函数转化的值，即：概率p=========")
print(re.predict_proba(X_test))  # sigmoid函数转化的值，即：概率p
output = open('logistic_lr.pkl', 'wb')
pickle.dump(lr, output, 0)  # 将训练后的线性模型保存
output.close()

pkl_file = open('logistic_lr.pkl', 'rb')
plr = pickle.load(pkl_file)

# 预测
Y_predict = plr.predict(X_test)
