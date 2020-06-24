import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
# 导入线性模型和多项式特征构造模块
from sklearn.preprocessing import PolynomialFeatures
import pickle



vector_dataFrame = pd.read_csv("gaze_vector_new.txt", sep='\t', header=None, index_col=False)

x_dataFrame = vector_dataFrame.drop([1], axis=1)
x_dataFrame.columns = range(9)
y_dataFrame = vector_dataFrame.drop([0], axis=1)
y_dataFrame.columns = range(9)
# print(x_dataFrame.head(3))
print(y_dataFrame.head(3))

x_df_train, x_df_test = train_test_split(x_dataFrame, test_size=0.2, shuffle=True)
y_df_train, y_df_test = train_test_split(y_dataFrame, test_size=0.2, shuffle=True)

print(y_df_train.loc[:, 0:6])
poly_reg = PolynomialFeatures(degree=1)
X_ploy = poly_reg.fit_transform(x_df_train.loc[:, 1:])
X_ploy_predict = poly_reg.fit_transform(x_df_test.loc[:, 1:])
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_ploy, x_df_train.loc[:, 0] / 1920)
predict = lin_reg_2.predict(X_ploy_predict)
#
print('Cofficients:', [round(x, 4) for x in lin_reg_2.coef_])
# 查看回归方程截距
print('intercept', lin_reg_2.intercept_)

from sklearn import metrics

print("MSE:", metrics.mean_squared_error(x_df_test.loc[:, 0] / 1920, predict))
