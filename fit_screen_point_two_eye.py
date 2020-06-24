import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
# 导入线性模型和多项式特征构造模块
from sklearn.preprocessing import PolynomialFeatures
import pickle
import os
from sklearn import svm

list_dataFrame = []
base_path = "E:/0_gaze_with_ZSSR/3_Gaze_MPII/gaze_vector_dataset/"
list_path_src_data = os.listdir(base_path)
for path in list_path_src_data:
    list_dataFrame.append(pd.read_csv(os.path.join(base_path, path), sep='\t', header=None, index_col=False))

vector_dataFrame = pd.concat(list_dataFrame, axis=0)
# print(vector_dataFrame)
# vector_dataFrame = pd.read_csv("gaze_vector_new.txt", sep='\t', header=None, index_col=False)
#
x_dataFrame = vector_dataFrame.copy()
y_dataFrame = vector_dataFrame.copy()

x_dataFrame = x_dataFrame.drop([1], axis=1)
x_dataFrame.columns = range(17)
y_dataFrame = y_dataFrame.drop([0], axis=1)
y_dataFrame.columns = range(17)
print(x_dataFrame.head(3))
print(y_dataFrame.head(3))
#
x_df_train, x_df_test = train_test_split(x_dataFrame, test_size=0.2, shuffle=True)
y_df_train, y_df_test = train_test_split(y_dataFrame, test_size=0.2, shuffle=True)
#
# print(y_df_train.loc[:, 0:6])
predict_x = None
predict_y = None
train_for_x = True
train_for_y = True
if train_for_x:
    poly_reg_x = PolynomialFeatures(degree=1)
    X_ploy = poly_reg_x.fit_transform(x_df_train.loc[:, 1:])
    X_ploy_predict = poly_reg_x.fit_transform(x_df_test.loc[:, 1:])
    lin_reg_2_x = linear_model.LinearRegression()
    lin_reg_2_x.fit(X_ploy, x_df_train.loc[:, 0] / 1920)
    predict_x = lin_reg_2_x.predict(X_ploy_predict)
    output = open('linear_x.pkl', 'wb')
    pickle.dump(lin_reg_2_x, output, 0)  # 将训练后的线性模型保存
    output.close()

if train_for_y:
    poly_reg_y = PolynomialFeatures(degree=1)
    Y_ploy = poly_reg_y.fit_transform(y_df_train.loc[:, 1:])
    Y_ploy_predict = poly_reg_y.fit_transform(y_df_test.loc[:, 1:])
    lin_reg_2_y = linear_model.LinearRegression()
    # lin_reg_2 = svm.SVR(kernel='linear', C=1e3)
    lin_reg_2_y.fit(Y_ploy, y_df_train.loc[:, 0] / 1080)
    predict_y = lin_reg_2_y.predict(Y_ploy_predict)
    # output = open('svr_y.pkl', 'wb')
    output = open('linear_y.pkl', 'wb')
    pickle.dump(lin_reg_2_y, output, 0)  # 将训练后的线性模型保存
    output.close()
# #
# print('Cofficients:', [round(x, 4) for x in lin_reg_2.coef_])
# # 查看回归方程截距
# print('intercept', lin_reg_2.intercept_)

from sklearn import metrics

if train_for_x:
    print("X MSE:", metrics.mean_squared_error(x_df_test.loc[:, 0] / 1920, predict_x))
if train_for_y:
    print("Y MSE:", metrics.mean_squared_error(y_df_test.loc[:, 0] / 1080, predict_y))

# pkl_file = open('linear_x.pkl', 'rb')
# plr = pickle.load(pkl_file)
#
# # 预测
# poly_reg = PolynomialFeatures(degree=1)
# ploy = poly_reg.fit_transform(x_df_test.loc[0, 1:])
# Y_predict = plr.predict(ploy)
