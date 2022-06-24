import numpy as np
import matplotlib.pyplot as plt
import csv
import io
import os
import pandas as pd
import shutil
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


dataset = read_csv('data/labeled/data.csv')

# 分离数据集
array = dataset.values
X = array[:, 2:7]
Y = array[:, 1]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

## 算法审查
# models = {}
# models['LR'] = LogisticRegression(max_iter=10000)
# models['LDA'] = LinearDiscriminantAnalysis()
# models['KNN'] = KNeighborsClassifier()
# models['CART'] = DecisionTreeClassifier()
# models['NB'] = GaussianNB()
# models['SVM'] = SVC()
#
# # 评估算法
# results = []
# cv_scores = []
# for key in models:
#     kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
#     cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     cv_scores.append(cv_results.mean())
#     print('%s; %f (%f)' % (key, cv_results.mean(), cv_results.std()))
#
# models_name = ['LR', 'LDA', 'KNN', 'CART', 'NB', 'SVM']
# plt.plot(models_name, cv_scores)
# plt.xlabel('K')
# plt.ylabel('Accuracy')  # 通过图像选择最好的参数
# plt.show()
#
# # 箱线图比较算法
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(models.keys())
# pyplot.show()

# 使用评估数据集评估算法
from sklearn.metrics import accuracy_score

knn = DecisionTreeClassifier()
knn.fit(X=X_train, y=Y_train)
predictions = knn.predict(X_validation)

for i in range(len(predictions)):
    if predictions[i] != Y_validation[i]:
        for j in range(len(array)):
            if array[j, 2] == X_validation[i, 0] and array[j, 3] == X_validation[i, 1] \
                    and array[j, 4] == X_validation[i, 2] and array[j, 5] == X_validation[i, 3] and \
                    array[j, 6] == X_validation[i, 4]:
                img_path = array[j, 0]
                print(img_path, predictions[i], Y_validation[i])
                shutil.copy('data/' + img_path, 'data/false/' + predictions[i] + '-' + Y_validation[i] + '.png')

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
