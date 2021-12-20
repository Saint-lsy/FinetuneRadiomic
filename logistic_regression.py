import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

train_data = pd.read_excel('train_feat2.xls')
Y_train = train_data['train_label']
X_train = train_data.iloc[:,2:10]

val_data = pd.read_excel('val_feat2.xls')
Y_val = val_data['val_label']
X_val = val_data.iloc[:,2:10]

test_data = pd.read_excel('test_feat2.xls')
Y_test = test_data['test_label']
X_test = test_data.iloc[:,2:10]



logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,Y_train)

'''
val
'''
Y_pred=logistic_regression.predict(X_val)

print('Accuracy: ',metrics.accuracy_score(Y_val, Y_pred))

fpr,tpr,threshold = metrics.roc_curve(Y_val,Y_pred)
auc = metrics.auc(fpr, tpr)
print(auc)
print(metrics.roc_auc_score(Y_val, Y_pred))


'''
test
'''
Y_pred=logistic_regression.predict(X_test)

print('Accuracy: ',metrics.accuracy_score(Y_test, Y_pred))

fpr,tpr,threshold = metrics.roc_curve(Y_test,Y_pred)
auc = metrics.auc(fpr, tpr)
print(auc)
print(metrics.roc_auc_score(Y_test, Y_pred))
