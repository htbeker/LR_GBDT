import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

path = './ml/'
df_train = 'train.csv'
df_test = 'test.csv'
data_train = pd.read_csv(path + df_train)
data_test = pd.read_csv(path + df_test)

""" 

根据kaggle对数据的描述：features分为包含‘ind’，‘reg’，‘car’，‘calc’等几类，其中‘bin’类为二分类特征，
‘cat’类为多分类特征， 不属于这两类为连续型特征。-1表示空值。对于分类特征进行one-hot处理，对于连续特征进行归一化处理。
train和test数据处理应一致，因此首先将train和test数据 集合并。

"""
#该数据集正负样本极不平衡需采样
pos_data = data_train[data_train.target ==1]
neg_data = data_train[data_train.target == 0].sample(21000,random_state = 666)
data_train = pd.concat([pos_data,neg_data]).sample(frac = 1)
data_test = data_test.sample(n = 40000)
train_sz = data_train.shape[0]
data_all = pd.concat([data_train,data_test])

#将三类特征取出，分别处理
bin_features = [ ]
cat_features = [ ]
other_features = [ ]
for i in data_all.columns.tolist():
    if 'bin' in i:
        bin_features.append(i)
    elif 'cat' in i:
        cat_features.append(i)
    elif 'id' != i and 'target' != i:
        other_features.append(i)
# print bin_features
# print cat_features
# print other_features

#将分类特征one-hot
df_one_hot = pd.DataFrame()
for col in bin_features+cat_features:
    df_one_hot = pd.concat([df_one_hot,pd.get_dummies(data_all[col],prefix= col)],axis = 1)
    
#将连续特征归一化
from sklearn.preprocessing import MinMaxScaler
for col in other_features:
    scaler = MinMaxScaler().fit(np.array(data_train[col]).reshape(-1,1))
    data_train[col] = scaler.transform(np.array(data_train[col]).reshape(-1,1))
data_train_all = pd.concat([data_train,df_one_hot[:train_sz]],axis = 1)

#LR
train_data,test_data,train_target,test_target = train_test_split(data_train_all.drop('target',axis = 1),data_train['target'])
lr = LogisticRegression(penalty= 'l1')
lr.fit(train_data,train_target)

train_predprob = lr.predict_proba(train_data)[:,1]
test_predprob = lr.predict_proba(test_data)[:,1]
fpr,tpr,thresholds = metrics.roc_curve(test_target.values,test_predprob)
print("AUC Score (Test): %.4f" % metrics.roc_auc_score(test_target, test_predprob))

#gbdt
train_data,test_data,train_target,test_target = train_test_split(data_train.drop('target',axis = 1),data_train['target'])
gbm = lgb.sklearn.LGBMClassifier(boosting_type = 'gbdt',learning_rate = 0.05,max_depth = 4,min_child_samples = 20,
                                 n_estimators= 100,num_leaves= 10)
gbm.fit(train_data,train_target)

train_predprob = gbm.predict_proba(train_data)[:,1]
test_predprob = gbm.predict_proba(test_data)[:,1]
fpr,tpr,thresholds = metrics.roc_curve(test_target.values,test_predprob)
print("AUC Score (Test): %.4f" % metrics.roc_auc_score(test_target, test_predprob))

#使用apply方法获得叶子节点编号
train_leaves = gbm.apply(data_train.drop('target',axis = 1))
train_leaves.shape
gbm.apply(data_train)[0]
n_estimators= 100
num_leaves= 10
#转换后训练集大小应为446409*（100*10）

#数据转换
transform_train_data = np.zeros([len(train_leaves),n_estimators*num_leaves],dtype = np.int32)
for i in np.arange(len(train_leaves)):
    temp = np.arange(n_estimators)*num_leaves+train_leaves[i]
    transform_train_data[i][temp] +=1
train_data,test_data,train_target,test_target = train_test_split(transform_train_data,data_train.target,random_state = 666)
lr = LogisticRegression(penalty= 'l1')
lr.fit(train_data,train_target)
train_predprob = lr.predict_proba(train_data)[:,1]
test_predprob = lr.predict_proba(test_data)[:,1]
fpr,tpr,thresholds = metrics.roc_curve(test_target.values,test_predprob)
print("AUC Score (Test): %.4f" % metrics.roc_auc_score(test_target, test_predprob))

#测试集做相同转换后可计算出类别概率
train_leaves = gbm.apply(data_test)
transform_train_test = np.zeros([len(train_leaves),n_estimators*num_leaves],dtype = np.int32)
for i in np.arange(len(train_leaves)):
    temp = np.arange(n_estimators)*num_leaves+train_leaves[i]
    transform_train_test[i][temp] +=1
lr.predict_proba(transform_train_test)
