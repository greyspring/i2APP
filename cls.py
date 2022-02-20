#-*- coding: UTF-8 -*-
import warnings
import random
warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import sklearn
from sklearn.svm import SVC
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import  KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from evaluation import evaluate
from evaluation import evaluateLight

def knn(trainsdata,traintags,testsdata,testtags):
    model= KNeighborsClassifier()
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def logisticregression(trainsdata,traintags,testsdata,testtags):
    from sklearn import linear_model
    print("logisticregression")
    model = linear_model.LogisticRegression()
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)


def xgboost(trainsdata,traintags,testsdata,testtags,params):
    xgtrain = xgb.DMatrix(trainsdata, label=traintags)
    xgtest = xgb.DMatrix(testsdata)
#     params = {'booster': 'gbtree',
#               'objective': 'binary:logistic',
# #              'eval_metric': 'auc',
#               'gamma':0,
#               'random_state':50,
#               'learning rate':0.01,
#               'max_depth': 3,
#               'lambda': 10,
#               'subsample': 0.8,
#               'colsample_bytree': 0.8,
#               'min_child_weight': 1,
#               'eta': 0.025,
#               'seed': 27,
#               'nthread': -1,
#               'scale_pos_weight' :1,
#               'silent': 0}
#     watchlist = [(xgtrain, 'train')]
    #bst = xgb.train(params, xgtrain, num_boost_round=110, evals=watchlist)
    bst = xgb.train(params, xgtrain)
    # xgb = XGBClassifier()
    # xgb.fit(trainsdata,traintags)
    # 输出概率
    ypred = bst.predict(xgtest)
    # y_pred = xgb.predict(testsdata)
    # y_score = xgb.predict_proba(trainsdata)
    # print y_pred
    # print y_score
    y_score = ypred
    y_pred = (ypred >= 0.5) * 1
    return evaluate(testtags, y_pred,np.array(y_score))
    # 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
    #y_pred = (ypred >= 0.5) * 1
def bayes(trainsdata, traintags,testsdata,testtags):
    print("GaussianNB")
    by = GaussianNB()
    print("Training")
    by.fit(trainsdata,traintags)
    y_pred = by.predict(testsdata)
    y_score = by.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)


def decisiontree(trainsdata,traintags,testsdata,testtags):
    print("decisiontree")
    model = DecisionTreeClassifier()
    print("print Training ")
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def svm(trainsdata,traintags,testsdata,testtags):
    print ("Svm Classifier")
    from sklearn import svm
    model = svm.SVC(probability=True)
    print("Svm training")
    model.fit(trainsdata, traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)
##after svm opt para
def svmOpt(testsdata,testtags,model):
    print ("SvmOpt Classifier")
#    from sklearn import svm
#    model = svm.SVC(probability=True)
#    print("Svm training")
#    model.fit(trainsdata, traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    # print y_score
#    print y_pred
#    print y_score
#    y_score = y_pred
#    for i in range(len(y_pred)):
#        if y_pred[i]>0.5:y_pred[i]=1
#        else:y_pred[i]=0
    from evaluation import svmEvaluate
    return  svmEvaluate(testtags, y_pred,y_score)

def randomforest(trainsdata,traintags,testsdata,testtags):
    from sklearn.ensemble import RandomForestClassifier
    print ("RandomForestClassifier")
    rf0 = RandomForestClassifier(oob_score=True)
    print ("Training")
    rf0.fit(trainsdata,traintags)
    y_pred = rf0.predict(testsdata)
    y_score = rf0.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def adaboost(trainsdata,traintags,testsdata,testtags):
    from sklearn.ensemble import AdaBoostClassifier
    print("AdaBoostClassifier")
    clf = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)
    print("Training")
    clf.fit(trainsdata,traintags)
    y_pred = clf.predict( testsdata)
    y_score = clf.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def lightgbm(traindata,traintags,testdata,testtags):
    #print('123')
    train_data = lgb.Dataset(traindata, label=traintags, silent=True)
    # validation_data=lgb.Dataset(testdata,label=testtags)
    params = LGBMClassifier().get_params()
    params['verbosity']='-1'
    clf = lgb.train(params, train_data)

    train_label = clf.predict(traindata, num_iteration=clf.best_iteration)
    y_pred = clf.predict(testdata, num_iteration=clf.best_iteration)
    y_raw = clf.predict(testdata, raw_score=True, num_iteration=clf.best_iteration)
    # print y_raw
    #    y_score = y_pred
    #    print y_score
    for i in range(len(train_label)):
        if train_label[i] > 0.5:
            train_label[i] = 1
        else:
            train_label[i] = 0
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    return evaluateLight(testtags, y_pred, y_raw)



