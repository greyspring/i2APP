#-*- coding: UTF-8 -*-
import numpy as np
from sklearn import metrics
from drawpic import rocauc
from sklearn.metrics import matthews_corrcoef  #MCC
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve



from sklearn.metrics import roc_auc_score,precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import hamming_loss

import warnings
import matplotlib.pyplot as plt

def svmEvaluate(y_true,y_pred,y_score=[]):
    dic = {}
    warnings.filterwarnings("ignore")
    # accuracy_score
    # print("accuracy_score "+str(accuracy_score(y_true, y_pred)))
    # return accuracy_score(y_true, y_pred, normalize=True)  # 类似海明距离，每个类别求准确后，再求微平均
    dic["acc"] = accuracy_score(y_true, y_pred, normalize=True)

    cm = confusion_matrix(y_true, y_pred)
    # dic['confusion'] = cm
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]
    dic['TP'] = TP
    dic['FP'] = FP
    dic['TN'] = TN
    dic['FN'] = FN
    if ((FP + TN) != 0):
        spec = float(float(TN) / (float(FP + TN)))
    else:
        spec = 'error'
    dic['spec'] = spec
    # sen  = float(float(TP)/float((TP+FN)))

    dic['sen'] = recall_score(y_true, y_pred)
    dic['f1_score'] = f1_score(y_true, y_pred)
    dic['mcc'] = matthews_corrcoef(y_true, y_pred)
    if (y_score.ndim == 2):
        fpr, tpr, thresholds = roc_curve(y_true, y_score[:,1])
        precision2, recall2, thresholds2 = precision_recall_curve(y_true,y_score[:,1])
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        precision2, recall2, thresholds2 = precision_recall_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)
    #dic["auc"] = roc_auc
    dic["auc"] = roc_auc_score(y_true,y_score[:,1])
    PRC = []
    PRC.append(recall2)
    PRC.append(precision2)
    aupr = auc( recall2,precision2)
    dic['aupr'] = aupr
    rocs = []

    rocs.append(fpr)
    rocs.append(tpr)
    # rocs.append(roc_auc)
    dic["rocs"] = rocs
    dic["prc"] = PRC
    return dic

def evaluate(y_true,y_pred,y_score=[]):
    dic = {}
    warnings.filterwarnings("ignore")
    # accuracy_score
    # print("accuracy_score "+str(accuracy_score(y_true, y_pred)))
    #return accuracy_score(y_true, y_pred, normalize=True)  # 类似海明距离，每个类别求准确后，再求微平均
    dic["acc"] = accuracy_score(y_true, y_pred, normalize=True)
    # 2, metrics
    # print("micro precision_score"+str(metrics.precision_score(y_true, y_pred, average='micro')))  # 微平均，精确率
    # print("macro precision_score"+str(metrics.precision_score(y_true, y_pred, average='macro')))  # 宏平均，精确率
    # # print(metrics.precision_score(y_true, y_pred, labels=[0, 1], average='macro'))  # 指定特定分类标签的精确率
    # # Out[133]: 0.5

    # #  *************召回率*************
    # print("micro recall_score "+str(metrics.recall_score(y_true, y_pred, average='micro')))
    # print("macro recall_score "+str(metrics.recall_score(y_true, y_pred, average='macro')))

    # #  *************F1*************
    # print("f1_score "+str(metrics.f1_score(y_true, y_pred, average='weighted')))

    # #  *************F2*************
    # # 根据公式计算

    # def calc_f2(label, predict):
    #     p = precision_score(label, predict)
    #     r = recall_score(label, predict)
    #     f2_score = 5 * p * r / (4 * p + r)
    #     return f2_score
    # f2_score = calc_f2(y_true,y_pred)
    # print ("f2_score "+str(f2_score))

    # #  *************混淆矩阵*************
    # print(confusion_matrix(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    #dic['confusion'] = cm
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]
    dic['TP'] = TP
    dic['FP'] = FP
    dic['TN'] = TN
    dic['FN'] = FN
    if ((FP+TN)!=0):
        spec = float(float(TN)/(float(FP+TN)))
    else:
        spec = 'error'
    dic['spec'] = spec
    # sen  = float(float(TP)/float((TP+FN)))
    dic['sen'] = recall_score(y_true,y_pred)
    dic['sen'] = float(float(TP)/(float(TP+FN)))
    dic['f1_score']=f1_score(y_true, y_pred)
    # print ("sen ",sen)
    # print ("spec ",spec)
    # print ("TP ",TP)
    # print ("FP ",FP)
    # print ("TN ",TN)
    # print ("FN ",FN)
    # MCC
    dic['mcc']=matthews_corrcoef(y_true,y_pred)
    # print ("*************ROC*************")
    # # 1，计算ROC值

   # 2，ROC曲线 参数
   #  print y_score.shape
   #  print type(y_score)
    if(y_score.ndim==2):      
        fpr, tpr, thresholds = roc_curve(y_true,y_score[:,1]) # 返回预测为1的概率
        
        precision2, recall2, thresholds2 = precision_recall_curve(y_true,y_score[:,1])
    
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        precision2, recall2, thresholds2 = precision_recall_curve(y_true,y_score)

    roc_auc = auc(fpr,tpr)
    #dic["auc"] = roc_auc
    dic["auc"] = roc_auc_score(y_true,y_score[:,1])
    rocs = []
    PRC = []
    PRC.append(recall2)
    PRC.append(precision2)
    rocs.append(fpr)       # 一个阈值 横坐标fpr 纵坐标tpr
    rocs.append(tpr)
    #rocs.append(roc_auc)
    aupr = auc( recall2,precision2)
    dic['aupr'] = aupr
    dic["rocs"] = rocs
    dic["prc"] = PRC
    return dic
def evaluateLight(y_true,y_pred,y_score):
    dic = {}
    warnings.filterwarnings("ignore")
    dic["acc"] = accuracy_score(y_true, y_pred, normalize=True)
    cm = confusion_matrix(y_true, y_pred)
    #dic['confusion'] = cm
    TP = cm[1][1]
    FP = cm[0][1]
    TN = cm[0][0]
    FN = cm[1][0]
    # dic['TP'] = TP
    # dic['FP'] = FP
    # dic['TN'] = TN
    # dic['FN'] = FN
    if ((FP + TN) != 0):
        spec = float(float(TN) / (float(FP + TN)))
    else:
        spec = 'error'

    dic['spec'] = spec
    dic['sen'] = recall_score(y_true, y_pred)
    dic['f1_score'] = f1_score(y_true, y_pred)
    dic['mcc'] = matthews_corrcoef(y_true, y_pred)
    dic["auc"] = roc_auc_score(y_true,y_score)

    return dic
