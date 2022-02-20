#-*- coding: UTF-8 -*-
import numpy as np
from sklearn import metrics
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
import warnings

def svmEvaluate(y_true,y_pred,y_score=[]):
    dic = {}
    warnings.filterwarnings("ignore")
    dic["acc"] = accuracy_score(y_true, y_pred, normalize=True)
    cm = confusion_matrix(y_true, y_pred)
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
    dic["auc"] = roc_auc_score(y_true,y_score[:,1])
    PRC = []
    PRC.append(recall2)
    PRC.append(precision2)
    aupr = auc( recall2,precision2)
    dic['aupr'] = aupr
    rocs = []
    rocs.append(fpr)
    rocs.append(tpr)
    dic["rocs"] = rocs
    dic["prc"] = PRC
    return dic

def evaluate(y_true,y_pred,y_score=[]):
    dic = {}
    warnings.filterwarnings("ignore")
    dic["acc"] = accuracy_score(y_true, y_pred, normalize=True)
    cm = confusion_matrix(y_true, y_pred)
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
    dic['sen'] = recall_score(y_true,y_pred)
    dic['sen'] = float(float(TP)/(float(TP+FN)))
    dic['f1_score']=f1_score(y_true, y_pred)
    dic['mcc']=matthews_corrcoef(y_true,y_pred)
    if(y_score.ndim==2):      
        fpr, tpr, thresholds = roc_curve(y_true,y_score[:,1]) # 返回预测为1的概率
        
        precision2, recall2, thresholds2 = precision_recall_curve(y_true,y_score[:,1])
    
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        precision2, recall2, thresholds2 = precision_recall_curve(y_true,y_score)

    roc_auc = auc(fpr,tpr)
    dic["auc"] = roc_auc_score(y_true,y_score[:,1])
    rocs = []
    PRC = []
    PRC.append(recall2)
    PRC.append(precision2)
    rocs.append(fpr)    
    rocs.append(tpr)
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
