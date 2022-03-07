# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
from lightgbm import LGBMClassifier
import numpy as np
from cls import randomforest, bayes, logisticregression, decisiontree, adaboost, knn
from cls import lightgbm
from cls import svmOpt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.model_selection import KFold
# from minepy import MINE
def dataprocessing(filepath):
    # print ("Loading feature files")
    dataset1 = pd.read_csv(filepath[0], header=None, low_memory=False)
    dataset2 = pd.read_csv(filepath[1], header=None, low_memory=False)
    dataset3 = pd.read_csv(filepath[2], header=None, low_memory=False)
    dataset4 = pd.read_csv(filepath[3], header=None, low_memory=False)

    dataset1 = dataset1.apply(pd.to_numeric, errors="ignore")
    dataset2 = dataset2.apply(pd.to_numeric, errors="ignore")
    dataset3 = dataset3.apply(pd.to_numeric, errors="ignore")
    dataset4 = dataset4.apply(pd.to_numeric, errors="ignore")
    dataset1.dropna(inplace=True)
    dataset2.dropna(inplace=True)
    dataset3.dropna(inplace=True)
    dataset4.dropna(inplace=True)

    traindata = pd.concat([dataset2, dataset4], axis=0).values
    testdata = pd.concat([dataset1, dataset3]).values

    smo = RandomUnderSampler(random_state=42)

    negtraintags = [0] * dataset2.shape[0]
    postraintags = [1] * dataset4.shape[0]
    traintags = negtraintags + postraintags
    # testdata = pd.concat([dataset1, dataset3])
    traindata, traintags = smo.fit_resample(traindata, traintags)
    negtesttags = [0] * dataset1.shape[0]
    postesttags = [1] * dataset3.shape[0]
    testtags = negtesttags + postesttags

    traintags = np.array(traintags)
    data = [traindata, traintags, testdata, testtags]
    return data


def datadic(filegroup):
    method = ["-DT.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-PSSM-DT.csv", "-CC-PSSM.csv",
              "-AC-PSSM.csv", "ACC-PSSM.csv", "kmer", "feature-AC.csv", "ACC.csv", "feature-CC.csv", "DP.csv", "DR.csv",
              "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "SC-PseAAC.csv", "SC-PseAAC-General.csv",
              'backward_3_Acthely_factors.csv',
              'backward_3_AESNN3.csv',
              'backward_3_ANN4D.csv',
              'backward_3_Binary_5_bit.csv',
              'backward_3_BLOSUM62.csv',
              'backward_3_Hydrophobicity_matrix.csv',
              'backward_3_Meiler_parameters.csv',
              'backward_3_Micheletti_potentials.csv',
              'backward_3_Miyazawa_energies.csv',
              'backward_3_One_hot.csv',
              'backward_3_One_hot_6_bit.csv',
              'backward_3_PAM250.csv',
              'backward_5_Acthely_factors.csv',
              'backward_5_AESNN3.csv',
              'backward_5_ANN4D.csv',
              'backward_5_Binary_5_bit.csv',
              'backward_5_BLOSUM62.csv',
              'backward_5_Hydrophobicity_matrix.csv',
              'backward_5_Meiler_parameters.csv',
              'backward_5_Micheletti_potentials.csv',
              'backward_5_Miyazawa_energies.csv',
              'backward_5_One_hot.csv',
              'backward_5_One_hot_6_bit.csv',
              'backward_5_PAM250.csv',
              'forward_3_Acthely_factors.csv',
              'forward_3_AESNN3.csv',
              'forward_3_ANN4D.csv',
              'forward_3_Binary_5_bit.csv',
              'forward_3_BLOSUM62.csv',
              'forward_3_Hydrophobicity_matrix.csv',
              'forward_3_Meiler_parameters.csv',
              'forward_3_Micheletti_potentials.csv',
              'forward_3_Miyazawa_energies.csv',
              'forward_3_One_hot.csv',
              'forward_3_One_hot_6_bit.csv',
              'forward_3_PAM250.csv',
              'forward_5_Acthely_factors.csv',
              'forward_5_AESNN3.csv',
              'forward_5_ANN4D.csv',
              'forward_5_Binary_5_bit.csv',
              'forward_5_BLOSUM62.csv',
              'forward_5_Hydrophobicity_matrix.csv',
              'forward_5_Meiler_parameters.csv',
              'forward_5_Micheletti_potentials.csv',
              'forward_5_Miyazawa_energies.csv',
              'forward_5_One_hot.csv',
              'forward_5_One_hot_6_bit.csv',
              'forward_5_PAM250.csv']
    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    postest = filegroup["postest"]
    negtest = filegroup["negtest"]
    file_method = {}
    filepath = []
    for methodname in method:
        for i in postrain:
            if methodname in i:
                postrain_method = i
                break

        for j in negtrain:
            if methodname in j:
                negtrain_method = j
                break

        for k in postest:
            if methodname in k:
                postest_method = k
                break

        for l in negtest:
            if methodname in l:
                negtest_method = l
                break
        filepath = [negtest_method, negtrain_method, postest_method, postrain_method]
        file_method[methodname] = dataprocessing(filepath)
    return file_method


def trainmodel_mutipleModel_SVM(datadic):
    train_feature = {}
    test_feature = {}
    metrics = {}
    index = []
    for i in datadic:
        data = datadic[i]
        print(i)
        (y_pred_train, y_pred_test) = trainmodel_multipleTags_SVM(data[0], data[1], data[2], data[3])
        feature_svm_01 = i + "_01_" + "SVM"
        feature_svm_rate = i + "_rate_" + "SVM"

        train_feature[feature_svm_01] = y_pred_train[0]
        train_feature[feature_svm_rate] = y_pred_train[1]

        test_feature[feature_svm_01] = y_pred_test[0]
        test_feature[feature_svm_rate] = y_pred_test[1]

    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
    print('ok')

    return data


def trainmodel_multipleTags_SVM(traindata, traintags, testdata, testtags):
    data = [traindata, traintags, testdata, testtags]
    c, g = SVMpara(data)
    model_SVC = SVC(probability=True, C=c, gamma=g)
    # model_SVC = SVC(probability=True)
    model_SVC.fit(data[0], data[1])

    traindata_pred_rate = model_SVC.predict_proba(traindata)[:, 1]
    traindata_pred_01 = model_SVC.predict(traindata)
    testdata_pred_rate = model_SVC.predict_proba(testdata)[:, 1]
    testdata_pred_01 = model_SVC.predict(testdata)

    print(accuracy_score(testtags, testdata_pred_01))

    trainLableList = [traindata_pred_01, traindata_pred_rate]
    y_predList = [testdata_pred_01, testdata_pred_rate]
    return trainLableList, y_predList


def trainmodel_mutipleModel_LGBM(datadic):
    train_feature = {}
    test_feature = {}
    metrics = {}
    index = []
    for i in datadic:
        data = datadic[i]
        print(i)
        (y_pred_train, y_pred_test) = trainmodel_multipleTags_LGBM(data[0], data[1], data[2], data[3])

        feature_Lgbm_01 = i + "_01_" + "Lgbm"
        feature_Lgbm_rate = i + "_rate_" + "Lgbm"

        train_feature[feature_Lgbm_01] = y_pred_train[0]
        train_feature[feature_Lgbm_rate] = y_pred_train[1]

        test_feature[feature_Lgbm_01] = y_pred_test[0]
        test_feature[feature_Lgbm_rate] = y_pred_test[1]

    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
    print('ok')

    return data

def  trainmodel_multipleTags_LGBM(traindata,traintags,testdata,testtags):
    train_data = lgb.Dataset(traindata, label=traintags, silent=True)
    # validation_data=lgb.Dataset(testdata,label=testtags)
    params = LGBMClassifier().get_params()
    clf = lgb.train(params, train_data)
    train_lable_Lgbm_rate = clf.predict(traindata, num_iteration=clf.best_iteration)
    train_lable_Lgbm_01 = train_lable_Lgbm_rate.copy()
    y_pred_Lgbm_rate = clf.predict(testdata, num_iteration=clf.best_iteration)
    y_pred_Lgbm_01 = y_pred_Lgbm_rate.copy()
    # y_raw = clf.predict(testdata, raw_score=True, num_iteration=clf.best_iteration)
    for i in range(len(train_lable_Lgbm_01)):
        if train_lable_Lgbm_01[i] > 0.5:
            train_lable_Lgbm_01[i] = 1
        else:
            train_lable_Lgbm_01[i] = 0
    for i in range(len(y_pred_Lgbm_01)):
        if y_pred_Lgbm_01[i] > 0.5:
            y_pred_Lgbm_01[i] = 1
        else:
            y_pred_Lgbm_01[i] = 0

    trainLableList = [train_lable_Lgbm_01, train_lable_Lgbm_rate]
    y_predList = [y_pred_Lgbm_01, y_pred_Lgbm_rate]
    return trainLableList, y_predList

def find_best_SVM(c,gamma,traindata,trainlabel,testdata,testlabel):
    model = SVC(C=c, gamma=gamma, probability=True)
    model.fit(traindata,trainlabel)
    prediction=model.predict(testdata)
    acc=accuracy_score(testlabel, prediction)
    return acc
def SVMpara(data):
    best_par_acc=0
    c = [ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000 ]
    gamma = [ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

    for C in c:
      for Gamma in gamma:
        accuracy=find_best_SVM(C,Gamma,data[0],data[1], data[2], data[3])
        # print(C," ",Gamma," ",accuracy)
        if(accuracy>best_par_acc):
           best_par_acc=accuracy
           best_C=C
           best_gamma=Gamma
    print("the highest acc is ",best_par_acc)
    print("the best C is ",best_C)
    print("the best gamma is ",best_gamma)
    return best_C,best_gamma

def main():
    tmpdir = "APP_feature"
    postrain = glob.glob(tmpdir + '/Training-P.fastafeature*')
    negtrain = glob.glob(tmpdir + '/Training-N.fastafeature*')
    postest = glob.glob(tmpdir + '/Test-P.fastafeature*')
    negtest = glob.glob(tmpdir + '/Test-N.fastafeature*')
    tmpdir2 = "cAPP"
    postrain2 = glob.glob(tmpdir2 + '/Training-P.txt*')
    negtrain2 = glob.glob(tmpdir2 + '/Training-N.txt*')
    postest2 = glob.glob(tmpdir2 + '/Test-P.txt*')
    negtest2 = glob.glob(tmpdir2 + '/Test-N.txt*')
    postrain.extend(postrain2)
    negtrain.extend(negtrain2)
    postest.extend(postest2)
    negtest.extend(negtest2)
    filegroup = {}  # 字典
    filegroup['postrain'] = postrain
    filegroup['negtrain'] = negtrain
    filegroup['postest'] = postest
    filegroup['negtest'] = negtest

    # datadics = datadic(filegroup)
    # np.save('APP.npy',datadics)
    datadics = np.load('APP.npy', allow_pickle=True).item()
    # kf = KFold(n_splits=10,shuffle=True)
    # train_all = datadics['-AC-PSSM.csv'][0]
    # label_all = datadics['-AC-PSSM.csv'][1]
    # data_new={}
    # for train_index, test_index in kf.split(train_all):
    #     filepath = './ten-cross/index'
    #     filename = filepath+str(i)+'train_index'
    #     filename2 = filepath+str(i)+'test_index'
    #     np.save(filename+'.npy',train_index)
    #     np.save(filename2+'.npy',test_index)
    #     i=i+1
    # for ss in range(1,11):
    #     filepath = './ten-cross/index/index'
    #     filename = filepath+str(ss)+'train_index'
    #     filename2 = filepath+str(ss)+'test_index'
    #     train_index = np.load(filename+'.npy', allow_pickle=True)
    #     test_index = np.load(filename2+'.npy', allow_pickle=True)
    #     for i in datadics:
    #         data_t = datadics[i]
    #         # print datadics[i][0].shape
    #         X_train, X_test = data_t[0][train_index],data_t[0][test_index]
    #         y_train, y_test = data_t[1][train_index],data_t[1][test_index]
    #         a=[X_train,y_train, X_test, y_test]
    #         data_new[i]=a
    #     # data1 = trainmodel_mutipleModel_LGBM(data_new)
    #     data2 = trainmodel_mutipleModel_SVM(data_new)
    #     # filepath = './ten-cross/lgbm/'
    #     # filename = filepath+str(ss)+'_lgbm.npy'
    #     filepath = './ten-cross/svm/'
    #     filename = filepath+str(ss)+'_svm.npy'
    #     np.save(filename,data2)
    # for ss in range(1,11):
    #     filepath = './ten-cross/lgbm/'
    #     filename = filepath+str(ss)+'_lgbm.npy'
    #     filepath2 = './ten-cross/svm/'
    #     filename2 = filepath2+str(ss)+'_svm.npy'

    #     data1 = np.load(filename,allow_pickle=True)
    #     data2 = np.load(filename2,allow_pickle=True)
    #     data3 = np.load(filename,allow_pickle=True)
    #     data3[0] = np.concatenate((data1[0],data2[0]),axis=1)
    #     data3[2] = np.concatenate((data1[2],data2[2]),axis=1)
    #     a, b = data3[0].shape
    #     list = []
    #     mine = MINE(alpha=0.6, c=15)
    #     for i in range(b):
    #         mine.compute_score(data3[0][:,i],data3[1])
    #         list.append(mine.mic())
    #     filepath = './ten-cross/mic/'
    #     filename = filepath+str(ss)+'_mic.npy'
    #     np.save(filename,list)

    # ten-cross
    for ss in range(1, 11):
        filepath = './ten-cross/lgbm/'
        filename = filepath + str(ss) + '_lgbm.npy'
        filepath2 = './ten-cross/svm/'
        filename2 = filepath2 + str(ss) + '_svm.npy'
        filepath3 = './ten-cross/mic/'
        filename3 = filepath3 + str(ss) + '_mic.npy'
        data1 = np.load(filename, allow_pickle=True)
        data2 = np.load(filename2, allow_pickle=True)
        data3 = np.load(filename, allow_pickle=True)
        data3[0] = np.concatenate((data1[0], data2[0]), axis=1)
        data3[2] = np.concatenate((data1[2], data2[2]), axis=1)
        mic = np.load(filename3, allow_pickle=True)
        list = []
        for i in range(len(mic)):
            if mic[i] >= 0.4:
                list.append(i)
        data = [data3[0][:, list], data3[1], data3[2][:, list], data3[3]]
        c, g = SVMpara(data)
        model_SVC = SVC(probability=True, C=c, gamma=g)
        model_SVC.fit(data[0], data[1])
        indemetric = svmOpt(data[2], data[3], model_SVC)
        random_inde = randomforest(data[0], data[1], data[2], data[3])
        bayes_inde = bayes(data[0], data[1], data[2], data[3])
        logisticregression_inde = logisticregression(data[0], data[1], data[2], data[3])
        adaboost_inde = adaboost(data[0], data[1], data[2], data[3])
        knn_inde = knn(data[0], data[1], data[2], data[3])
        lightgbm_inde = lightgbm(data[0], data[1], data[2], data[3])
        decisiontree_inde = decisiontree(data[0], data[1], data[2], data[3])
        metric = pd.DataFrame(indemetric)
        random_inde = pd.DataFrame(random_inde)
        bayes_inde = pd.DataFrame(bayes_inde)
        logisticregression_inde = pd.DataFrame(logisticregression_inde)
        adaboost_inde = pd.DataFrame(adaboost_inde)
        knn_inde = pd.DataFrame(knn_inde)
        decisiontree_inde = pd.DataFrame(decisiontree_inde)
        lightgbm_inde = pd.DataFrame(lightgbm_inde, index=[0])
        col = ['acc', 'auc', 'sen', 'spec', 'mcc', 'f1_score']
        piece = metric.loc[0, col]
        random_inde_piece = random_inde.loc[0, col]
        bayes_inde_piece = bayes_inde.loc[0, col]
        logisticregression_inde_piece = logisticregression_inde.loc[0, col]
        adaboost_inde_piece = adaboost_inde.loc[0, col]
        knn_inde_piece = knn_inde.loc[0, col]
        decisiontree_inde_piece = decisiontree_inde.loc[0, col]
        lightgbm_inde_piece = lightgbm_inde.loc[0, col]
        piece.name = 'Svm'
        random_inde_piece.name = 'Randomforest'
        bayes_inde_piece.name = 'Bayes'
        logisticregression_inde_piece.name = 'logisticregression'
        adaboost_inde_piece.name = 'Adaboost'
        knn_inde_piece.name = 'Knn'
        decisiontree_inde_piece.name = 'Decision_tree'
        lightgbm_inde_piece.name = "lightgbm"
        outCome = pd.concat(
            [piece, random_inde_piece, bayes_inde_piece, logisticregression_inde_piece, adaboost_inde_piece,
             knn_inde_piece, decisiontree_inde_piece, lightgbm_inde_piece], axis=1)
        filepath = './ten-cross/result/'
        filename = filepath + str(ss) + '_cross.csv'
        # outCome.to_csv(filename)
        print(outCome)

    #Independence

    # data1 = trainmodel_mutipleModel_LGBM(datadics)
    # data1 = trainmodel_mutipleModel_SVM(datadics)
    # np.save('2-lgbm-APP.npy',data1)
    # np.save('2-TOPsvm-APP.npy',data1)
    data1 = np.load('2-lgbm-APP.npy', allow_pickle=True)
    data2 = np.load('2-TOPsvm-APP.npy', allow_pickle=True)
    data3 = np.load('2-lgbm-APP.npy', allow_pickle=True)
    data3[0] = np.concatenate((data1[0], data2[0]), axis=1)
    data3[2] = np.concatenate((data1[2], data2[2]), axis=1)

    # a, b = data1[0].shape
    # list = []
    # mine = MINE(alpha=0.6, c=15)
    # for i in range(b):
    #     mine.compute_score(data1[0][:,i],data1[1])
    #     list.append(mine.mic())
    # matrix1 = np.array(list)

    mic = np.load('MICindex_LGBM_SVM.npy')

    list = []
    for i in range(len(mic)):
        if mic[i] >= 0.4:
            list.append(i)

    data = data3
    s2 = data3[0][:, list]
    s4 = data3[2][:, list]
    data = [s2, data1[1], s4, data1[3]]

    c, g = SVMpara(data)

    model_SVC = SVC(probability=True, C=c, gamma=g)
    model_SVC.fit(data[0], data[1])
    indemetric = svmOpt(data[2], data[3], model_SVC)
    random_inde = randomforest(data[0], data[1], data[2], data[3])
    bayes_inde = bayes(data[0], data[1], data[2], data[3])
    logisticregression_inde = logisticregression(data[0], data[1], data[2], data[3])
    adaboost_inde = adaboost(data[0], data[1], data[2], data[3])
    knn_inde = knn(data[0], data[1], data[2], data[3])
    lightgbm_inde = lightgbm(data[0], data[1], data[2], data[3])
    decisiontree_inde = decisiontree(data[0], data[1], data[2], data[3])
    metric = pd.DataFrame(indemetric)
    random_inde = pd.DataFrame(random_inde)
    bayes_inde = pd.DataFrame(bayes_inde)
    logisticregression_inde = pd.DataFrame(logisticregression_inde)
    adaboost_inde = pd.DataFrame(adaboost_inde)
    knn_inde = pd.DataFrame(knn_inde)
    decisiontree_inde = pd.DataFrame(decisiontree_inde)
    lightgbm_inde = pd.DataFrame(lightgbm_inde, index=[0])
    col = ['acc', 'auc', 'sen', 'spec', 'mcc', 'f1_score']
    piece = metric.loc[0, col]
    random_inde_piece = random_inde.loc[0, col]
    bayes_inde_piece = bayes_inde.loc[0, col]
    logisticregression_inde_piece = logisticregression_inde.loc[0, col]
    adaboost_inde_piece = adaboost_inde.loc[0, col]
    knn_inde_piece = knn_inde.loc[0, col]
    decisiontree_inde_piece = decisiontree_inde.loc[0, col]
    lightgbm_inde_piece = lightgbm_inde.loc[0, col]
    piece.name = 'Svm'
    random_inde_piece.name = 'Randomforest'
    bayes_inde_piece.name = 'Bayes'
    logisticregression_inde_piece.name = 'logisticregression'
    adaboost_inde_piece.name = 'Adaboost'
    knn_inde_piece.name = 'Knn'
    decisiontree_inde_piece.name = 'Decision_tree'
    lightgbm_inde_piece.name = "lightgbm"
    outCome = pd.concat(
        [piece, random_inde_piece, bayes_inde_piece, logisticregression_inde_piece, adaboost_inde_piece,
         knn_inde_piece, decisiontree_inde_piece, lightgbm_inde_piece], axis=1)
    # filename = "IndependentTest" + '.csv'
    # outCome.to_csv(filepath)
    print(outCome)

if __name__ == "__main__":
    main()