#-*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
def drawaccauc(x,yyacc,yyauc,t= 'Independent test',name='drawaccauc'):
    # parameter
    # x:list  feature num
    # yy: list  acc
    # t: tile
    # name :png name
    maxacc = max(yyacc)
    maxauc = max(yyauc)

    indexacc = yyacc.index(maxacc)

    indexauc = yyauc.index(maxauc)
    maxaccfea = x[indexacc]
    maxaucfea = x[indexauc]

    print (maxaccfea)
    print (indexauc)
    print (maxaucfea)
    print (maxauc)
    print ("max_acc:",maxacc,"feanum:",indexacc)
    print ("max_auc:",maxauc,"feanum",indexauc)

    # 返回最大值
    plt.annotate(' max auc'+"("+str(maxaucfea)+","+str(maxauc)+")", xy=(maxaucfea, maxauc), xytext=(maxaucfea, maxauc),arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(' max acc'+"("+str(maxaccfea)+","+str(maxacc)+")", xy=(maxaccfea, maxacc), xytext=(maxaccfea, maxacc),arrowprops=dict(facecolor='black', shrink=0.05))

    plt.plot(x,yyacc)
    plt.plot(x,yyauc)

    plt.xlabel('Feature quantity ')
    plt.ylabel('Score')
    t1 = 'Independent test'
    t2 = 'Cross Validation'
    if(t==t2):
        plt.title(t2)
    else:
        plt.title(t1)
    # plt.title('ROC curve')

    #plt.legend(data.keys(),loc='upper right')
    plt.legend(["Acc","Auc"])
    # plt.savefig("testselectfrommodel2.png")
    plt.savefig("roc/"+name+".png")
    plt.show()

def drawfeature_acc(x,yy,t= 'Independent test',name='drawfeature_acc'):
    # parameter
    # x:list  feature num
    # yy: list  acc

    maxacc = max(yy)
    index = yy.index(maxacc)
    maxaccfea = x[index] # 返回最大值
    plt.annotate(' max acc'+"("+str(maxaccfea)+","+str(maxacc)+")", xy=(maxaccfea, maxacc), xytext=(maxaccfea, maxacc),arrowprops=dict(facecolor='black', shrink=0.05))
    plt.plot(x,yy)
    plt.xlabel('Feature quantity ')
    plt.ylabel('Accuracy_score')
    t1 = 'Independent test'
    t2 = 'Cross Validation'
    if(t==t2):
        plt.title(t2)
    else:
        plt.title(t1)
    plt.legend()
    plt.savefig("roc/"+name+".png")
    # plt.savefig("testselectfrommodel2.png")
    #plt.show()
def drawfeature_auc(x,yy,t= 'Independent test',name='drawfeature_auc'):
    # parameter
    # x:list  feature num
    # yy: list  acc
    maxauc = max(yy)
    index = yy.index(maxauc)
    maxaucfea = x[index] # 返回最大值
    plt.annotate(' max auc'+"("+str(maxaucfea)+","+str(maxauc)+")", xy=(maxaucfea, maxauc), xytext=(maxaucfea, maxauc),arrowprops=dict(facecolor='black', shrink=0.05))
    plt.plot(x,yy)
    plt.xlabel('Feature quantity ')
    plt.ylabel('Auc')
    t1 = 'Independent test'
    t2 = 'Cross Validation'
    if(t==t2):
        plt.title(t2)
    else:
        plt.title(t1)
    # plt.title('ROC curve')
    #plt.legend(data.keys(),loc='upper right')
    plt.legend()
    # plt.savefig("testselectfrommodel2.png")
    plt.savefig("roc/"+name+".png")
    # plt.show()
def rocauc(data,t= 'Independent test',name='rocauc'):
    #print data
    # data = dict(sorted(data.items(), key=lambda x: x[0]))
    # values = data.values()

    plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--')
    #label='Keras (area = {:.3f})'.format(auc)
    for i in data:
        name = name + i
        print (data[i]['auc'])
        plt.plot(data[i]['rocs'][0], data[i]['rocs'][1],"--",label=(i+' (auc = %0.2f)') % data[i]['auc'])
    t1 = 'Independent test'
    t2 = 'Cross Validation'
    if(t==t2):
        plt.title(t2)
    else:
        plt.title(t1)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    #plt.legend(data.keys(),loc='upper right')
    plt.legend()

    plt.savefig("roc/"+name+".png")
    # plt.show()
def crossrocauc(data):
    pass

def draw(x,meandic,picname):
    # yyacc = []
    # yyauc = []
    zzacc =[]
    zzauc=[]
    pp = []
    metricinde = []
    metriccv = []
    for i in range(len(x)):
        # yyauc.append(it[i]['auc'])
        # yyacc.append(it[i]['acc'])
        zzacc.append(meandic[i]['acc'])
        zzauc.append(meandic[i]['auc'])
    #drawaccauc(x, yyacc, yyauc, t='Independent test', name=picname_in)
    drawaccauc(x, zzacc, zzauc, t='Cross Validation', name=picname)