import numpy as np
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

def get_content_data(k):
    df = pd.read_csv("codewords/mfcc_hist_"+str(k)+".csv",header=0,sep=',')
    return df

def get_user_data(user_num):
    usr = int(user_num)
    ### MUSICAS TOCADAS
    df = pd.read_csv("codewords/user_play_counts.csv",header=None,sep=',')
    df.columns = ['user', 'song','play_count']
    usr_count = df.loc[df['user'] == usr]
    play_count = usr_count['song'].values
    played = list(play_count)
    ### MUSICAS ESCUTADAS
    df = pd.read_csv("codewords/user_listened_counts.csv",header=None,sep=',')
    df.columns = ['user', 'song','play_count']
    usr_count = df.loc[df['user'] == usr]
    listened_count = usr_count['song'].values
    listened_count = list(listened_count)
    # TODO
    # SELECIONAR QUEM OUVIU ACIMA DE UM VALOR E SALTOU ACIMA DE OUTRO VALOR
    listened = list(set(play_count).intersection(listened_count))
    skiped = list(set(play_count) - set(listened_count))
    return listened, skiped, played

def user_count():
    df = pd.read_csv("codewords/user_listened_counts.csv",header=None,sep=',')
    df.columns = ['user', 'song','play_count']
    users = set(df['user'])
    return users

def logistic_regression(train_x, train_y):
    from sklearn import linear_model
    from sklearn import metrics
    logistic_regression_model = linear_model.LogisticRegression(penalty='l1')
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model

#TODO
def cba(codeMatrix, bernMatrix, beta_value):
    codeMatrix = np.array(codeMatrix)
    bernMatrix = np.array(bernMatrix)
    beta = np.random.rand(codeMatrix.shape[1])
    diffArray = []
    diff = 10.0
    while diff > beta_value:
        j = 0
        w = 0
        k = 0
        h = []
        for j in range(codeMatrix.shape[0]):
            h_k = []
            if int(bernMatrix[j]) == 1:
                for k in range(codeMatrix.shape[1]):
                    soma = 0.0
                    for idx in range(codeMatrix.shape[1]):
                        soma = soma + (float(codeMatrix[j][idx])*beta[idx])
                    h_k.append(float(codeMatrix[j][k])*beta[k]/soma)
            else:
                for k in range(codeMatrix.shape[1]):
                    soma = 0.0
                    for idx in range(codeMatrix.shape[1]):
                        soma = soma + (float(codeMatrix[j][idx])*(1.0-beta[idx]))
                    h_k.append(float(codeMatrix[j][k])*(1.0-beta[k])/soma)
            h.append(h_k)
    
        somah = 0.0
        soma = 0.0
        betaN = np.zeros(shape=(len(codeMatrix[0])))
        for k in range(len(codeMatrix[0])):
            soma = 0.0
            somah = 0.0
            for j in range(len(h)):
                soma = soma + (bernMatrix[j]*h[j][k])
                somah = somah + h[j][k]
            betaN[k] = soma/somah
    
        diff = np.sqrt(np.sum(np.power(np.subtract(betaN,beta),2)))
        diffArray.append(diff)
        beta = betaN
    return beta

def cba_calc_probs(beta, test_x):
    test_x = np.array(test_x)
    prob = []
    for mfcc in test_x:
        soma = 0.0
        for j in range(len(mfcc)):
            soma = soma + (float(mfcc[j])*float(beta[j]))
        prob.append( soma/np.sum(mfcc.astype(float)))
    prob = np.array(prob)
    return prob 

def Find_Optimal_Cutoff_roc(target, predicted):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    #i = np.arange(len(tpr))
    #roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    #roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    #return list(roc_t['threshold'])
    return optimal_threshold

def calc_metrics(pred_y, y_test,pred_y_train,y_train):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0.005, .999), copy=False)
    thd = 0.5
    if len(pred_y_train) > 1: 
        scaler.fit_transform(np.array(pred_y_train).reshape(-1, 1))
        thd = Find_Optimal_Cutoff_roc(y_train,pred_y_train)

    y_pred = []
    for a in pred_y:
        if a >= thd: 
            y_pred.append(1)
        else:
            y_pred.append(0)
    #y_pred = np.around(y_pred)

    from sklearn.metrics import accuracy_score
    acc =  accuracy_score(y_test, y_pred)
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, fscore, an = precision_recall_fscore_support(y_test, y_pred,average='binary')
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_test, y_pred)
    from sklearn.metrics import roc_auc_score
    roc = 0.5
    if len(set(y_test)) > 1: roc = roc_auc_score(y_test, y_pred)
    return acc,precision,recall,fscore,ap,roc

def plot_diversity(div):
    plt.boxplot(div, 0, '')
    plt.xticks(range(1,len(div)+1), ['5','10','25','50','100','200'])
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":

    #LE ARQUIVOS DE EXECUCOES PARA DETECTAR QUANTOS USUARIOS EXISTEM NA BASE
    users = user_count()
#    print(str(len(users)) + " USUARIOS")
    min_lst = 100

    log_prec = []
    log_fm = []
    log_recall = []
    log_train_prec = []
    log_train_fm = []
    log_train_recall = []
    cba_prec = []
    cba_fm = []
    cba_recall = []
    cba_train_prec = []
    cba_train_fm = []
    cba_train_recall = []

    knum = [5,10,25,50,100,200]
    for k_num in knum:
        preds_acum_log = []
        preds_acum_log_train = []
        preds_acum_cba = []
        preds_acum_cba_train = []
        preds_acum_xgb = []
        preds_acum_xgb_train = []
        preds_acum_rand = []
        test_y_acum = []
        train_y_acum = []

        error_log = []
        error_cba = []

        #LE DADOS DE MFCC DAS MUSICAS
        mfcc = get_content_data(k_num)
        print(mfcc.shape)
    
        metrics_log = []
        metrics_log_train = []
        metrics_cba = []
        metrics_cba_train = []
    
        for r in range(20):
            #PARA CADA USUARIO SELECIONA MUSICAS TOCADAS E MUSICAS OUVIDAS
            for usr in users:
                #SELECIONA DENTRE AS MUSICAS TOCAS QUAIS FORAM OUVIDAS E QUAIS PULADAS
                lst,skp,pld = get_user_data(usr)
                #SELECIONAR QUEM OUVIU ACIMA DE UM VALOR E SALTOU ACIMA DE OUTRO VALOR
                if len(lst) < min_lst: continue

                #FILTRA A MATRIZ DE MFCC RESTANDO APENAS AS QUE FORAM TOCADAS
                mfcc_played = []
                mfcc_played = mfcc.loc[np.array(pld)-1]
                mfcc_played = mfcc_played.drop('ano',1)
                ids = mfcc_played['id'].copy()
                mfcc_played = mfcc_played.drop('id',1)
                #NORMALIZACAO COM A SOMA DE TODOS VALORES DO HISTOGRAMA
                #PARA COMPENSAR MUSICAS COM DURACOES DIFERENTES
                mfcc_played = mfcc_played.div(mfcc_played.sum(axis=1), axis=0)
                #mfcc_played = mfcc_played.div(mfcc_played.max(axis=1), axis=0)
                mfcc_played = mfcc_played.assign(play=np.zeros(len(pld)))
                #ALTERAR ISSO PARA ALGUMA REGRA DIRETA!!!!
                mfcc_lst = list(ids)
                for ls in lst:
                    if ls in mfcc_lst:
                        mfcc_played.at[ls-1, 'play'] = 1

                #EMBARALHA LINHAS
                mfcc_played = mfcc_played.sample(frac=1)
                #SEPARA EM DADOS DE TREINO E TESTE
                lmt = int(len(pld)*0.8)
                train_y = mfcc_played['play'][:lmt]
                test_y = mfcc_played['play'][lmt:]
                train_x = mfcc_played.drop('play',1)[:lmt]
                test_x = mfcc_played.drop('play',1)[lmt:]

                # REGRESSAO LOGISTICA
                #print("LOG REG")
                log_model = logistic_regression(train_x,train_y)
                preds_log = log_model.predict_proba(test_x)[:,1]
                preds_log_train = log_model.predict_proba(train_x)[:,1]
                acc,precision,recall,fscore,ap,aroc = calc_metrics(preds_log,test_y,preds_log_train,train_y)
                metrics_log.append([acc,precision,recall,fscore,ap,aroc])
                acc,precision,recall,fscore,ap,aroc = calc_metrics(preds_log_train,train_y,preds_log_train,train_y)
                metrics_log_train.append([acc,precision,recall,fscore,ap,aroc])
                #CBA
                #print("CBA")
                beta_value = float(k_num)/100
                beta_matrix = cba(train_x,train_y,beta_value)
                preds_cba = cba_calc_probs(beta_matrix, test_x)  
                preds_cba_train = cba_calc_probs(beta_matrix, train_x)  
                acc,precision,recall,fscore,ap,aroc = calc_metrics(preds_cba,test_y,preds_cba_train,train_y)
                metrics_cba.append([acc,precision,recall,fscore,ap,aroc])
                acc,precision,recall,fscore,ap,aroc = calc_metrics(preds_cba_train,train_y,preds_cba_train,train_y)
                metrics_cba_train.append([acc,precision,recall,fscore,ap,aroc])
   
        print(k_num)

        print('####### TEST ###########')
        metrics_log = np.array(metrics_log)
        print(  "REG: ",
                "%.3f" % round(np.mean(metrics_log[:,1]),3),
                "(", "%.3f" % round(np.std(metrics_log[:,1]),3),")",
                "%.3f" % round(np.mean(metrics_log[:,2]),3),
                "(","%.3f" % round(np.std(metrics_log[:,2]),3),")",
                "%.3f" % round(np.mean(metrics_log[:,3]),3),
                "(","%.3f" % round(np.std(metrics_log[:,3]),3),")",
                #round(2*((np.mean(metrics_log[:,1])*np.mean(metrics_log[:,2]))/(np.mean(metrics_log[:,1])+np.mean(metrics_log[:,2]))),3),
                "%.3f" % round(np.mean(metrics_log[:,4]),3),
                "(","%.3f" % round(np.std(metrics_log[:,4]),3),")",
                "%.3f" % round(np.mean(metrics_log[:,5]),3),
                "(","%.3f" % round(np.std(metrics_log[:,5]),3),")")

        metrics_cba = np.array(metrics_cba)
        print(  "CBA: ",
                "%.3f" % round(np.mean(metrics_cba[:,1]),3),
                "(","%.3f" % round(np.std(metrics_cba[:,1]),3),")",
                "%.3f" % round(np.mean(metrics_cba[:,2]),3),
                "(","%.3f" % round(np.std(metrics_cba[:,2]),3), ")",
                "%.3f" % round(np.mean(metrics_cba[:,3]),3),
                "(","%.3f" % round(np.std(metrics_cba[:,3]),3), ")",
                "%.3f" % round(np.mean(metrics_cba[:,4]),3),
                "(","%.3f" % round(np.std(metrics_cba[:,4]),3),")",
                #2*((np.mean(metrics_cba[:,1])*np.mean(metrics_cba[:,2]))/(np.mean(metrics_cba[:,1])+np.mean(metrics_cba[:,2]))),
                "%.3f" % round(np.mean(metrics_cba[:,5]),3),
                "(","%.3f" % round(np.std(metrics_cba[:,5]),3),")")

        print('####### TRAIN ###########')
        metrics_log_train = np.array(metrics_log_train)
        print(  "REG: ",
                "%.3f" % round(np.mean(metrics_log_train[:,1]),3),
                "(", "%.3f" % round(np.std(metrics_log_train[:,1]),3),")",
                "%.3f" % round(np.mean(metrics_log_train[:,2]),3),
                "(","%.3f" % round(np.std(metrics_log_train[:,2]),3),")",
                "%.3f" % round(np.mean(metrics_log_train[:,3]),3),
                "(","%.3f" % round(np.std(metrics_log_train[:,3]),3),")",
                #round(2*((np.mean(metrics_log[:,1])*np.mean(metrics_log[:,2]))/(np.mean(metrics_log[:,1])+np.mean(metrics_log[:,2]))),3),
                "%.3f" % round(np.mean(metrics_log_train[:,4]),3),
                "(","%.3f" % round(np.std(metrics_log_train[:,4]),3),")",
                "%.3f" % round(np.mean(metrics_log_train[:,5]),3),
                "(","%.3f" % round(np.std(metrics_log_train[:,5]),3),")")

        metrics_cba_train = np.array(metrics_cba_train)
        print(  "CBA: ",
                "%.3f" % round(np.mean(metrics_cba_train[:,1]),3),
                "(","%.3f" % round(np.std(metrics_cba_train[:,1]),3),")",
                "%.3f" % round(np.mean(metrics_cba_train[:,2]),3),
                "(","%.3f" % round(np.std(metrics_cba_train[:,2]),3), ")",
                "%.3f" % round(np.mean(metrics_cba_train[:,3]),3),
                "(","%.3f" % round(np.std(metrics_cba_train[:,3]),3), ")",
                "%.3f" % round(np.mean(metrics_cba_train[:,4]),3),
                "(","%.3f" % round(np.std(metrics_cba_train[:,4]),3),")",
                #2*((np.mean(metrics_cba[:,1])*np.mean(metrics_cba[:,2]))/(np.mean(metrics_cba[:,1])+np.mean(metrics_cba[:,2]))),
                "%.3f" % round(np.mean(metrics_cba_train[:,5]),3),
                "(","%.3f" % round(np.std(metrics_cba_train[:,5]),3),")")

        log_prec.append(metrics_log[:,1])
        log_recall.append(metrics_log[:,2])
        log_fm.append(metrics_log[:,3])
        log_train_prec.append(metrics_log_train[:,1])
        log_train_recall.append(metrics_log_train[:,2])
        log_train_fm.append(metrics_log_train[:,3])
        cba_prec.append(metrics_cba[:,1])
        cba_recall.append(metrics_cba[:,2])
        cba_fm.append(metrics_cba[:,3])
        cba_train_prec.append(metrics_cba_train[:,1])
        cba_train_recall.append(metrics_cba_train[:,2])
        cba_train_fm.append(metrics_cba_train[:,3])
        print("-------------------------------------")

    plot_diversity(log_prec)
    plot_diversity(log_recall)
    plot_diversity(log_fm)
    plot_diversity(log_train_prec)
    plot_diversity(log_train_recall)
    plot_diversity(log_train_fm)
    plot_diversity(cba_prec)
    plot_diversity(cba_recall)
    plot_diversity(cba_fm)
    plot_diversity(cba_train_prec)
    plot_diversity(cba_train_recall)
    plot_diversity(cba_train_fm)

 
    

