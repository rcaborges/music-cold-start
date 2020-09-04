import numpy as np
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from mlp import *
#from cnn import *
from cba import *
import time
import matplotlib.pyplot as plt
import ast

def get_content_data(k):
    df = pd.read_csv("../data/bpmd/codewords/mfcc/mfcc_hist_"+str(k)+".csv",header=0,sep=',')
    return df

def get_user_data(user_num):
    usr = int(user_num)
    ### MUSICAS TOCADAS
    df = pd.read_csv("../data/bpmd/user_play_counts.csv",header=None,sep=',')
    df.columns = ['user', 'song','play_count']
    usr_count = df.loc[df['user'] == usr]
    play_count = usr_count['song'].values
    played = list(play_count)
    ### MUSICAS ESCUTADAS
    df = pd.read_csv("../data/bpmd/user_listened_counts.csv",header=None,sep=',')
    df.columns = ['user', 'song','play_count']
    usr_count = df.loc[df['user'] == usr]
    listened_count = usr_count['song'].values
    listened_count = list(listened_count)

    listened = list(set(play_count).intersection(listened_count))
    skiped = list(set(play_count) - set(listened_count))
    return listened, skiped, played

def user_count():
    df = pd.read_csv("../data/bpmd/user_listened_counts.csv",header=None,sep=',')
    df.columns = ['user', 'song','play_count']
    users = set(df['user'])
    return users

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

    # TODO
    y_pred = []
    for a in pred_y:
        if a >= thd: 
            y_pred.append(1)
        else:
            y_pred.append(0)

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

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def plot_comparison(data_a, data_b, ticks, method):
    plt.figure()
    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
    #bpr = plt.boxplot(data_c, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')
    #set_box_color(bpr, '#2C7BB6')
    
    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='CBA + Codewords')
    plt.plot([], c='#2C7BB6', label='MLP + Codewords')
    #plt.plot([], c='#2C7BB6', label='CNN + STFT')
    plt.legend()
    plt.title(method)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    #plt.ylim(np.min(np.concatenate((data_a,data_b),axis=1)), np.max(np.concatenate((data_a,data_b),axis=1)))
    plt.tight_layout()
    plt.savefig('plot/boxcompare_'+method+'.png')

def print_results(method_name, results):
    results = np.array(results)
    print(  method_name+": ",
            "%.3f" % round(np.mean(results[:,1]),3),
            "(","%.3f" % round(np.std(results[:,1]),3),")",
            "%.3f" % round(np.mean(results[:,2]),3),
            "(","%.3f" % round(np.std(results[:,2]),3), ")",
            "%.3f" % round(np.mean(results[:,3]),3),
            "(","%.3f" % round(np.std(results[:,3]),3), ")",
            "%.3f" % round(np.mean(results[:,4]),3),
            "(","%.3f" % round(np.std(results[:,4]),3),")",
            "%.3f" % round(np.mean(results[:,5]),3),
            "(","%.3f" % round(np.std(results[:,5]),3),")")

if __name__ == "__main__":
    #LE ARQUIVOS DE EXECUCOES PARA DETECTAR QUANTOS USUARIOS EXISTEM NA BASE
    users = user_count()
    min_lst = 100
    arocs_cba, arocs_mlp = [], []
    fmeas_cba, fmeas_mlp = [], []

    # LOAD SONGS SPECTROGRAM
    #df_specs = pd.read_csv('../data/specs.csv', sep=';',names=['id','spec'])
    #spec = ast.literal_eval(specs.iloc[0]['spec'].strip())

    knum = [5,10,25,50,100,200]
    for k_num in knum:
        error_cba, error_mlp = [], []
        metrics_cba, metrics_mlp = [], []
        metrics_cba_train, metrics_mlp_train = [], []
        #LE DADOS DE MFCC DAS MUSICAS
        mfcc = get_content_data(k_num)
        print(mfcc.shape)
    
        for r in range(20):
            #PARA CADA USUARIO SELECIONA MUSICAS TOCADAS E MUSICAS OUVIDAS
            for usr in users:
                #SELECIONA DENTRE AS MUSICAS TOCAS QUAIS FORAM OUVIDAS E QUAIS PULADAS
                lst,skp,pld = get_user_data(usr)
                #SELECIONAR QUEM OUVIU ACIMA DE UM VALOR E SALTOU ACIMA DE OUTRO VALOR
                if len(lst) < min_lst: continue
                if len(skp) < min_lst: continue
                #AMOSTRA MUSICAS ESCUTADAS PARA BALANCEAR EM RELACAO A SALTADAS
                # CUIDADO NO CASO QUE SALTOS E MAIOR QUE MUSICAS ESCUTADAS
                lst = np.random.choice(lst, size=len(skp), replace=False)
                pld = [*lst, *skp]

                #FILTRA A MATRIZ DE MFCC RESTANDO APENAS AS QUE FORAM TOCADAS
                mfcc_played = []
                mfcc_played = mfcc.loc[mfcc['id'].isin(pld)]
                mfcc_played = mfcc_played.drop('ano',1)
                ids = mfcc_played['id'].copy()
                mfcc_played = mfcc_played.drop('id',1)
                #NORMALIZACAO COM A SOMA DE TODOS VALORES DO HISTOGRAMA
                #PARA COMPENSAR MUSICAS COM DURACOES DIFERENTES
                mfcc_played = mfcc_played.div(mfcc_played.sum(axis=1), axis=0)
                mfcc_played['id'] = ids
                mfcc_played = mfcc_played.assign(play=np.zeros(len(pld)))
                mfcc_played.loc[mfcc_played['id'].isin(lst),'play'] = 1.0

                #EMBARALHA LINHAS
                mfcc_played = mfcc_played.sample(frac=1)
                #SEPARA EM DADOS DE TREINO E TESTE
                lmt = int(len(pld)*0.8)
                train_y = mfcc_played['play'][:lmt]
                test_y = mfcc_played['play'][lmt:]
                train_x = mfcc_played.drop('play',1)[:lmt]
                test_x = mfcc_played.drop('play',1)[lmt:]

                #CBA
                beta_value = float(k_num)/100
                beta_matrix = cba(train_x,train_y,beta_value)
                preds_cba = cba_calc_probs(beta_matrix, test_x)  
                preds_cba_train = cba_calc_probs(beta_matrix, train_x)  
                acc,precision,recall,fscore,ap,aroc = calc_metrics(preds_cba,test_y,preds_cba_train,train_y)
                metrics_cba.append([acc,precision,recall,fscore,ap,aroc])
                acc,precision,recall,fscore,ap,aroc = calc_metrics(preds_cba_train,train_y,preds_cba_train,train_y)
                metrics_cba_train.append([acc,precision,recall,fscore,ap,aroc])

                #MLP
                preds_mlp, preds_mlp_train = dbc(train_x, train_y, test_x, test_y)
                acc,precision,recall,fscore,ap,aroc = calc_metrics(preds_mlp,test_y,preds_mlp_train,train_y)
                metrics_mlp.append([acc,precision,recall,fscore,ap,aroc])
                acc,precision,recall,fscore,ap,aroc = calc_metrics(preds_mlp_train,train_y,preds_mlp_train,train_y)
                metrics_mlp_train.append([acc,precision,recall,fscore,ap,aroc])
   
        print(k_num)
        print('####### TEST ###########')
        print_results("CBA",metrics_cba)
        fmeas_cba.append(np.array(metrics_cba)[:,3])
        arocs_cba.append(np.array(metrics_cba)[:,5])
        print_results("MLP",metrics_mlp)
        fmeas_mlp.append(np.array(metrics_mlp)[:,3])
        arocs_mlp.append(np.array(metrics_mlp)[:,5])

        print('####### TRAIN ###########')
        print_results("CBA",metrics_cba_train)
        print_results("MLP",metrics_mlp_train)

#PLOT BOXPLOT
plot_comparison(fmeas_cba, fmeas_mlp, knum, 'fmeasure')
plot_comparison(arocs_cba, arocs_mlp, knum, 'aroc')
    



