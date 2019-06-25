import numpy as np

def cba(codeMatrix, bernMatrix, beta_value):
    """
    input:
       codeMatrix - matrix containing codewords of songs
       bernMatrix - matrix containing implicit feedback
       beta_value - the minimum difference between estimated matrix 
    output:
       beta - the matrix learned as the model to be used in the prediction  

    """
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
    """
    input:
       beta - matrix learned during the training 
       test_x - codewords of test songs
    output:
       prob - probabilities estimated for new songs

    """
    test_x = np.array(test_x)
    prob = []
    for mfcc in test_x:
        soma = 0.0
        for j in range(len(mfcc)):
            soma = soma + (float(mfcc[j])*float(beta[j]))
        prob.append( soma/np.sum(mfcc.astype(float)))
    prob = np.array(prob)
    return prob 

    

