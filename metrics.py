import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, expon

def N_w(REF_pred, weight_REF):
    return np.sum(weight_REF*np.exp(REF_pred))

def N_r(REF_pred, weight_REF):
    return np.sum(weight_REF)

def output_classifier(x, N_w, N_R):
    return np.exp(x)*N_w*1./N_R/(1+np.exp(x)*N_w*1./N_R)

def MSE_test(pred, true, weight):
    return np.mean(weight*(pred-true)**2)

def MCE_fixed_thr_test(DATA_pred, REF_pred, weight_DATA, weight_REF):
    pi = np.sum(weight_DATA)*1./(np.sum(weight_DATA)+np.sum(weight_REF))
    return 0.5*np.sum(weight_DATA[DATA_pred<pi])*1./np.sum(weight_DATA) + 0.5*np.sum(weight_REF[REF_pred>pi])*1./np.sum(weight_REF)

def MCE_test(DATA_pred, REF_pred, weight_DATA, weight_REF, show=False, print_res=False):
    MCE_list = []
    thr_list = []
    for thr in np.arange(0., 1.01, 0.01):
        thr_list.append(thr)
        MCE_list.append(0.5*np.sum(weight_DATA[DATA_pred<thr])*1./np.sum(weight_DATA) + 0.5*np.sum(weight_REF[REF_pred>thr])*1./np.sum(weight_REF))
    if show:
        plt.plot(thr, MCE_list)
        plt.xlabel('threshold')
        plt.ylabel('MCE')
        plt.show()
        plt.close()
    if print_res:
        print('MCE: %f, thr: %f'%(np.min(MCE_list), thr_list[np.argmin(MCE_list)]))
    return np.min(MCE_list), thr_list[np.argmin(MCE_list)]

def BA_test(DATA_pred, REF_pred, weight_DATA, weight_REF, show=False, print_res=False):
    mce, thr = MCE_test(DATA_pred, REF_pred, weight_DATA, weight_REF, show=show, print_res=False)
    if print_res:
        print('BA: %f, thr: %f'%(1-mce, thr))
    return 1-mce, thr

def ACC_test(DATA_pred, REF_pred, weight_DATA, weight_REF, show=False, print_res=False):
    ACC_list = []
    thr_list = []
    for thr in np.arange(0., 1.01, 0.01):
        thr_list.append(thr)
        ACC_list.append(1- (np.sum(weight_DATA[DATA_pred<thr])+np.sum(weight_REF[REF_pred>thr]))*1./(np.sum(weight_DATA) + np.sum(weight_REF)) )
    if show:
        plt.plot(thr, ACC_list)
        plt.xlabel('threshold')
        plt.ylabel('ACC')
        plt.show()
        plt.close()
    if print_res:
        print('ACC: %f, thr: %f'%(np.max(ACC_list), thr_list[np.argmax(ACC_list)]))
    return np.max(ACC_list),thr_list[np.argmax(ACC_list)]

def ROC(DATA_pred, REF_pred, weight_DATA, weight_REF):
    false_positive, true_positive = [], []
    for thr in np.flip(np.arange(0.,1.01, 0.01)):
        false_positive.append(np.sum(weight_REF[REF_pred>thr])*1./np.sum(weight_REF))
        true_positive.append(np.sum(weight_DATA[DATA_pred>thr])*1./np.sum(weight_DATA))
    return false_positive, true_positive

def AUC_test(DATA_pred, REF_pred, weight_DATA, weight_REF):
    false_positive, true_positive = ROC(DATA_pred, REF_pred, weight_DATA, weight_REF)
    auc = 0
    for i in range(len(false_positive)-1):
        auc += 0.5*(false_positive[i+1]-false_positive[i])*(true_positive[i+1]+true_positive[i])
    return auc

def KS_test(DATA_pred, REF_pred, weight_DATA, weight_REF, bins, show=False):
    #bins = np.arange(0., 1.001, 0.001)
    hD=plt.hist(DATA_pred, weights=weight_DATA, histtype='step', bins=bins, label='DATA prediction')
    hR=plt.hist(REF_pred,  weights=weight_REF,  histtype='step', bins=bins, label='REF prediction')
    x = 0.5*(hR[1][1:]+hR[1][:-1])
    if show: plt.show()
    plt.close()
    cdfD=np.cumsum(hD[0])/np.sum(hD[0])
    cdfR=np.cumsum(hR[0])/np.sum(hR[0])
    plt.plot(cdfD)
    plt.plot(cdfR)
    if show: plt.show()
    plt.close()
    KStest = np.abs(cdfD-cdfR)
    return np.max(KStest), x[np.argmax(KStest)]

def SsqrtB_Pois_test(DATA_pred, REF_pred, weight_DATA, weight_REF, show=False, print_res=False, Bthr=0):
    SsqrtB   = []
    thr_list = []
    B_list = []
    S_list = []
    for thr in np.arange(0., 1.01, 0.01):
        B = np.sum(weight_REF[REF_pred>thr])
        D = np.sum(weight_DATA[DATA_pred>thr])
        if B > Bthr:
            tmp = norm.isf(poisson.sf(D, mu=B))
            if tmp==np.inf: tmp=100
            SsqrtB.append(tmp)
            thr_list.append(thr)
            B_list.append(B)
            S_list.append(D-B)
            
    SsqrtB   = np.array(SsqrtB)
    thr_list = np.array(thr_list)
    if show:
        plt.plot(thr, SsqrtB)
        plt.xlabel('threshold')
        plt.ylabel(r'$S/\sqrt{B}$')
        plt.show()
        plt.close()
    if print_res:
        #print(SsqrtB)
        print('THR:%f, S: %f, B: %f'%(Bthr, S_list[np.argmax(SsqrtB)], B_list[np.argmax(SsqrtB)]))
        print('S/sqrt(B): %f, thr: %f'%(np.max(SsqrtB), thr_list[np.argmax(SsqrtB)]))
    return np.max(SsqrtB), thr_list[np.argmax(SsqrtB)]

def SsqrtB_test(DATA_pred, REF_pred, weight_DATA, weight_REF, show=False, print_res=False, Bthr=5):
    SsqrtB   = []
    thr_list = []
    B_list = []
    S_list = []
    for thr in np.arange(0., 1., 0.001):
        B = np.sum(weight_REF[REF_pred>thr])
        S = np.sum(weight_DATA[DATA_pred>thr])-B
        if B > Bthr: 
            SsqrtB.append(S/np.sqrt(B))
            thr_list.append(thr)
            B_list.append(B)
            S_list.append(S)
    SsqrtB   = np.array(SsqrtB)
    thr_list = np.array(thr_list)
    if show:
        plt.plot(thr, SsqrtB)
        plt.xlabel('threshold')
        plt.ylabel(r'$S/\sqrt{B}$')
        plt.show()
        plt.close()
    if print_res:
        print('THR:5, S: %f, B: %f'%(S_list[np.argmax(SsqrtB)], B_list[np.argmax(SsqrtB)]))
        print('S/sqrt(B): %f, thr: %f'%(np.max(SsqrtB), thr_list[np.argmax(SsqrtB)]))
    return np.max(SsqrtB), thr_list[np.argmax(SsqrtB)]

def SBsqrtB_test(DATA_pred, REF_pred, weight_DATA, weight_REF, show=False, print_res=False, Bthr=0):
    SsqrtB   = []
    thr_list = []
    B_list = []
    S_list = []
    for thr in np.arange(0., 1., 0.001):
        B = np.sum(weight_REF[REF_pred>thr])
        S = np.sum(weight_DATA[DATA_pred>thr])
        if B > Bthr:
            SsqrtB.append(S/np.sqrt(B))
            thr_list.append(thr)
            B_list.append(B)
            S_list.append(S)
    SsqrtB   = np.array(SsqrtB)
    thr_list = np.array(thr_list)
    if show:
        plt.plot(thr, SsqrtB)
        plt.xlabel('threshold')
        plt.ylabel(r'$S/\sqrt{B}$')
        plt.show()
        plt.close()
    if print_res:
        print('THR: 5, S: %f, B: %f'%(S_list[np.argmax(SsqrtB)], B_list[np.argmax(SsqrtB)]))
        print('(S+B)/sqrt(B): %f, thr: %f'%(np.max(SsqrtB), thr_list[np.argmax(SsqrtB)]))
    return np.max(SsqrtB), thr_list[np.argmax(SsqrtB)]

def CHI2_test(DATA, REF, weight_DATA, weight_REF, bins, show=False):
    E = plt.hist(REF, weights=weight_REF, histtype='step', bins=bins, label='REF')[0]
    O = plt.hist(DATA, weights=weight_DATA, histtype='step', bins=bins, label='DATA')[0]
    if show:
        plt.yscale('log')
        plt.legend()
        plt.show()
    plt.close()
    return np.sum((O-E)**2*1./E)

def CHI2FLAT_test(DATA, REF, weight_DATA, weight_REF, e_i, N_R, analytic=False, show=False):
    if analytic:
        bins = expon.ppf(np.arange(0., 1., e_i), scale=1)
        if np.max(DATA)>bins[-1]:
            bins = np.append(bins, np.max(DATA))
    else:
        bins = np.quantile(REF, np.arange(0., 1., e_i))
        if np.max(REF)>bins[-1]:
            bins = np.append(bins, np.max(REF))
        if np.min(REF)<bins[0]:
            bins = np.append(np.min(REF), bins)
        if np.max(DATA)>bins[-1]:
            bins[-1]=np.max(DATA)
        if np.min(DATA)<bins[0]:
            bins[0]=np.min(DATA)
    
    #bins = np.append(bins, [np.exp(5.)])
    if not analytic:
        E = plt.hist(REF, weights=weight_REF, histtype='step', bins=bins, label='REF')[0]
    O = plt.hist(DATA, weights=weight_DATA, histtype='step', bins=bins, label='DATA')[0]
    if not analytic:
        # merge bins for which E==0
        E_fixed = []
        O_fixed = []
        O_tmp = 0
        for i in range(len(E)):
            if E[i]==0: 
                O_tmp += O[i]
                continue
            else:
                E_fixed.append(E[i])
                O_fixed.append(O[i]+O_tmp)
                O_tmp = 0
        E_fixed = np.array(E_fixed)
        O_fixed = np.array(O_fixed)
    if show:
        #plt.scatter(E, label='expected')
        #plt.scatter(O, label='observed')
        plt.yscale('log')
        plt.legend()
        plt.show()
    plt.close()
    if analytic:
        return np.sum((O-e_i*N_R)**2*1./(e_i*N_R))
    else:
        return np.sum((O_fixed-E_fixed)**2*1./E_fixed)

def KSextended_test(DATA_pred, REF_pred, weight_DATA, weight_REF, bins, show=False):
    #bins = np.arange(0., 1.001, 0.001)
    hD=plt.hist(DATA_pred, weights=weight_DATA, histtype='step', bins=bins, label='DATA prediction')
    hR=plt.hist(REF_pred,  weights=weight_REF,  histtype='step', bins=bins, label='REF prediction')
    if show: plt.show()
    plt.close()

    cdfD=np.cumsum(hD[0])/np.sum(hR[0]) # both normalized to N(R)
    cdfR=np.cumsum(hR[0])/np.sum(hR[0])
    plt.plot(cdfD)
    plt.plot(cdfR)
    if show: plt.show()
    plt.close()
    return np.max(np.abs(cdfD-cdfR))

def ROCextended(DATA_pred, REF_pred, weight_DATA, weight_REF):
    false_positive, true_positive = [], []
    for thr in np.flip(np.arange(0.,1.01, 0.01)):
        false_positive.append(np.sum(weight_REF[REF_pred>thr])/np.sum(weight_REF))
        true_positive.append(np.sum(weight_DATA[DATA_pred>thr])/np.sum(weight_REF)) # both normalized to N(R)
    return false_positive, true_positive

def AUCextended_test(DATA_pred, REF_pred, weight_DATA, weight_REF):
    false_positive, true_positive = ROCextended(DATA_pred, REF_pred, weight_DATA, weight_REF)
    auc = 0
    for i in range(len(false_positive)-1):
        auc += 0.5*(false_positive[i+1]-false_positive[i])*(true_positive[i+1]+true_positive[i])
    return auc
##############################
def ECDF_edges(data, weights):
    Ndata_weighted = np.sum(weights)
    idx_sort       = np.argsort(data, axis=0)
    data_sort      = data[idx_sort].reshape((-1,))
    weights_sort   = weights[idx_sort].reshape((-1,))
    ECDF = np.cumsum(weights_sort,axis=0)
    ECDF/= float(ECDF[-1])
    return data_sort, weights_sort, ECDF

def ECDF_eval(x, x_edges, ECDF_edges):
    idx = np.sum((x_edges<x)*1)
    if idx==0: 
        xmin, xmax = 0, x_edges[idx]
        ymin, ymax = 0, ECDF_edges[idx]
    elif idx==len(x_edges):
        xmin, xmax = x_edges[idx-1], x
        ymin, ymax = ECDF_edges[idx-1], 1
    else:
        xmin, xmax = x_edges[idx-1], x_edges[idx]
        ymin, ymax = ECDF_edges[idx-1], ECDF_edges[idx]
    y = ymin +(ymax-ymin)/(xmax-xmin)*(x-xmin)
    return y

def Cramer_von_Mises_test(x_scan, x_edges, ECDF_edges, x0_edges, ECDF0_edges, print_res=False):
    integral = 0
    x = np.sort(x_scan)
    for i in range(x_scan.shape[0]-1):
        weight = 1
        FN = ECDF_eval(x[i], x_edges, ECDF_edges)
        F0 = ECDF_eval(x[i], x0_edges, ECDF0_edges)
        g_left = (FN-F0)**2*weight
        FN = ECDF_eval(x[i+1], x_edges, ECDF_edges)
        F0 = ECDF_eval(x[i+1], x0_edges, ECDF0_edges)
        g_right = (FN-F0)**2*weight
        integral += (g_left+g_right)*(x[i+1]-x[i])*0.5
    Ndata = x_edges.shape[0]
    if print_res:
        print('Cramer von Mises: %f'%(integral*Ndata))
    return integral*Ndata

def Anderson_Darling_test(x_scan, x_edges, ECDF_edges, x0_edges, ECDF0_edges, print_res=False):
    integral = 0
    x = np.sort(x_scan)
    for i in range(x_scan.shape[0]-1):
        FN = ECDF_eval(x[i], x_edges, ECDF_edges)
        F0 = ECDF_eval(x[i], x0_edges, ECDF0_edges)
        if np.abs(F0)<1e-6 or np.abs(F0-1)<1e-6: continue
        weight = 1./F0/(1-F0)
        g_left = (FN-F0)**2*weight
        FN = ECDF_eval(x[i+1], x_edges, ECDF_edges)
        F0 = ECDF_eval(x[i+1], x0_edges, ECDF0_edges)
        if np.abs(F0)<1e-6 or np.abs(F0-1)<1e-6: continue
        weight = 1./F0/(1-F0)
        g_right = (FN-F0)**2*weight
        integral += (g_left+g_right)*(x[i+1]-x[i])*0.5
    Ndata = x_edges.shape[0]
    if print_res:
        print('Anderson Darling: %f'%(integral*Ndata))
    return integral*Ndata

#############################################
def Moran_test(data, x0_edges, ECDF0_edges, epsilon=1e-10):
    # apply cumulative from Reference
    y = np.array([ECDF_eval(x, x0_edges, ECDF0_edges) for x in data[:, 0]]).reshape((-1,))
    y = np.sort(y, axis=0)
    if y[0]>0: y = np.append([0.], y)
    if y[-1]<1: y = np.append(y, [1.])
    s = y[1:]-y[:-1]
    s[s==0] = epsilon # control points that are in the first and last bin of the EDF (flat=>s=0)
    return -1*np.sum(np.log(s))

def RPS_test(data, x0_edges, ECDF0_edges, epsilon=1e-10):
    # apply cumulative from Reference
    y = np.array([ECDF_eval(x, x0_edges, ECDF0_edges) for x in data[:,0]]).reshape((-1,))
    y = np.sort(y, axis=0)
    if y[0]>0: y = np.append([0.], y)
    if y[-1]<1: y = np.append(y, [1.])
    rps = 0
    s = y[1:]-y[:-1]
    s[s==0] = epsilon # control points that are in the first and last bin of the EDF (flat=>s=0)
    while len(s)>1:
        rps -= np.sum(np.log(s))
        s = (s[1:]+s[:-1]) # centroids
        s/= np.sum(s)  #between 0 and 1
    min_rps = np.sum([j*np.log(j) for j in np.arange(1, data.shape[0]+2, 1)]) # for x_1=0 and x_n=1
    return min_rps/rps

##########################
def Cut_and_Count_1_test(DATA_pred, REF_pred, weight_DATA, weight_REF, show=False, print_res=False):
    sat_LRT  = []
    thr_list = []
    #Ndata = np.sum(weight_DATA)
    #NR = np.sum(weight_REF)
    for thr in np.arange(0., 1.01, 0.01):
        B = np.sum(weight_REF[REF_pred>thr])
        D = np.sum(weight_DATA[DATA_pred>thr])
        if B > 0 and D>0:
            sat_LRT.append(2*(B-D + D*np.log(D/B)))
            thr_list.append(thr)


    sat_LRT  = np.array(sat_LRT)
    thr_list = np.array(thr_list)
    if show:
        plt.plot(thr_list, sat_LRT)
        plt.xlabel('threshold')
        plt.ylabel('sat LRT (1 bin)')
        plt.show()
    plt.close()
    if print_res:
        print('THR:%f, sat. LRT: %f'%(thr_list[np.argmax(sat_LRT)], np.max(sat_LRT)))
    return np.max(sat_LRT), thr_list[np.argmax(sat_LRT)]

def Cut_and_Count_2_test(DATA_pred, REF_pred, weight_DATA, weight_REF, show=False, print_res=False):
    sat_LRT  = []
    thr_list = []
    Ndata = np.sum(weight_DATA)
    NR = np.sum(weight_REF)
    for thr in np.arange(0., 1.01, 0.01):
        B = np.sum(weight_REF[REF_pred>thr])
        D = np.sum(weight_DATA[DATA_pred>thr])
        if B > 0 and D>0 and (Ndata-D)>0 and (NR-B)>0:
            sat_LRT.append(2*(NR-Ndata + D*np.log(D/B)+ (Ndata-D)*np.log((Ndata-D)/(NR-B))))
            thr_list.append(thr)


    sat_LRT  = np.array(sat_LRT)
    thr_list = np.array(thr_list)
    if show:
        plt.plot(thr_list, sat_LRT)
        plt.xlabel('threshold')
        plt.ylabel('sat LRT (2 bins)')
        plt.show()
    plt.close()
    if print_res:
        print('THR:%f, sat. LRT: %f'%(thr_list[np.argmax(sat_LRT)], np.max(sat_LRT)))
    return np.max(sat_LRT), thr_list[np.argmax(sat_LRT)]
