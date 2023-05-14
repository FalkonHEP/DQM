import numpy as np
import pandas as pd

import os, time

import torch

from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy.spatial.distance import pdist

from scipy.stats import ks_2samp



def candidate_sigma(data, perc=90):

    pairw = pdist(data)

    return round(np.percentile(pairw,perc),1)




def get_logflk_config(M,flk_sigma,lam,weight,iter=[100],seed=None,cpu=False):
    return {
            'kernel' : GaussianKernel(sigma=flk_sigma),
            'M' : M, #number of Nystrom centers,
            'penalty_list' : lam, # list of regularization parameters,
            'iter_list' : iter, #list of number of CG iterations,
            'options' : FalkonOptions(cg_tolerance=np.sqrt(1e-7), keops_active='no', use_cpu=cpu, debug = False),
            'seed' : seed, # (int or None), the model seed (used for Nystrom center selection) is manually set,
            'loss' : WeightedCrossEntropyLoss(kernel=GaussianKernel(sigma=flk_sigma), neg_weight=weight),
            }




def learn_t(Xtorch,Ytorch,model):
    
    model.fit(Xtorch, Ytorch)
    preds = model.predict(Xtorch)
    
    return 2 * torch.sum(preds[Ytorch.flatten()==1]).item(), preds[Ytorch.flatten()==0]





def read_data(file,features,rnd=None):
    data = pd.read_csv(file, usecols=features)
    data = data.sample(frac=1,random_state=rnd).reset_index(drop=True)
    return data.to_numpy()





def higgs_standardize(X):

    Xnorm = X.copy()

    for j in range(Xnorm.shape[1]):
        column = Xnorm[:, j]

        mean = np.mean(column)
        std = np.std(column)
    
        if np.min(column) < 0:
            column = (column-mean)*1./ std
        elif np.max(column) > 1.0:                                                                                                                                        
            column = column *1./ mean

    
        Xnorm[:, j] = column
    
    return Xnorm

def standardize(X):

    Xnorm = X.copy()

    for j in range(Xnorm.shape[1]):
        column = Xnorm[:, j]

        mean = np.mean(column)
        std = np.std(column)
    
        column = (column-mean)*1./ std

    
        Xnorm[:, j] = column
    
    return Xnorm





def run_toys(reference, data, filename, N0, N1, rng,  flk_config, n_toys=10, std=True, replacement=False, plt_freq=0, plot_dir=None, data_label=None):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    #save config file (temporary solution)
    with open(os.path.dirname(filename)+"/flk_config.txt","w") as f:
        f.write( str(flk_config) )

    N = N0 + N1
    dim = data.shape[1]
    weight = N1/N0

    if not replacement:
        if (data.shape[0] / N1)<n_toys:
            print("Not enough events to process {} toys without replacement. Processing {} toys instead.".format(n_toys,np.floor(data.shape[0] / N1)))
            n_toys = int(np.floor(data.shape[0] / N1))
            max_length = N1*n_toys
            data_idx_batches =  np.array_split(range(max_length),n_toys)
        else:   
            max_length = N1*n_toys
            #data_idx_batches =  np.array_split(range(data.shape[0]),np.ceil(data.shape[0] / N1))
            data_idx_batches =  np.array_split(range(max_length),n_toys)

    toys = range(n_toys)

    for i in toys:

        st_time = time.time()

        print("[--] Toy {}: ".format(i))
        # build training set
        # initialize dataset
        X = np.zeros(shape=(N,dim))
        # fill with ref
        X[:N0,:] = rng.choice(reference,size=N0,replace=False)
        # fill with data
        if replacement: X[N0:,:] = rng.choice(data,size=N1,replace=False)
        else: X[N0:,:] = data[data_idx_batches[i],:]
        # initialize labes
        Y = np.zeros(shape=(N,1))
        # fill with data labels
        Y[N0:,:] = np.ones((N1,1))

        if std=='higgs': Xnorm = higgs_standardize(X)
        elif std=='scaler': Xnorm = standardize(X)
        else: Xnorm = X

        Xtorch = torch.from_numpy(Xnorm)
        Ytorch = torch.from_numpy(Y)

        print("Reference shape:{}".format(X[Y.flatten()==0].shape))
        print("Data shape:{}".format(X[Y.flatten()==1].shape)) 

        # learn_t
        flk_config['seed']=i #seed for center selection, different for every toy
        model = LogisticFalkon(**flk_config)

        t, ref_preds = learn_t(Xtorch,Ytorch,model)
        
        dt = round(time.time()-st_time,2)

        print("t = {}\nTime = {} sec\n\t".format(t,dt))

        with open(filename, 'a') as f:
            f.write('{},{},{}\n'.format(i,t,dt))
        
        if plt_freq!=0 and i in toys[::plt_freq]:
            for j in range(X.shape[1]):
                if j in range(4): 
                    #plot_inputs(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_weight=weight,data_weight=1,title="Drift time, layer "+str(j+1),xlabel="ns",labels=('Ref',data_label),save_path=plot_dir+str(i)+'/'+str(j)+'.pdf')
                    #plot_lkratio(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,xlabel="ns",label=data_label,save_path=plot_dir+str(i)+'/dratio_'+str(j)+'.pdf')
                    plot_reco(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,ref_weight=weight,data_weight=1,title="Drift time, layer "+str(j+1),xlabel="ns",labels=('Ref',data_label),save_path=plot_dir+str(i)+'/in_reco_'+str(j)+'.pdf')
                elif j==4: 
                    #plot_inputs(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_weight=weight,data_weight=1,title="Slope",xlabel=r"$\theta$",labels=('Ref',data_label),save_path=plot_dir+str(i)+'/'+str(j)+'.pdf')
                    #plot_lkratio(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,xlabel=r"$\theta$",label=data_label,save_path=plot_dir+str(i)+'/dratio_'+str(j)+'.pdf')
                    plot_reco(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,ref_weight=weight,data_weight=1,title="Slope",xlabel=r"$\theta$",labels=('Ref',data_label),save_path=plot_dir+str(i)+'/in_reco_'+str(j)+'.pdf')
                elif j==5: 
                    #plot_inputs(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_weight=weight,data_weight=1,title="$Hits$",xlabel="n",labels=('Ref',data_label),save_path=plot_dir+str(i)+'/'+str(j)+'.pdf')
                    #plot_lkratio(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,xlabel="n",label=data_label,save_path=plot_dir+str(i)+'/dratio_'+str(j)+'.pdf')
                    plot_reco(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,ref_weight=weight,data_weight=1,title="Hits",xlabel="n",labels=('Ref',data_label),save_path=plot_dir+str(i)+'/in_reco_'+str(j)+'.pdf')


def plot_inputs(ref,data,ref_weight=1,data_weight=1,title=None,xlabel=r"$t^{(1)}_{drift}$ (ns)",labels=('Ref','Data'),save_path=None):
    bin_max = np.max([np.max(ref),np.max(data)])
    bin_min = np.min([np.min(ref),np.min(data)])
    bins = np.linspace(bin_min,bin_max,20)
    #bins=np.linspace(0,500,30)
    plt.hist(ref, bins=bins, weights=np.ones_like(ref)*ref_weight, alpha=0.7, color='lightskyblue', edgecolor='black', label=labels[0])
    plt.hist(data, bins=bins, weights=np.ones_like(data)*data_weight, fill=False, edgecolor='green',lw=3.5,  label=labels[1])
    plt.ticklabel_format(axis='y',style='sci',scilimits=(1,2))
    if title: plt.title(title,fontsize=18)
    plt.xlabel(xlabel,fontsize=18)
    plt.legend(loc='upper right',fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_lkratio(ref,data,ref_preds,xlabel=r"$t^{(1)}_{drift}$ (ns)",label=None,save_path=None):

    bin_max = np.max([np.max(ref),np.max(data)])
    bin_min = np.min([np.min(ref),np.min(data)])
    bins = np.linspace(bin_min,bin_max,20)

    x = (bins[1:] + bins[:-1])/2


    learned_weights = np.exp(ref_preds).reshape(ref.shape)

    ref_bins, _ = np.histogram(ref,  bins=bins)
    learned_data_bins, _ = np.histogram(ref, bins=bins, weights = learned_weights)

    learned_ratio = (learned_data_bins)/(ref_bins+1e-4)

    plt.plot(x,learned_ratio, '-o', color='lightskyblue', label=label)
    
    if np.max(learned_ratio)>2: plt.ylim(0,np.max(learned_ratio)+0.5)
    else: plt.ylim(0,2)

    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel("p(x|1)/p(x|0)",fontsize=18)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    if label: plt.legend(loc='upper right',fontsize=18)

    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def get_data_path(name):
    if name=='reference': path = "/data/marcol/DQM/6D/reference/RUN000085_events.csv"
    elif name=='Thr 75%': path = "/data/marcol/DQM/6D/thresholds/thresholds_75/RUN000086_events.csv"
    elif name=='Thr 50%': path = "/data/marcol/DQM/6D/thresholds/thresholds_50/RUN000087_events.csv"
    elif name=='Thr 25%': path = "/data/marcol/DQM/6D/thresholds/thresholds_25/RUN000088_events.csv"
    elif name=='Ca 75%': path = "/data/marcol/DQM/6D/cathods/cathods_75/RUN000089_events.csv"
    elif name=='Ca 50%': path = "/data/marcol/DQM/6D/cathods/cathods_50/RUN000090_events.csv"
    elif name=='Ca 25%': path = "/data/marcol/DQM/6D/cathods/cathods_25/RUN000091_events.csv"

    return path




def run_toys_2(reference, data, key, filename, N0, N1, rng,  flk_config, n_toys=10, std=True, p=None, replacement=False, plt_freq=0, plot_dir=None):

    colors = {
                "Ref": 'black',
                "Thr 75%": '#67a9cf',
                "Thr 50%": '#1c9099',
                "Thr 25%": '#016c59',
                "Ca 25%" : '#fc8d59',
                "Ca 50%" : '#e34a33',
                "Ca 75%" : '#b30000' 
    }


    os.makedirs(os.path.dirname(filename), exist_ok=True)

    #save config file (temporary solution)
    with open(os.path.dirname(filename)+"/flk_config.txt","w") as f:
        f.write( str(flk_config) )

    N = N0 + N1
    dim = data.shape[1]
    weight = N1/N0

    if not replacement:
        if (data.shape[0] / N1)<n_toys:
            print("Not enough events to process {} toys without replacement. Processing {} toys instead.".format(n_toys,np.floor(data.shape[0] / N1)))
            n_toys = int(np.floor(data.shape[0] / N1))
            max_length = N1*n_toys
            data_idx_batches =  np.array_split(range(max_length),n_toys)
        else:   
            max_length = N1*n_toys
            #data_idx_batches =  np.array_split(range(data.shape[0]),np.ceil(data.shape[0] / N1))
            data_idx_batches =  np.array_split(range(max_length),n_toys)

    toys = range(n_toys)

    for i in toys:

        st_time = time.time()

        print("[--] Toy {}: ".format(i))
        # build training set
        # initialize dataset
        X = np.zeros(shape=(N,dim))
        # fill with ref
        X[:N0,:] = rng.choice(reference,size=N0,replace=False)
        # fill with data
        if replacement: X[N0:,:] = rng.choice(data,size=N1,replace=False)
        else: X[N0:,:] = data[data_idx_batches[i],:]
        if p: 
            frac = round(p*N1)
            X[N0+frac:,:] = np.random.default_rng(i).choice(reference,size=N1-frac,replace=False)
        # initialize labes
        Y = np.zeros(shape=(N,1))
        # fill with data labels
        Y[N0:,:] = np.ones((N1,1))

        if std=='higgs': Xnorm = higgs_standardize(X)
        elif std=='scaler': Xnorm = standardize(X)
        else: Xnorm = X

        Xtorch = torch.from_numpy(Xnorm)
        Ytorch = torch.from_numpy(Y)

        print("Reference shape:{}".format(X[Y.flatten()==0].shape))
        print("Data shape:{}".format(X[Y.flatten()==1].shape)) 

        # learn_t
        flk_config['seed']=i #seed for center selection, different for every toy
        model = LogisticFalkon(**flk_config)

        t, ref_preds = learn_t(Xtorch,Ytorch,model)

        

        ref_preds = ref_preds.numpy()

        print(X[Y.flatten()==0][:,0].shape)
        print()
        
        dt = round(time.time()-st_time,2)

        print("t = {}\nTime = {} sec\n\t".format(t,dt))

        with open(filename, 'a') as f:
            f.write('{},{},{}\n'.format(i,t,dt))
        
        if plt_freq!=0 and i in toys[::plt_freq]:
            for j in range(X.shape[1]):
                if j in range(4): 
                    #plot_inputs(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_weight=weight,data_weight=1,title="Drift time, layer "+str(j+1),xlabel="ns",labels=('Ref',data_label),save_path=plot_dir+str(i)+'/'+str(j)+'.pdf')
                    #plot_lkratio(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,xlabel="ns",label=data_label,save_path=plot_dir+str(i)+'/dratio_'+str(j)+'.pdf')
                    plot_reco_2(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,ref_weight=1/len(X[Y.flatten()==0][:,j]),data_weight=1/len(X[Y.flatten()==1][:,j]), text = "Layer = " +str(j+1), xlabel="Drift time (ns)",color=colors[key],labels=('Reference',key),save_path=plot_dir+str(i)+'/in_reco_'+str(j)+'.pdf')
                elif j==4: 
                    #plot_inputs(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_weight=weight,data_weight=1,title="Slope",xlabel=r"$\theta$",labels=('Ref',data_label),save_path=plot_dir+str(i)+'/'+str(j)+'.pdf')
                    #plot_lkratio(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,xlabel=r"$\theta$",label=data_label,save_path=plot_dir+str(i)+'/dratio_'+str(j)+'.pdf')
                    plot_reco_2(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,ref_weight=1/len(X[Y.flatten()==0][:,j]),data_weight=1/len(X[Y.flatten()==1][:,j]),xlabel="Slope (degrees)",color=colors[key],labels=('Reference',key),save_path=plot_dir+str(i)+'/in_reco_'+str(j)+'.pdf')
                elif j==5: 
                    #plot_inputs(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_weight=weight,data_weight=1,title="$Hits$",xlabel="n",labels=('Ref',data_label),save_path=plot_dir+str(i)+'/'+str(j)+'.pdf')
                    #plot_lkratio(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,xlabel="n",label=data_label,save_path=plot_dir+str(i)+'/dratio_'+str(j)+'.pdf')
                    plot_reco_2(X[Y.flatten()==0][:,j],X[Y.flatten()==1][:,j],ref_preds,ref_weight=1/len(X[Y.flatten()==0][:,j]),data_weight=1/len(X[Y.flatten()==1][:,j]),xlabel=r"$n_{hits}$",color=colors[key],labels=('Reference',key),save_path=plot_dir+str(i)+'/in_reco_'+str(j)+'.pdf')



def plot_reco_2(ref,data,ref_preds,ref_weight=1,data_weight=1,title=None,xlabel=r"$t^{(1)}_{drift}$ (ns)", text=None, labels=('Reference','Data'),color='black', save_path=None):

    bin_max = np.max([np.max(ref),np.max(data)])
    bin_min = np.min([np.min(ref),np.min(data)])
    bins = np.linspace(bin_min,bin_max,15)

    x = (bins[1:]+ bins[:-1])/2

    #font = font_manager.FontProperties(family='serif', size=20)
    font = font_manager.FontProperties(family='serif', size=17)




    fig = plt.figure(figsize=(10, 8))                                                                                                                                            
    fig.patch.set_facecolor('white')
    ax=fig.add_axes([0.13 , 0.1+0.3, 0.85, 0.55])

    hr=ax.hist(ref, bins=bins, weights=np.ones_like(ref)*ref_weight, alpha=0.8, color='#cccccc', label=labels[0])[0]
    hd=ax.hist(data, bins=bins, weights=np.ones_like(data)*data_weight, histtype='step',  edgecolor=color, lw=3,  label=labels[1])[0]
    hl=ax.hist(ref, bins=bins, weights=ref_weight*np.exp(ref_preds), alpha=0.)[0]
    plt.scatter(x, hl, color='black', label='learned')

    counts = np.append(hr, [hd,hl])
    ymax = np.max(counts)*1.2
    #plt.legend(loc='upper left', ncol=3, prop=font, frameon=False, handlelength=1.6, handletextpad=0.4, columnspacing=1)
    plt.legend(loc='upper right', ncol=3, prop=font, frameon=False, handlelength=1.5, handletextpad=0.3, columnspacing=1.)
    plt.yticks(fontsize=22, fontname="serif")
    plt.xticks(fontsize=22, fontname="serif")
    plt.ylabel("Probability",fontsize=26, fontname="serif")

    ax.ticklabel_format(axis='y',style='plain',scilimits=(1,2),useOffset=False, useLocale=None, useMathText=True)
    ax.tick_params(which='both', labelbottom=False) #labelsize=16)
    ax.set_ylim(0.01,ymax)

    if title: ax.set_title(title,fontsize=22)
    
    ax2=fig.add_axes([0.13 , 0.1, 0.85, 0.3])
    learned_weights = np.exp(ref_preds).reshape(ref.shape)

    ref_bins, _ = np.histogram(ref,  bins=bins)
    learned_data_bins, _ = np.histogram(ref, bins=bins, weights = learned_weights)
    binned_ratio = hd/(hr+1e-4)
    learned_ratio = (learned_data_bins)/(ref_bins+1e-4)

    ax2.plot(x,binned_ratio, '-o', label='true (binned)', color=color, lw=2)
    ax2.plot(x,learned_ratio, '-', label='learned', color='black', lw=2)
    ax2.plot(x,np.ones_like(x), '--', color='black', lw=1)
    
    #if np.max(learned_ratio)>2: axs[1].set_ylim(0,np.max(learned_ratio)+0.5)
    #else: axs[1].set_ylim(0,2)

    plt.xlabel(xlabel,fontsize=26, fontname="serif")
    plt.ylabel("Ratio",fontsize=26, fontname="serif")

    #ax2.tick_params(labelsize=16)
    #ax2.get_yticklabels()[-1].set_visible(False)
    ax2.set_ylim(0.0005,5000)
    plt.yscale('log')
    plt.yticks(fontsize=22, fontname="serif")
    plt.xticks(fontsize=22, fontname="serif")
    font = font_manager.FontProperties(family='serif', size=20)
    plt.legend(loc='upper left', ncol=2, prop=font, frameon=False)
    if text: plt.text(x=0.02, y=0.9, s=text, color='black', fontsize=17, fontname='serif', transform = ax.transAxes)
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()