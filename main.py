import numpy as np
import os


from utils import get_logflk_config, read_data, run_toys, standardize, get_data_path
from datetime import datetime



# which features
features = ['t_1','t_2','t_3','t_4','theta'] #,'n_hits']

# data parameters
N0=2000
N1=500
weight=N1/N0

n_ref_toys = 50
n_data_toys = 20

data_keys = ["Thr 75%"] #["Thr 75%","Thr 50%","Thr 25%","Ca 75%","Ca 50%","Ca 25%"]
reference_path = get_data_path("Ref")

output_dir="./runs/"+datetime.now().strftime("%d%b%y_%H%M%S")+f"/{len(features)}D_{N0}_{N1}"

rng = np.random.default_rng(1)

reference = read_data(reference_path,features=features,rnd=rng)

# hyperparameters
M = 2000
lam = 1e-7
#flk_sigma = candidate_sigma(standardize(reference[:20000,:])) # used to tune sigma on a (small) reference sample
flk_sigma = 4.5

flk_config = get_logflk_config(M,flk_sigma,[lam],weight=weight,iter=[1000],seed=None,cpu=False) # seed is re-set inside learn_t function


run_toys(reference, reference, "Ref", output_dir+"/Ref/t.txt", N0, N1,  flk_config, n_toys=n_ref_toys, std='scaler', p=1, replacement=True, plt_freq=0, plot_dir=output_dir+"/Ref/plot_ref")


for key in data_keys:
    print("[--] DATA: "+key)
    data = read_data(get_data_path(key),features=features,rnd=rng)
    run_toys(reference, data, key, output_dir+"/"+key+"/t.txt", N0, N1,  flk_config, n_toys=n_data_toys, std='scaler', p=.5, replacement=False, plt_freq=0, plot_dir=output_dir+"/"+key+"/plot_data")

