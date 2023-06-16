import numpy as np
import os


from utils import get_logflk_config, read_data, run_toys, standardize, get_data_path
from datetime import datetime



# which features
features = ['t_1','t_2','t_3','t_4','theta'] #,'n_hits']

# reference and data sizes
N0=2000
N1=500
weight=N1/N0

# number of reference pseudoexperiments and measurements (toys)
n_ref_toys = 10
n_data_toys = 10

# anomalous data to include in the run
data_keys = ["Thr 75%"] #["Thr 75%","Thr 50%","Thr 25%","Ca 75%","Ca 50%","Ca 25%"]
reference_path = get_data_path("Ref")

output_dir="./runs/"+datetime.now().strftime("%d%b%y_%H%M%S")+f"/{len(features)}D_{N0}_{N1}"

rng = np.random.default_rng(1)

# since the dataset is not very large, it is ok im this case to load all the reference data at once and split later into reference sample and toys
# for different applications it might be better to implement a smarter loader
reference = read_data(reference_path,features=features,rnd=rng)

# hyperparameters (instructions in the paper)
M = 2000
lam = 1e-7
# tune sigma on a (small) reference sample
#flk_sigma = candidate_sigma(standardize(reference[:20000,:]))
flk_sigma = 4.5


flk_config = get_logflk_config(M,flk_sigma,[lam],weight=weight,iter=[100000],seed=None,cpu=False) # seed is re-set inside learn_t function


# run reference toys to estimate the distribution under the null hypothesis
run_toys(reference, reference, "Ref", output_dir+"/Ref/t.txt", N0, N1,  flk_config, n_toys=n_ref_toys, std='scaler', p=1, replacement=True, plt_freq=2, plot_dir=output_dir+"/Ref/plot_ref")

# run data toys to estimate the distribution under the alternative hypothesis
for key in data_keys:
    print("[--] DATA: "+key)
    data = read_data(get_data_path(key),features=features,rnd=rng)
    run_toys(reference, data, key, output_dir+"/"+key+"/t.txt", N0, N1,  flk_config, n_toys=n_data_toys, std='scaler', p=1, replacement=False, plt_freq=2, plot_dir=output_dir+"/"+key+"/plot_data")

