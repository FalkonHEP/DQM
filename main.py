import numpy as np
import os


from utils import get_logflk_config, read_data, run_toys_2, standardize
from datetime import datetime



############## new code

features = ['t_1','t_2','t_3','t_4','theta'] #,'n_hits']

N0=5000
N1=500
weight=N1/N0


reference_path = "/data/marcol/DQM/6D/reference/RUN000085_events.csv"
data_paths = {
                "Thr 75%": "/data/marcol/DQM/6D/thresholds/thresholds_75/RUN000086_events.csv",
                #"Thr 25%": "/data/marcol/DQM/6D/thresholds/thresholds_25/RUN000088_events.csv",
                #"Thr 50%": "/data/marcol/DQM/6D/thresholds/thresholds_50/RUN000087_events.csv",
                #"Ca 25%": "/data/marcol/DQM/6D/cathods/cathods_25/RUN000091_events.csv",
                #"Ca 50%": "/data/marcol/DQM/6D/cathods/cathods_50/RUN000090_events.csv",
                #"Ca 75%": "/data/marcol/DQM/6D/cathods/cathods_75/RUN000089_events.csv",
            }

output_dir="./runs/"+datetime.now().strftime("%d%M%Y_%H%m%S")+f"/{len(features)}D_{N0}_{N1}/ref"

rng = np.random.default_rng(1)

reference = read_data(reference_path,features=features,rnd=rng)

M = 1000
lam = 1e-8
#flk_sigma = candidate_sigma(standardize(reference[:20000,:])) # used to tune sigma on a (small) reference sample
flk_sigma = 4.5

print("[--] FALKON SIGMA: {}".format(flk_sigma))

flk_config = get_logflk_config(M,flk_sigma,[lam],weight=weight,iter=[1000],seed=None,cpu=False) # seed is re-set inside learn_t function

run_toys_2(reference, reference, "Ref", output_dir+"/t_ref.txt", N0, N1, rng,  flk_config, n_toys=5, std='scaler', p=None, replacement=True, plt_freq=5, plot_dir=output_dir+"/plot_ref")

#run_toys(reference, reference, output_dir+"/t_ref.txt", N0, N1, rng, flk_config, n_toys=100, std='scaler', replacement=True, plt_freq=25, plot_dir=output_dir+"/plot_ref/", data_label="Ref")



for key,value in data_paths.items():
    print("[--] DATA: "+key)
    data = read_data(value,features=features,rnd=rng)
    new_output_dir=os.path.dirname(output_dir)+"/"+key
    run_toys_2(reference, data, key, new_output_dir+"/t_data.txt", N0, N1, rng,  flk_config, n_toys=5, std='scaler', p=None, replacement=False, plt_freq=2, plot_dir=new_output_dir+"/plot_data")
    #run_toys(reference, data, new_output_dir+"/t_data.txt", N0, N1, rng, flk_config, n_toys=100, std='scaler', replacement=False, plt_freq=3, plot_dir=new_output_dir+"/plot_data/", data_label=key)

