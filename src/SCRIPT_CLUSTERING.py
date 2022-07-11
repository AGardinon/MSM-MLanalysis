#!
import numpy as np
from tqdm import tqdm
import miscTools as mtool
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
#
#   INPUT VARs
#
# -----------------------------------------------------------------

sysDir = './RESULTS/turbo_nmax8_lmax0_Nsp8_0-500-1/'

# --- Clustering
# - KMeans inputs
from deeptime.clustering import KMeans, RegularSpace

cluster_mode = 'KMEANS' #'KMEANS' #'REGSPACE'

# Reg space clustering dict
kmeans_mode = 'Fixed' #'Adaptive'
NCL = 150
kmeans_input_dict = dict(
    n_clusters = None,
    fixed_seed = 73,
    max_iter = 5000,
    n_jobs=8,
    progress=tqdm,
)

# Reg space clustering dict
regspace_input_dict = dict(
    dmin=0.2,  # minimum distance between cluster centers
    max_centers=250,  # maximum number of cluster centers
    n_jobs=8
)

# -----------------------------------------------------------------
#
#   MAIN PIPE
#
# -----------------------------------------------------------------

# - get the files
trajFiles = mtool.get_filesEW(folder=sysDir+'1.TICA/', 
                              extension='.npy', 
                              verbose=False)

saveDir = mtool.mkdir(path=sysDir, folder_name='2.LABELS/')
print(saveDir)

# loop over the target files
for file in trajFiles:
    print("\n# --- Analysis:\n"+file+"\n")
    X = np.load(sysDir+'1.TICA/'+file)
    print(X.shape)
    
    if cluster_mode == 'KMEANS':
        if kmeans_mode == 'Adaptive':
            n = X.shape[2]
            kmeans_input_dict.update(n_clusters=n)
            print(f"Adaptive n_clusters: {n}")
        elif kmeans_mode == 'Fixed':
            kmeans_input_dict.update(n_clusters=NCL)
        # CLUSTERING INIT
        estimator = KMeans(**kmeans_input_dict)
        savefilename = saveDir+f"KMEANS_"
            
    elif cluster_mode == 'REGSPACE':
        # CLUSTERING INIT
        estimator = RegularSpace(**regspace_input_dict)
        savefilename = saveDir+f"REGSPACE_"
    

    # FIT TRANSFORM
    XX = np.concatenate(X)
    clustering = estimator.fit(XX).fetch_model()
    labels = clustering.transform(XX)
    print("- Saving files ...\n")
    np.savetxt(savefilename+file.replace('.npy', '.labels'), labels)
    np.savetxt(savefilename+file.replace('.npy', '.centers'), clustering.cluster_centers)
    # if cluster_mode == 'KMEANS':
    #     np.savetxt(savefilename+file.replace('.npy', '.inertia'), clustering.inertias)
    
print("\n### END ###")
