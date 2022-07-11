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

n_particles = 114
eigenval_treshold = None #.6
fixed_eigenval = 18

# -----------------------------------------------------------------
#
#   MAIN PIPE
#
# -----------------------------------------------------------------
import deeptime.markov.tools as mrktool    
import deeptime.markov as markov

estimator = markov.msm.MaximumLikelihoodMSM(
    reversible=True,
    stationary_distribution_constraint=None,
)

# - get the files
trajFiles = mtool.get_filesEW(folder=sysDir+'2.LABELS/', 
                              extension='.labels', 
                              verbose=True)
saveDir = mtool.mkdir(path=sysDir, folder_name=f'3.MSMeigenval{fixed_eigenval}/')
print(saveDir)

# loop over the target files
for file in tqdm(trajFiles):
    print("\n# --- Analysis:\n"+file+"\n")
    discretetraj = np.loadtxt(sysDir+'2.LABELS/'+file).astype(int)
    lagTICA = mtool.numberFromString(string=file, target='lag')
    print(f"Lagtime: {lagTICA}t* (used for tICA)")
    
    # --- MARKOV STATE MODEL
    # --- from KMeans discretized space
    print("\n# --- MLMSM\n")
    msm = estimator.fit_from_discrete_timeseries(
        discrete_timeseries=discretetraj.reshape(n_particles,-1),
        lagtime=lagTICA,
        count_mode='sliding').fetch_model()    
    print(f"Number of states: {msm.n_states}")
    savename1 = 'MLMSM_'+file.replace('.labels','')
    np.savetxt(saveDir+savename1+'.trmtrx', msm.transition_matrix)
    np.savetxt(saveDir+savename1+'.eigenval', msm.eigenvalues())
    
    # --- PCCA+ coarse graining
    print("\n# --- PCCA+\n")
    if eigenval_treshold:
        CGstates = len([e for e in msm.eigenvalues() if e >= eigenval_treshold])
    elif fixed_eigenval:
        CGstates = fixed_eigenval
    print(f"Relevant Eigenvalues: {CGstates} (treshold={eigenval_treshold})")
    pcca = msm.pcca(n_metastable_sets=CGstates)
    CGTRmatrix = pcca.coarse_grained_transition_matrix
    # PCCA_CGlabels = pcca_labels = np.argmax(pcca.memberships[discretetraj], axis=1)
    PCCA_CGlabels = np.argmax(pcca.memberships[discretetraj], axis=1)
    
    # --- saving file
    savename2 = f'PCCA_'+file.replace('.labels','')
    np.savetxt(saveDir+savename2+'.trmtrx', CGTRmatrix)
    np.savetxt(saveDir+savename2+'.labels', PCCA_CGlabels)
    np.savetxt(saveDir+savename2+'.eigenval', mrktool.analysis.eigenvalues(CGTRmatrix))
    
print("\n### END ###")
