#!
import numpy as np
from tqdm import tqdm
import miscTools as mtool
from deeptime.decomposition import TICA

# -----------------------------------------------------------------
#
#   INPUT VARs
#
# -----------------------------------------------------------------

sysDir = './0.SOAP/TEST/'

lag_scan = [1,2,4,8,16,32,64]

tica_input_dict = dict(
    lagtime = None,
    var_cutoff  = .95,
)

saveDir = None

# -----------------------------------------------------------------
#
#   MAIN PIPE
#
# -----------------------------------------------------------------

# - get the files
trajFiles = mtool.get_filesEW(folder=sysDir, 
                              extension='.npy', 
                              verbose=True)

vampscore_list = list()

# loop over the target files
for file in trajFiles:
    print("\n# --- Analysis:\n"+file+"\n")
    
    # - creating folder
    if saveDir:
        sf = saveDir
    else:
        mask = [0,3,4,5,-1]
        sf = '_'.join(file.split('_')[i] for i in mask).replace('.npy','')
    saveDirTICA = './RESULTS/'+sf+'/1.TICA/'
    _ = mtool.mkdir(path='', 
                    folder_name=saveDirTICA, 
                    overwrite=True)
    print(' ')
    
    # --- tICA embedding
    for lagt in tqdm(lag_scan, desc='tICA lag scans'):
        # dict update for the lagscan
        tica_input_dict.update(lagtime=lagt)
        print(f"lag= {lagt}t*")
        # - Timeseries
        timeseries = mtool.build_timeseries(sysDir+file, lagtime=lagt)
        # - tICA
        tica = TICA(**tica_input_dict)
        model = tica.fit_from_timeseries(timeseries).fetch_model()
        # - FIT
        X = model.transform(timeseries.trajectories)
        print(f"- Outputs:\n\tNcomopnents= {X.shape[2]} (max:{len(model.cumulative_kinetic_variance)})\n\tVamps-2 score= {model.score(r=2)}")
        vampscore_list.append(model.score(r=2))
        print("- Saving files ...\n")
        savefilename = saveDirTICA+f"TICA_lag{lagt}_"
        np.save(savefilename+file, X)
        np.savetxt(savefilename+file.replace('.npy', '.kvar'), 
                   model.cumulative_kinetic_variance)
        np.savetxt(saveDirTICA+"vampscore_w_lag.dat", vampscore_list)
        # - freeing memory
        del model
        del X
        del timeseries    

print("\n### END ###")
