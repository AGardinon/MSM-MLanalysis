#
import numpy as np
import time
import pipeTrajectory as pT
import pipeDescriptor as pD

# -----------------------------------------------------------------
#
#   INPUT DICTs
#
# -----------------------------------------------------------------

# TRAJ
traj_dict = dict(
    dirname='/mnt/c/Users/andre/Documents/Work/1.Cambridge/0.systems/1.ioanSystems/data/LiquidElectrolyte/',
    #'/data/ibm26/MLdata/LiquidElectrolyte/',
    sysname='traj_2.1.xyz',
    #'traj_2.1.xyz',
    read_frame_tuple = (0,250,1),
    
    traj_species_dict = dict(
        rcut_correction = {'H':1,'C':1,'O':1,'Li':0.1,'P':1,'F':1},
        molecular_species = ['EC', 'EMC', 'Li', 'PF6']),
    
    zshift_tuple = ('EC', [6, 8]),
    
    unwrap_dict = None
    # dict(
    #     species = ['Li','PF6'],
    #     method = 'hybrid'),

)

# -----------------------------------------------------------------
#
#   MAIN PIPE
#
# -----------------------------------------------------------------

# --- 1. Traj

# - Init the analysis
trajObj = pT.TrajLoader(**traj_dict)

# - Reading the amounts of frames needed
t0 = time.time()
traj_read = trajObj.readTraj()
traj_COM_read = trajObj.readTrajCOM(saveFile=True)
t1 = time.time()
print(f"{np.round(t1-t0, 1)}s")
    
print("\n### END ###\n")
