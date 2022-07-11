#
import numpy as np
from ase.io import read,write 
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
    read_frame_tuple = (0,2500,1),
    
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
#   MAIN
#
# -----------------------------------------------------------------

# --- 1. Traj

# - Init the analysis
trajObj = pT.TrajLoader(**traj_dict)

# - Reading the amounts of frames needed
t0 = time.time()
traj_read = trajObj.readTraj()
t1 = time.time()
print(f"{np.round(t1-t0, 1)}s")

write('traj_2.1_0-2500-1_newatoms.xyz', traj_read)

print("### END ###")

