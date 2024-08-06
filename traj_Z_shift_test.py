#!

import os
import numpy as np
import msmanalysis.pipeTrajectory as pT

path = os.path.dirname(os.path.abspath(__file__))+'/sample_data/'

traj_dict = dict(
    dirname = path,
    sysname='/traj_2.1_0-100-1.xyz',
    read_frame_tuple = (0,100,1),
    traj_species_dict = dict(
        rcut_correction = {'H':1,'C':1,'O':1,'Li':0.1,'P':1,'F':1},
        molecular_species = ['EC', 'EMC', 'Li', 'PF6']),
    zshift_dict = {'mol_to_shift': 'EC', 
                   'z_to_shift': [6,8], 
                   'shifting': {6:14, # change 6 -> 14
                                8:16} # change 8 -> 16
                    }, #set to None to suppress chemical substitution
    unwrap_dict = None
)

trajObj = pT.TrajLoader(**traj_dict)
traj = trajObj.readTraj()

print('Z numbers',np.unique(trajObj.Znumbers, return_counts=True))
print('Original',np.unique(trajObj.OriginalZnumbers, return_counts=True))

print('From read traj',np.unique(traj[0].numbers, return_counts=True))

print(traj[-1])