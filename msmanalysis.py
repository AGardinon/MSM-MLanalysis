#!

import numpy as np
from tqdm import tqdm
import argparse

from deeptime.decomposition import TICA
from deeptime.clustering import KMeans, RegularSpace
import deeptime.markov.tools as mrktool
import deeptime.markov as markov

from .msmanalysis.utils import misc

# -----------------------------------------------------------------
#
#   Functions
#
# -----------------------------------------------------------------

from deeptime.util.data import TrajectoryDataset, TrajectoriesDataset

# - init timeseries
def build_timeseries(trajfile, lagtime=1):
    if isinstance(trajfile, str):
        data = np.load(trajfile)
    else:
        data = trajfile
    # for multiparticles
    if len(np.shape(data)) > 2:
        print(f"Multi particles data - {np.shape(data)}")
        traj = [np.vstack([ts[li] for ts in data]) for li in np.arange(0,data.shape[1])]
        # create time series
        traj_list = [TrajectoryDataset(lagtime=lagtime,
                                       trajectory=ts.astype(np.float64))
                     for ts in traj]
        dataset = TrajectoriesDataset(traj_list)

    else:
        dataset = TrajectoryDataset(lagtime=lagtime, trajectory=data.astype(np.float64))

    return dataset


# -----------------------------------------------------------------
#
#   Input
#
# -----------------------------------------------------------------

space_discretisation_modes = dict(
    Kmeans = KMeans,
    RegularSpace = RegularSpace
)

msm_estimator_modes = dict(
    MaximumLikelihoodMSM = markov.msm.MaximumLikelihoodMSM,
)

# -----------------------------------------------------------------
#
#   MAIN
#
# -----------------------------------------------------------------


def main(config):

    # --- Input config

    # define the workspace and files from config
    descr_dir_path = config.descriptor_file_path
    descr_files = misc.get_files_from(folder=descr_dir_path,
                                      ew=config.descriptor_file_extension,
                                      verbose=True)
    
    # tica config params
    tica_analysis_dict = config.tica_analysis_dict
    tica_analysis_dict['tica_param'].update(lagtime=None)

    # space discretisation config params
    clustering_analysis_dict = config.clustering_analysis_dict
    _clMethod = clustering_analysis_dict['method']
    clustering_analysis_dict.update(method=space_discretisation_modes[_clMethod])
    clustering_analysis_dict['param_dict'].update(progress=tqdm)

    # msm config params
    msm_analysis_dict = config.msm_analysis_dict
    _msmEstimator = msm_analysis_dict['estimator']
    msm_analysis_dict.update(estimator=msm_estimator_modes[_msmEstimator])


    # --- Cycle for each descriptor file

    for file in descr_files:
        print(f"\n# --- Analysis:\n{file}\n")

        # result dir
        result_dir_name = 'result_'+file.replace('.npy', '')+'/'
        result_folder = misc.py_mkdir(path=descr_dir_path, 
                                    folder_name=result_dir_name)

        tica_result_dir_name = '1.tica/'
        tica_folder = misc.py_mkdir(path=result_folder, 
                                    folder_name=tica_result_dir_name)
        
        regspace_result_dir_name = '2.cluster/'
        cluster_folder = misc.py_mkdir(path=result_folder, 
                                    folder_name=tica_result_dir_name)
        
        msm_result_dir_name = '3.msm/'
        msm_folder = misc.py_mkdir(path=result_folder, 
                                folder_name=msm_result_dir_name)  
        
        # vampscore_dict[file.replace('npy', '')] = list()
        
        # --- Cycle for each lag times

        for lagt in tqdm(tica_analysis_dict['lag_time_scan'], desc='tICA lag scans'):
            
            print(f'- tICA analysis/
            descr file {file}')
            # dict update for the lag time
            tica_analysis_dict['tica_param'].update(lagtime=lagt)
            print(f"lag= {lagt}t*")
            # build the descr timeseries
            timeseries = build_timeseries(descr_dir_path+file, lagtime=lagt)

            # --- train tICA model
            tica = TICA(**tica_analysis_dict['tica_param'])
            model = tica.fit_from_timeseries(timeseries).fetch_model()
            # predict
            X = model.transform(timeseries.trajectories)
            print(f"- tICA Outputs:/
            Ncomopnents= {X.shape[2]} (max:{len(model.cumulative_kinetic_variance)})/
            Vamps-2 score= {model.score(r=2)}/n")
            # !!! save relevant tica files
            # vampscore_dict[file.replace('npy', '')].append(model.score(r=2))
            # save files
            np.save(tica_folder+f'tica_lag{lagt}', X)
            np.savetxt(tica_folder+f'tica_lag{lagt}.kvar', model.cumulative_kinetic_variance)

            # --- space fragmentation, via clustering
            print(f'\n- Space segmentation step')
            estimator = clustering_analysis_dict['method'](**clustering_analysis_dict['param_dict'])
            # !!!
            XX = np.concatenate(X)
            clustering = estimator.fit(XX).fetch_model()
            labels = clustering.transform(XX)
            np.savetxt(cluster_folder+f'cluster_lag{lagt}_nc{len(np.unique(labels))}.labels',
                    labels)
            np.savetxt(cluster_folder+f'cluster_lag{lagt}_nc{len(np.unique(labels))}.centers',
                    clustering.cluster_centers)
            # - freeing memory (does this work?)
            del X
            del XX
            del tica


            # --- MSM analysis

            print(f'\n- MSM analysis')
            msm_estimator = msm_analysis_dict['estimator'](**msm_analysis_dict['estimator_param'])

            msm_experimet = msm_estimator.fit_from_discrete_timeseries(
                discrete_timeseries=labels.reshape(msm_analysis_dict['msm_analysis_param']['n_particles']),
                lagtime=lagt,
                count_mode=msm_analysis_dict['count_mode']
            ).fetch_model()
            print(f'/
            Number of states: {msm_experimet.n_states}')
            np.savetxt(msm_folder+f'msm_lag{lagt}_states{msm_experimet.n_states}.trmtrx',
                    msm_experimet.transition_matrix)
            np.savetxt(msm_folder+f'msm_lag{lagt}.eigenval', msm_experimet.eigenvalues())

            # - PCCA analysis
            print(f'\n- PCCA+ analysis')
            if msm_analysis_dict['msm_analysis_param']['eigenval_treshold']:
                eigenval_treshold = msm_analysis_dict['msm_analysis_param']['eigenval_treshold']
                print(f'set eingeval treshold to {eigenval_treshold}')
                cg_states = len([e for e in msm_experimet.eigenvalues() if e >= eigenval_treshold])
                print(f'total number of MSM-CG states: {cg_states}')
            
            if msm_analysis_dict['msm_analysis_param']['fixed_eigensates']:
                print(f'set a fixed number of eigenstaes')
                cg_states = msm_analysis_dict['msm_analysis_param']['fixed_eigensates']
                print(f'total number of MSM-CG states: {cg_states}')

            pcca = msm_experimet.pcca(n_metastable_sets=cg_states)
            msm_cg_mtrx = pcca.coarse_grained_transition_matrix
            pcca_labels = np.argmax(pcca.memberships[labels], axis=1)

            np.savetxt(msm_folder+f'pcca_lag{lagt}.trmtrx', msm_cg_mtrx)
            np.savetxt(msm_folder+f'pcca_lag{lagt}.labels', pcca_labels)
            np.savetxt(msm_folder+f'pcca_lag{lagt}.eigenval', mrktool.analysis.eigenvalues(msm_cg_mtrx))


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='config', type=str, help='Config file')
    args = parser.parse_args()
    main(misc.parse_config(args.config))
        
