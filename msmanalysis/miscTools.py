#!
import numpy as np
import os
import re

# -------------------------------------------------- #
# --- Files

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_filesEW(folder, extension, verbose=True):
    file_list = list()
    for entry in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, entry)):
            if entry.endswith(extension):
                file_list.append(entry)
    file_list.sort(key=natural_keys)
    if verbose:
        print(f"Files:\n{file_list}, ({len(file_list)})")
    return file_list


def numberFromString(string, target, separator='_'):
    chunks = string.split(separator)
    lag_string = [s for s in chunks if target in s][0]
    return int(lag_string.replace(target,''))


# -------------------------------------------------- #
# --- handling folders

# - creating a directory
def mkdir(path, folder_name, overwrite=False):
    new_dir_ = path + folder_name
    if not new_dir_[-1] == '/':
        new_dir = new_dir_ + '/'
    else:
        new_dir = new_dir_

    # makes the folder
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"Created folder\n{new_dir}")
    else:
        if not overwrite:
            counter = 0
            while os.path.exists(new_dir):
                # print(f"Folder already exist!\n{SOAP_DIR}")
                if counter == 0:
                    new_dir = new_dir[:-1] + '_copy' + str(counter) + '/'
                else:
                    new_dir = new_dir[:-2] + '_copy' + str(counter) + 'copy/'
                counter += 1
            os.makedirs(new_dir)
            print(f"Folder already exist!")
            print(f"Created copy ... {new_dir}")
        else:
            pass

    return new_dir


# -------------------------------------------------- #
# --- Deeptime utilities
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