# -------------------------------------------------- #
# Computes - trajectory module
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
from tqdm import tqdm

# -------------------------------------------------- #
# --- Coordinate unwrapper

class XYZunwrapper:
    """Class to Unwrap XYZ trajectories."""

    def __init__(self, 
                 method: str):
        try:
            self._method = self._methods_dict[method]
        except:
            raise NameError("Chosen method not available.\n"
                            f"Choose from {self._methods_dict.keys()}")
    pass

    @property
    def _methods_dict(self):
        methods_function_dict = dict(
            heuristic = heuristic_unwrapping,
            displacement = displacement_unwrapping,
            hybrid = hybrid_unwrapping,
        )
        return methods_function_dict
    
    @property
    def method(self):
        print(f"Chosen method: {self._method.__doc__}")
        return self._method

    @method.setter
    def method(self, 
               value: str):
        """Set the unwrapping method.

        :param value: unwrapping method.
        :type value: str
        :raises NameError: if the name is not supported.
        """
        try:
            self._methods_dict[value]
            print(f"Chosen method: {self._methods_dict[value].__doc__}")
        except:
            raise NameError("Chosen method not available.\n"
                            f"Choose from {self._methods_dict.keys()}")        
        
        self._method = self._methods_dict[value]
        pass
    

    def fit(self, 
            xyz: np.ndarray, 
            box: list) -> np.ndarray:
        """Applies the unwrapping mehtod chosen to a XYZ trajectory.

        :param xyz: xyz coordinates trajectory.
        :type xyz: np.ndarray
        :param box: extent of the simulation box.
        :type box: list
        :return: unwrapped coordinates trajectory.
        :rtype: np.ndarray
        """
        return self._method(w=xyz, 
                            box=box)

# -------------------------------------------------- #

# -- Unwrap functions
def heuristic_unwrapping(w: np.ndarray,
                         box: list) -> np.ndarray:
    """Heuristic unwrapping.

    """
    # init the coordinates
    u = np.empty((len(w),3))
    # set the same starting point
    u[0] = w[0]
    # computing the diff of the w traj
    difw = np.diff(w, axis=0)
    # ---
    # Eq. 1 Heuristic method
    for i,dw in enumerate(difw):
        u[i+1] = w[i+1]-np.floor((w[i+1]-u[i])/box[i+1]+0.5)*box[i+1]
    return u


def displacement_unwrapping(w: np.ndarray,
                            box: list) -> np.ndarray:
    """Displacement unwrapping.

    """
    # init the coordinates
    u = np.empty((len(w),3))
    # set the same starting point
    u[0] = w[0]
    # computing the diff of the w traj
    difw = np.diff(w, axis=0)
    # ---
    # Eq 2. Displacemnet method
    for i,dw in enumerate(difw):
        u[i+1] = u[i]+(w[i+1]-w[i])-np.floor((w[i+1]-w[i])/box[i+1]+0.5)*box[i+1]
    return u


def hybrid_unwrapping(w: np.ndarray,
                      box: list) -> np.ndarray:
    """Hybrid unwrapping.

    """
    # init the coordinates
    u = np.empty((len(w),3))
    # set the same starting point
    u[0] = w[0]
    # computing the diff of the w traj
    difw = np.diff(w, axis=0)
    # ---
    # Eq. 12 Hybrid mehtod
    for i,dw in enumerate(difw):
        u[i+1] = u[i]+(w[i+1]-w[i])-np.floor((w[i+1]-w[i])/box[i+1]+0.5)*box[i+1]-np.floor((w[i]-u[i])/box[i]+0.5)*(box[i+1]-box[i])
    return u

# -------------------------------------------------- #