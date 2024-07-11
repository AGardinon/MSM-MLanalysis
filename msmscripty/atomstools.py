# -------------------------------------------------- #
# ASE tools - atoms tools module
# 
#
# AUTHOR: Andrea Gardin, Ioan-Bogdan Magdau
# -> I made use of just a few of ibm tools;
# -> the complete tools list can be foun in 
# -> ibm repo @ https://github.com/imagdau/Python-Atoms
# -------------------------------------------------- #

import numpy as np
from tqdm import tqdm
import ase
from ase import Atoms, neighborlist
from scipy import sparse
from typing import Union, Tuple, List
from .utils import traj, misc

# -------------------------------------------------- #
# --- Atom tools

@misc.my_timer
def get_molIDs(at: ase.ase.Atoms, 
               fct: Union[float, ase.ase.Atoms]) -> list:
    """Computes the molecules IDs, a numerical index that differentiate
    each uniques molecule.

    :param at: atoms configuration.
    :type at: ase.ase.Atoms
    :param fct: rcut correction, defaults to 1.0 (i.e., no correction)
    :type fct: Union[float, ase.ase.Atoms]
    :return: list of indexes of the molecules of the atomic configuration.
    :rtype: list
    """
    # doc @ https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html
    _, molID = get_connected_atoms(at=at, 
                                   fct=fct)
    at.arrays['molID'] = molID
    return molID

@misc.my_timer
def get_molSym(at: ase.ase.Atoms,
               molIDs: np.ndarray,
               mol_name: dict) -> list:
    """Get the molecules symbols as they appear in the system xyz configuration.

    :param at: atoms configuration.
    :type at: ase.ase.Atoms
    :param molIDs: molecules IDs.
    :type molIDs: np.ndarray
    :param mol_name: names association to chemical formulas.
    :type mol_name: dict
    :return: list of molecules of the atomic configuration.
    :rtype: list
    """
    molSym = list()
    for m in np.unique(molIDs):
        mol = at[molIDs == m]
        molSym.append(mol_name[mol.symbols.get_chemical_formula()])
    print(f"Total numner of molecules: {len(molSym)}")
    return molSym


def get_chemFormulas(at: ase.ase.Atoms, 
                     fct: Union[float, dict] = 1.0) -> dict:
    """Returns a dictionary with the whole molecules inside the
    system (dependent on the rcut correction, fct parameter).

    :param at: atomic configuration in ase format.
    :type at: ase.ase.Atoms
    :param fct: rcut correction, defaults to 1.0 (i.e., no correction)
    :type fct: Union[float, dict], optional
    :return: whole molecules dictionary.
    :rtype: dict
    """
    _, molID = get_connected_atoms(at, fct)
    chemFormulas_list = list()
    for m in np.unique(molID):
        mol = at[molID==m]
        chemFormulas_list.append(mol.symbols.get_chemical_formula())
    chemFormulas_dict = dict()
    for i,chem in enumerate(np.unique(chemFormulas_list)):
        chemFormulas_dict[chem] = f'mol{i+1}'
    return chemFormulas_dict
    
    
# - computes molID for single config, not adding molID to atoms.arrays
def find_molecules(at: ase.ase.Atoms, 
                   fct: Union[float, dict]):
    """Computes the whole molecules based on the LJ cutoff values of each 
    atoms in the configuration.

    :param at: ase atom configuration.
    :type at: ase.ase.Atoms
    :param fct: scaling parameters for the LJ cutoffs.
    :type fct: Union[float, dict]
    :return: _description_
    :rtype: _type_
    """
    _, molID = get_connected_atoms(at, fct)
    Natoms, Nmols = np.unique(np.unique(molID, return_counts=True)[1], return_counts=True)
    return Nmols, Natoms


def get_connected_atoms(at: ase.ase.Atoms, 
                        fct: Union[float, dict]) -> Tuple[int, np.ndarray]:
    """Computes connected atoms based on the natural LJ cutoff range.
    Doc @ https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html

    :param at: ase atom configuration.
    :type at: ase.ase.Atoms
    :param fct: scaling parameters for the LJ cutoffs.
    :type fct: Union[float, dict]
    :return: _description_
    :rtype: Tuple[int, np.ndarray]
    """
    cutOff = modif_natural_cutoffs(at, fct)
    nbLst = neighborlist.NeighborList(cutOff, 
                                      self_interaction=False, 
                                      bothways=True)
    nbLst.update(at)
    conMat = nbLst.get_connectivity_matrix(sparse=True)
    Nmol, molID = sparse.csgraph.connected_components(conMat)
    return Nmol, molID


def modif_natural_cutoffs(at: ase.ase.Atoms,
                          fct: Union[float, dict]) -> dict:
    """Modifies the natural cutoff of the LJ interactions.

    :param at: ase atom configuration.
    :type at: ase.ase.Atoms
    :param fct: newly defined scaling parameters for the LJ cutoffs.
    :type fct: Union[float, dict]
    :raises NameError: only accepts int and dictionary values.
    :return: newly defined dictionary containing the LJ scaling values for each atoms.
    :rtype: dict
    """
    if type(fct) is int or type(fct) is float:
        return neighborlist.natural_cutoffs(at, mult=fct)
    elif type(fct) is dict:
        cutOff = neighborlist.natural_cutoffs(at, mult=1)
        return [ctf*fct[el] for ctf, el in zip(cutOff, at.get_chemical_symbols())]
    else:
        raise NameError('Unknown fct type '+str(type(fct)))


def ZnumberShift(Znumbers: np.ndarray, 
                 molSymbols: list,
                 molIDs: list,
                 to_shift: Tuple[str, list]) -> np.ndarray:
    """_summary_

    :param Znumbers: _description_
    :type Znumbers: np.ndarray
    :param molSymbols: _description_
    :type molSymbols: list
    :param molIDs: _description_
    :type molIDs: list
    :param to_shift: _description_
    :type to_shift: Tuple[str, list]
    :return: _description_
    :rtype: np.ndarray
    """
    mol_to_shift, Z_to_shift = to_shift
    dummy = np.max(Znumbers)
    shifted_Znumbers = Znumbers.copy()
    #
    for idx, mol in enumerate(molSymbols):
        if mol == mol_to_shift:
            mask = molIDs == idx
            #
            for i, tgt in enumerate(mask):
                if tgt:
                    if shifted_Znumbers[i] in Z_to_shift:
                        shifted_Znumbers[i] += dummy
                    else:
                        pass
    return shifted_Znumbers


def center_of_mass(ase_db: List[ase.ase.Atoms],
                   molSymbols: list,
                   molIDs: list) -> List[ase.ase.Atoms]:
    """Computes the COM of a given ase atoms databas of frames.

    :param ase_db: ase atoms database.
    :type ase_db: List[ase.ase.Atoms]
    :param molSymbols: list of molecule-wise symbols as they appear in the frame configuration.
    :type molSymbols: list
    :param molIDs: list of molecule-wise ids as they appear in the frame configuration.
    :type molIDs: list
    :return: ase atoms database containitng the COM position.
    :rtype: List[ase.ase.Atoms]
    """
    ase_db_com_list = list()
    for at in tqdm(ase_db, desc='Computing COM:'):
        mol_com_tmp = list()
        for m in np.unique(molIDs):
            mol = at[molIDs==m] #copy by value
            mass = mol.get_masses()
            cm = np.sum(mol.positions*mass.reshape(-1,1), axis=0)/np.sum(mass)
            mol_com_tmp.append(cm)
        new_com_at = Atoms(positions=np.array(mol_com_tmp), pbc=True, cell=at.cell)
        new_com_at.arrays['molSym'] = np.array(molSymbols)
        ase_db_com_list.append(new_com_at)
    return ase_db_com_list

@misc.my_timer
def ase_mol_unwrap(ase_db: List[ase.ase.Atoms], 
                   molecule_to_unwrap: List[str], 
                   molSymbols: list,
                   method: str) -> dict:
    """_summary_

    :param ase_db: _description_
    :type ase_db: List[ase.ase.Atoms]
    :param molecule_to_unwrap: _description_
    :type molecule_to_unwrap: str
    :param molSymbols: _description_
    :type molSymbols: list
    :param method: _description_
    :type method: str
    :return: _description_
    :rtype: dict
    """
    # - init unwrapping
    unwrap_obj = traj.XYZunwrapper(method=method)
    print(f"Chosen method: {unwrap_obj._method.__doc__}\n")
    unwrap_coord_dict = dict()
    # - unwrapping quantitities
    box_values = [at.cell[0][0] for at in ase_db]
    mol_to_unwrap_idxs = [
        [i for i,s in enumerate(molSymbols) if s == mol] 
        for mol in molecule_to_unwrap
    ]
    # unwrapping
    for i,mask in enumerate(mol_to_unwrap_idxs):
        unwrap_coord_dict[molecule_to_unwrap[i]] = list()
        for idx in tqdm(mask, desc=f"Unwrapping: {molecule_to_unwrap[i]}"):
            # get the wrapped coordinates
            wrapped_coords_tmp = np.array([at[idx].position for at in ase_db])
            # get the unwrapped coordinates
            unwrap_coord_dict[molecule_to_unwrap[i]].append(
                unwrap_obj.fit(xyz=wrapped_coords_tmp,
                               box=box_values)
            )
    return unwrap_coord_dict
