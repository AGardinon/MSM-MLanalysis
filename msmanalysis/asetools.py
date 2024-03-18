# -------------------------------------------------- #
# ASE tools - trajectories handler tool in ASE
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
from tqdm import tqdm
from typing import Union, List
from ase.io import read, write
from .atomstools import *
from .utils import misc

# -------------------------------------------------- #
# --- Universe 

class Universe:

    @misc.my_timer
    def __init__(self, 
                 projectName: str, 
                 trajPath: str,
                 **kwargs):
        """ASE universe tool (similar to the MDA one).
        Optional kwargs are:
        - rcutCorrection: dict (Default=1.0)
        - moleculeNames: list (Default='molXXX')

        :param projectName: Name of the project (aka the system).
        :type projectName: str
        :param trajPath: Path for the traj file.
        :type trajPath: str
        """
        # --- Universe info
        # -
        self.projectName = projectName
        self.trajPath = trajPath
        self.rcutCorrection = None
        self.moleculeNames = None
        # --- init the project
        # -
        self._get_config
        try:
            self.rcutCorrection = kwargs['rcutCorrection']
            print(f"rcut correction: {self.rcutCorrection}\n")
        except:
            print("!!! Warning: `rcutCorrection` not set !!!\n"
                  "Default values will be used, "
                  "this might cause artifacts in the molecules detection."
                  )
            # creates the default one
            self.rcutCorrection = {k:1.0 for k in self.symbols}
            print(f"rcutCorrection = {self.rcutCorrection}\n")
        # -
        print("Searching for molecules in the system ...\n")
        try:
            self.moleculeNames = kwargs['moleculeNames']
            print(self.moleculeNames)
        except:
            print("!!! Warning: `moleculeNames` not set !!!\n"
                  "Default molecules names will be used\n")
        self.find_molecs(mol_name=self.moleculeNames)
        self.get_mol_info
        # - Update the dictionary
        self.projectDictionary = dict(
            projectName = self.projectName,
            trajPath = self.trajPath,
            rcutCorrection = self.rcutCorrection,
            moleculeNames = self.moleculeNames,
            moleculeFormulas = self.moleculeFormulas,
            molIDs = self.molIDs,
            molSym = self.molSym
        )
        # -
        print("<end>")
            
    @property
    def _get_config(self):
        """Reads the first frame to get
        information about the system, 
        effectively building the Universe.
        """
        print("Gathering the Universe ...\n")
        self._at0 = read(filename=self.trajPath, index='0')
        self.symbols = np.unique(self._at0.symbols)
        print(f"Total atoms: {len(self._at0.symbols)}\n"
              f"Atom types: {self.symbols}\n"
              )
        pass


    def set_rcut_correction(self, 
                            rcut_dict: dict) -> dict:
        """Allow the setting of a rcut dictionary for
        the atoms types in the system.

        :param rcut_dict: LJ radius correction parameter (1.0 = unchanged).
        :type rcut_dict: dict
        :raises ValueError: Only defined if the atoms are in the system.
        :return: Dictionary containing kw: atoms and args: rcut correction.
        :rtype: dict
        """
        if set(rcut_dict.keys()) == set(self.symbols):
            self.rcutCorrection = rcut_dict
            return print(f"Updated rcutCorrection = {self.rcutCorrection}\n")
        else:
            raise ValueError("Given atom types does not match with the system types.\n")

    @misc.my_timer
    def find_molecs(self, 
                    mol_name: list = None) -> None:
        """Finds the molecules as whole defined by the 
        current LJ rcut.

        :param mol_name: list of names in the same order of appearence.
        :type mol_name: list
        """
        self.moleculeFormulas = get_chemFormulas(self._at0, 
                                                 fct=self.rcutCorrection)
        self.moleculeNames = list(self.moleculeFormulas.values())
        print(f"Uniques molecules found: {len(self.moleculeFormulas)}")
        if mol_name:
            self.moleculeFormulas = self.set_mol_names(mol_name=mol_name)
        print(f"Molecules found: {self.moleculeFormulas}")
        pass
    

    def set_mol_names(self, 
                      mol_name: list) -> dict:
        """Set new specific names for molecules found by
        Universe.find_molecs().

        :param mol_name_list: List of names in the same order of appearence.
        :type mol_name_list: list
        :raises ValueError: The molecules needs to be defined before this operation.
        :return: Dictionary with as args: custom mol names
        :rtype: dict
        """
        self.moleculeNames = mol_name
        if self.moleculeFormulas:
            if len(self.moleculeFormulas) == len(mol_name):
                for key,val in zip(self.moleculeFormulas.keys(),mol_name):
                    self.moleculeFormulas[key] = val
                return self.moleculeFormulas
            else:
                raise ValueError("The list of molecules provided is not compatible,"
                                 "(the system has {len(self.moleculeFormulas)} molecs)")
        else:
            raise ValueError("The molecules have to be found before setting the name.")

    @property
    def get_mol_info(self):
        """TODO
        """
        # ---
        print("Computing MolIDs ...")
        self.molIDs = get_molIDs(at=self._at0,
                                 fct=self.rcutCorrection)
        # ---
        print("Computing MolSymbols ...")
        self.molSym = get_molSym(at=self._at0,
                                 molIDs=self.molIDs,
                                 mol_name=self.moleculeFormulas)
        pass


# -------------------------------------------------- #
# --- trajtools

# should be improved with proper exception creation!
def frame_type_checker(frame_value : Union[tuple, list]) -> None:
    try:
        if isinstance(frame_value, (tuple, str)):
            if frame_value == str and not frame_value == 'all':
                raise ValueError("ValueError: string values supported is only 'all'.")
        else:
            raise ValueError("ValueError: only tuple and string values are accepted.")
    except ValueError as error:
        print("!!! " + repr(error)) 


class ASEtraj(Universe):
    """Child class for handling trajectories and doing operation whitin them.

    :param Universe: sets the Ase Universe as a baseline to compute trajectory 
    related quantities.
    :type Universe: class
    """
    def __init__(self, 
                 projectName: str, 
                 trajPath: str, 
                 frameRange : Union[tuple, list] = 'all',
                 **kwargs):
        super().__init__(projectName, trajPath, **kwargs)

        # - specific attributes
        _ = frame_type_checker(frame_value=frameRange)
        self._frameRange = frameRange
        pass

    @property
    def frameRange(self) -> Union[tuple, list]:
        """Get the frames used for the analysis.

        :return: Frame value range.
        :rtype: Union[tuple, list]
        """
        return self._frameRange
    
    @frameRange.setter
    def frameRange(self, 
                   value: Union[tuple, list]) -> None:
        """Allow to set different frames values for the analysis.

        :param value: Frame value range.
        :type value: Union[tuple, list]
        """
        _ = frame_type_checker(frame_value=value)
        self._frameRange = value
        pass
    
    @misc.my_timer
    def read(self, 
             frameRange: Union[tuple, list] = None,
             Zshift: Tuple[str, list] = None,
             COM: bool = False,
             save_to_file: str = None) -> List[ase.ase.Atoms]:
        """Read the selected number of the trajectory provided in the project.

        :param frameRange: frame range , defaults to None
        :type frameRange: Union[tuple, list], optional
        :param Zshift: Z numbers shift for atoms of a single molecule, defaults to None
        :type Zshift: Tuple[str, list], optional
        :param save_to_file: _description_, defaults to None
        :type save_to_file: str, optional
        :return: Ase "database" of frame in list form.
        :rtype: List[ase.ase.Atoms]
        """
        if frameRange:
            self.frameRange = frameRange
        else:
            pass
        # ---
        # traj reader
        if COM:
            ase_db = self._readCOM
        else:
            ase_db = self._read
            # zshift
            if Zshift:
                newZ = ZnumberShift(Znumbers=self._at0.numbers,
                                    molSymbols=self.molSym,
                                    molIDs=self.molIDs,
                                    to_shift=Zshift)
                for snap in tqdm(ase_db, desc='Applying Z shift'):
                    snap.numbers = newZ
        # # ---
        # # file saver
        # # may be not needed
        # if save_to_file:
        #     print("Not yet done.")

        return ase_db

    @property
    def _read(self) -> List[ase.ase.Atoms]:
        """Read a given trajectory using ase tools.

        :return: ase atoms databese of frames.
        :rtype: List[ase.ase.Atoms]
        """
        if isinstance(self._frameRange, tuple):
            # frame tuple complition
            try:
                b,e,s = self._frameRange
            except:
                s = 1
                b,e = self._frameRange
            print("Reading traj:\n"
                  f"Begin: {b} | End: {e} | Stride: {s}")
            return read(self.trajPath, index=f'{b}:{e}:{s}')
        else:
            return read(self.trajPath, index=f':')
        
    @property
    def _readCOM(self) -> List[ase.ase.Atoms]:
        """Compute the center of mass of a given ase trajectory using
        the molecule information of the project.

        :return: ase atoms databese of frames.
        :rtype: List[ase.ase.Atoms]
        """
        return center_of_mass(ase_db=self._read,
                              molSymbols=self.molSym,
                              molIDs=self.molIDs)

