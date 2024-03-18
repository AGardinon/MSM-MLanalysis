# -------------------------------------------------- #
# Descriptors tools 
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import quippy
from .utils import misc
from .quipTools import QUIPtools


class QUIP(QUIPtools):
    """Class for computing quippy SOAP and turbo SOAP.

    :param QUIPtools: helper class to parse input.
    :type QUIPtools: class
    """

    def __init__(self, 
                 method: str = None, 
                 descr_dict: dict = None) -> None:
        super().__init__(method, descr_dict)

        try:
            self.descriptor_str = self.getString
        except:
            raise ValueError("Need to define both method and desct_dict parameters.\n"
                             f"methdo: {self.method}\n"
                             f"descr_dict: {self.descriptorDict}\n")
        pass

    @misc.my_timer
    def fit(self, 
            ase_db: list) -> list:
        print(f"Computing quippy {self.method} ...")
        return self._fit(ase_db=ase_db)


    def _fit(self, 
             ase_db: list):
        """Generate the quippy object and compute the descriptor.
        More information available at the official quippy doc page.

        :param ase_db: _description_
        :type ase_db: list
        :return: _description_
        :rtype: _type_
        """
        quippy_object = quippy.descriptors.Descriptor(self.descriptor_str)
        return quippy_object.calc_descriptor(ase_db)
