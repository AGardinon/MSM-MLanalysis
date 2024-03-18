# -------------------------------------------------- #
# QUIP tools - Handler for quipd descriptors and 
#              related objects
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import copy
from .utils.misc import parse_config

# -------------------------------------------------- #
# --- QUIP descriptor tools


def soap_to_string(param_dict: dict) -> str:
    """Interprets the quip-SOAP praramenters and
    returns the quip formatted input string.

    :param param_dict: quip-SOAP parameters dictionary.
    :type param_dict: dict
    :return: quip-SOAP input string.
    :rtype: str
    """
    # str building
    descriptor_str = 'soap '
    for param, value in param_dict.items():
        if isinstance(value, list):
            str_entry = param+'={'+' '.join(map(str,value))+'} '
        else:
            str_entry = param+'='+str(value)+' '

        descriptor_str += str_entry

    return descriptor_str


def soapturbo_to_string(param_dict: dict) -> str:
    """Interprets the quip-TurboSOAP praramenters and
    returns the quip formatted input string.

    :param param_dict: quip-TruboSOAP parameters dictionary.
    :type param_dict: dict
    :return: quip-TurboSOAP input string.
    :rtype: str
    """
    # str building
    descriptor_str = ''
    for Z in param_dict['Zs']:
        param_dict["central_index"] = param_dict["species_Z"].index(Z) + 1
        quip_str_tmp = _turbo_to_str_helper(original_param=param_dict, 
                                            multi_param=param_dict['multi'])
        descriptor_str += quip_str_tmp+' '

    return descriptor_str


def _turbo_to_str_helper(original_param: dict, 
                         multi_param: dict) -> str:
    """Helper function to convert input parameters to
    quip-TurboSoap string.

    :param original_param: quip-TruboSOAP parameters dictionary.
    :type original_param: dict
    :param multi_param: quip-TruboSOAP multi cutoff parameters dictionary.
    :type multi_param: dict
    :return: quip-TurboSOAP input string.
    :rtype: str
    """
    params = copy.deepcopy(original_param)
    params["n_species"] = len(params["species_Z"])
    # init Z-wise strings
    Zstr = "{"
    for Z in params['species_Z']:
        Zstr += str(Z) + " "
    Zstr = Zstr[:-1]+"}"
    # updating param dict with Z-wise strings
    params['species_Z'] = Zstr
    # building of the individua quip strings
    for key, value in multi_param.items():
        quip_string = "{"
        for _ in range(0, params['n_species']):
            quip_string += str(value) + " "
            params[key] = quip_string[:-1] + "}"
    quip_string = 'soap_turbo '
    for key, value in params.items():
        if key != 'Zs':
            quip_string += key + "=" + str(value) + " "
    
    return quip_string


class QUIPtools:
    """Class to parse and prepare the input dictionary for
    the quip SOAP / turbo-SOAP analysis.
    """

    def __init__(self, 
                 method: str =None,
                 descr_dict: dict =None) -> None:
        self._method = method
        self.descriptor_str = None
        if isinstance(descr_dict, dict):
            self._descr_dict = descr_dict
        elif isinstance(descr_dict, str):
            try:
                self._descr_dict = parse_config(descr_dict, 
                                                toNameSpace=False)
            except:
                raise NameError("File given has to be .json or .toml.\n"
                                "(Check the path as well).\n")  
        else:
            print("!!! Warning.\n"
                  "Descriptor dictionary not set.\n")
            self._descr_dict = descr_dict
        pass

    @property
    def method(self) -> str:
        """Quip SOAP method.

        :return: method chosen.
        :rtype: _type_
        """
        return self._method
    
    @method.setter
    def method(self, 
               value: str) -> None:
        """Set the method for the Quip-SOAP evaluation.

        :param value: method string.
        :type value: str
        """
        self._method = value
        pass

    @property
    def descriptorDict(self) -> dict:
        """Quip-SOAP parameters dictionary.

        :return: Quip-SOAP parameters dictionary.
        :rtype: dict
        """
        return self._descr_dict

    @descriptorDict.setter
    def descriptorDict(self, 
                       value: dict) -> None:
        """Set the Quip-SOAP parameter dictionary.

        :param value: Quip-SOAP parameters dictionary.
        :type value: dict
        """
        self._descr_dict = value
        pass

    @property
    def getString(self) -> str:
        """Generate the Quip-SOAP descriptor string.

        :raises ValueError: Quip-SOAP method type not recognised.
        :return: descriptor string for the selected method type.
        :rtype: str
        """
        if self._method == 'soap':
            self.descriptor_str = soap_to_string(param_dict=self._descr_dict)
        elif self._method == 'soap_turbo':
            self.descriptor_str = soapturbo_to_string(param_dict=self._descr_dict)
        else:
            raise ValueError("Quip-SOAP method not recognised.")
        
        return self.descriptor_str