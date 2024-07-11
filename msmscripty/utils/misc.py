# -------------------------------------------------- #
# Computes - miscellaneous stuffs
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

# import

# -------------------------------------------------- #
# --- custom exception
class WrongConfigFormat(Exception):
    pass

# -------------------------------------------------- #
# --- get todays date string
from datetime import date

def todayDate(date_format="%d%b%Y"):
    """
    Gets you todays date in according to the format choosen
    Default: DayMonthYear (ex. 13Jan1992)
    """
    today = date.today()
    return today.strftime(date_format)

# -------------------------------------------------- #
# --- parse dict file .json .toml
import json
import toml
from types import SimpleNamespace

def parse_config(filename: str, 
                 toNameSpace: bool =False) -> dict:
    """Parse a file with the .json or .toml extension
    to a py dictionary.

    :param filename: file absolute path.
    :type filename: str
    :param toNameSpace: return Namespace, defaults to False
    :type toNameSpace: bool, optional
    :raises WrongConfigFormat: wrong file extension.
    :return: parsed dictionary.
    :rtype: dict
    """
    with open(filename, 'r') as f:
        if filename.endswith('.json'):
            _config = json.load(f)
        elif filename.endswith('.toml'):
            _config = toml.load(f)
        else:
            raise WrongConfigFormat(
                f"Format of the '{filename}' is not supported. "
                "Available formats: .json, .toml"
            )
        if toNameSpace:
            return SimpleNamespace(**_config)
        else:
            return _config

# -------------------------------------------------- #
# --- file and folders handling
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_files_from(folder: str, ew: str=None, sw: str=None, verbose: bool=True):
    """Simple function to select files from a location
    with possible restraints.
    folder : file str location
    ew : "endswith" str selection
    sw : "startswith" str selection
    """
    file_list = list()
    # file selection following the constraints
    for entry in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, entry)):
            if ew:
                if entry.endswith(ew):
                    file_list.append(entry)
            elif sw:
                if entry.startswith(sw):
                    file_list.append(entry)
            else:
                file_list.append(entry)
    # sorting of the files :)
    file_list.sort(key=natural_keys)
    if verbose:
        print(f"Files:\n{file_list}, ({len(file_list)})")
    return file_list


def number_from_string(string, target, separator='_'):
    """Simple function to extract number from a string
    in a specific position"""
    chunks = string.split(separator)
    lag_string = [s for s in chunks if target in s][0]
    return int(lag_string.replace(target,''))


# - creating a directory
def py_mkdir(path: str, folder_name: str, overwrite=False) -> str:
    new_dir_ = path + folder_name
    # check for syntax and add '/' if not present
    if not new_dir_[-1] == '/':
        new_dir = new_dir_ + '/'
    else:
        new_dir = new_dir_
    # making the folder
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"Created folder\n{new_dir}")
    else:
        last_dir = [f for f in os.listdir(path) if folder_name in f][-1]
        if not overwrite:
            if '_copy' in last_dir:
                counter = number_from_string(last_dir, 'copy')
                new_dir = last_dir.replace(f'_copy{counter}', f'_copy{counter+1}')
            else:
                new_dir = last_dir + '_copy0/'
            os.makedirs(new_dir)
            print(f"Folder already exist!")
            print(f"Created copy ... {new_dir}")
        else:
            pass

    return new_dir


# -------------------------------------------------- #
# --- Custom decorators
import time
import functools


def my_timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"{run_time:.4f}s\n")
        return value
    return wrapper_timer