# Copyright 2019 The Optcom Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""".. moduleauthor:: Sacha Medaer"""

import sys

OPTCOM_LOG_FILENAME = 'optcom.log'
def set_log_filename(filename: str):
    if (isinstance(filename, str)):
        global OPTCOM_LOG_FILENAME
        OPTCOM_LOG_FILENAME = filename
    else:

        raise TypeError('set_log_filename argument must be a string.')

def get_log_filename() -> str:

    return OPTCOM_LOG_FILENAME


PRINT_LOG = False
def set_print_log(flag: bool):
    if (isinstance(flag, bool)):
        global PRINT_LOG
        PRINT_LOG = flag
    else:

        raise TypeError('set_print_log argument must be a bool.')

def get_print_log() -> bool:

    return PRINT_LOG


STR_SEPARATOR_TERMINAL: str = "----------------------------------------------"\
                              "---------------------------------\n"# 79 charact
def set_separator_terminal(separator: str):
    if (isinstance(separator, str)):
        global STR_SEPARATOR_TERMINAL
        STR_SEPARATOR_TERMINAL = separator
    else:

        raise TypeError('set_separator_terminal argument must be a string.')

def get_separator_terminal() -> str:

    return STR_SEPARATOR_TERMINAL


RK4IP_OPTI_GNLSE: bool = True
def set_rk4ip_opti_gnlse(flag: bool):
    if (isinstance(flag, bool)):
        global RK4IP_OPTI_GNLSE
        RK4IP_OPTI_GNLSE = flag
    else:

        raise TypeError('set_rk4ip_opti_gnlse argument must be a bool.')

def get_rk4ip_opti_gnlse() -> bool:

    return RK4IP_OPTI_GNLSE


FILE_EXT: str = 'pk1'  # Must comply with pickle.dump() method
def set_file_extension(extension: str):
    if (isinstance(extension, str)):
        global FILE_EXT
        FILE_EXT = extension
    else:

        raise TypeError('set_file_extension argument must be a string.')

def get_file_extension() -> str:

    return FILE_EXT


SAVE_LEAF_FIELDS: bool = True
def set_save_leaf_fields(flag: bool):
    if (isinstance(flag, bool)):
        global SAVE_LEAF_FIELDS
        SAVE_LEAF_FIELDS = flag
    else:

        raise TypeError('set_save_leaf_fields argument must be a bool.')

def get_save_leaf_fields() -> bool:

    return SAVE_LEAF_FIELDS


MAX_NBR_PASS: int = (sys.getrecursionlimit()-1)
def set_max_nbr_pass(nbr: int):
    if (isinstance(nbr, int)):
        global MAX_NBR_PASS
        MAX_NBR_PASS = nbr
    else:

        raise TypeError('set_max_nbr_pass argument must be a int.')

def get_max_nbr_pass() -> int:

    return MAX_NBR_PASS


FIELD_OP_MATCHING_OMEGA: bool = True
def set_field_op_matching_omega(flag: bool):
    if (isinstance(flag, bool)):
        global FIELD_OP_MATCHING_OMEGA
        FIELD_OP_MATCHING_OMEGA = flag
    else:

        raise TypeError('set_field_op_matching_omega argument must be a bool.')

def get_field_op_matching_omega() -> bool:

    return FIELD_OP_MATCHING_OMEGA


FIELD_OP_MATCHING_REP_FREQ: bool = True
def set_field_op_matching_rep_freq(flag: bool):
    if (isinstance(flag, bool)):
        global FIELD_OP_MATCHING_REP_FREQ
        FIELD_OP_MATCHING_REP_FREQ = flag
    else:

        raise TypeError('set_field_op_matching_rep_freq argument must be a '
                        'bool.')

def get_field_op_matching_rep_freq() -> bool:

    return FIELD_OP_MATCHING_REP_FREQ


MULTIPROCESSING: bool = True
def set_multiprocessing(flag: bool):
    if (isinstance(flag, bool)):
        global MULTIPROCESSING
        MULTIPROCESSING = flag
    else:

        raise TypeError('set_multiprocessing argument must be a bool.')

def get_multiprocessing() -> bool:

    return MULTIPROCESSING
