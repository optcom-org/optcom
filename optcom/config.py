import sys

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
