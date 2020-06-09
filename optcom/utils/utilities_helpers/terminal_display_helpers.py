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

import logging
from typing import Any, Callable, List, Optional, overload, Set, Tuple, Union

import optcom.config as cfg


def print_terminal(to_print: Optional[str] = None,
                   sep_type: str = cfg.STR_SEPARATOR_TERMINAL) -> None:
    """Print a string on the terminal."""
    str_to_log = ''
    str_to_log += sep_type
    if (to_print is not None):
        str_to_log += to_print
    if (cfg.PRINT_LOG):
        print(str_to_log)
    logging.basicConfig(filename=cfg.OPTCOM_LOG_FILENAME, level=logging.INFO)
    logging.info(str_to_log)


def warning_terminal(to_print: str,
                     sep_type: str = cfg.STR_SEPARATOR_TERMINAL) -> None:
    """Print a warning on the terminal."""
    str_to_log = ''
    str_to_log += sep_type
    str_to_log += to_print
    if (cfg.PRINT_LOG):
        print("!WARNING!: ", str_to_log)
