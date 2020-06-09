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

from typing import Any, Callable, List, Optional, overload, Set, Tuple, Union

import optcom.config as cfg


def print_terminal(to_print: Optional[str] = None,
                   sep_type: str = cfg.STR_SEPARATOR_TERMINAL) -> None:
    """Print a string on the terminal."""
    print(sep_type, end='')
    if (to_print is not None):
        print(to_print)


def warning_terminal(to_print: str,
                     sep_type: str = cfg.STR_SEPARATOR_TERMINAL) -> None:
    """Print a warning on the terminal."""
    print(sep_type, end='')
    print("!WARNING!: ", to_print)
