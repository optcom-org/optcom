# This file is part of Optcom.
#
# Optcom is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Optcom is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Optcom.  If not, see <https://www.gnu.org/licenses/>.

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
