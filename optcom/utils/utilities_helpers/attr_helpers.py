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

import operator
from typing import Any, Callable, List, Optional, overload, Set, Tuple, Union

from optcom.utils.utilities_helpers.terminal_display_helpers \
    import warning_terminal


def check_attr_type(attr: Any, attr_name: str, *types: Any) -> None:
    """Check if the provided attribute is of the provided type."""
    res = False
    for type in types:
        if (type is None):    # Allow attr to be None
            if (attr is None):
                res = True
        else:
            res = res or isinstance(attr, type)   # Keep the True
    if (not res):
        types_name = locals()['types']
        raise TypeError("{} must be one of the following type: {}"
                        .format(attr_name, types_name))


def check_attr_range(attr: Any, attr_name: str, min_attr: Union[int, float],
                     max_attr: Union[int, float], strict_left: bool = False,
                     strict_right: bool = False) -> None:
    """Check if the provided attribute is in the range provided."""
    if (strict_left):
        op_left = operator.lt
    else:
        op_left = operator.le
    if (strict_right):
        op_right = operator.gt
    else:
        op_right = operator.ge

    if (not(op_left(min_attr, attr) and op_right(max_attr, attr))):
        raise IndexError("Attribute {} out of range, must be in interval "
                         "[{},{}]"
                         .format(attr_name, min_attr, max_attr))


def check_attr_value(attr: Any, attr_values: List[Any], attr_default: Any
                     ) -> Any:
    """Check if the provided attribute is one of the allowed values
    provided. If not, return a provided default value.
    """
    if (attr not in attr_values):
        warning_terminal("The attribute specified '{}' is not "
            "supported, default attribute '{}' set instead."
            .format(attr, attr_default))

        return attr_default
    else:

        return attr
