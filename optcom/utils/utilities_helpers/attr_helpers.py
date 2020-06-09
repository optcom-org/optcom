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
