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

import math
from typing import Any, Callable, List, Optional, overload, Set, Tuple, Union

import numpy as np


# Exceptions
class DimensionError(Exception):
    pass


def pulse_positions_in_time_window(nbr_channels: int, rep_freq: List[float],
                                   time_window: float, position: List[float]
                                   ) -> List[np.ndarray]:
    """Return the relative position of the pulses in the considered time
    window depending on the specified absolute positions and repetition
    frequencies.

    Parameters
    ----------
    nbr_channels :
        The number of channels.
    rep_freq :
        The repetion frequency of each channel. :math:`[THz]`
    time_window :
        The time window. :math:`[ps]`
    position :
        The position in the time window for each channel.

    Returns
    -------
    :
        The relative positions of the channels in the provided time
        window depending on the repetion frequencies.

    """
    # Calculate number of pulses that fit in the provided time window
    if ((len(rep_freq) < nbr_channels) or (len(position) < nbr_channels)):

        raise DimensionError("Provided repetion frequencies and positions "
            "do not comply with the provided number of channels.")
    nbr_pulses: List[int] = []
    for i in range(nbr_channels):
        if (rep_freq[i]):
            nbr_temp = math.floor(time_window * rep_freq[i])
            if (nbr_temp):
                nbr_pulses.append(nbr_temp)
            else:   # time window to thin to hold more than 1 pulse
                nbr_pulses.append(1)
        else:
            nbr_pulses.append(1)
    # Set the position of the pulses in the time window
    rel_pos: List[np.ndarray] = []
    pos_step: float
    dist_from_center: float
    for i in range(nbr_channels):
        pos_step = 1/nbr_pulses[i]
        if (nbr_pulses[i]%2):  # Odd
            dist_from_center = nbr_pulses[i]//2 * pos_step
        else:
            dist_from_center = (nbr_pulses[i]//2 - 1)*pos_step + pos_step/2
        rel_pos.append(np.linspace(position[i] - dist_from_center,
                                   position[i] + dist_from_center,
                                   num=nbr_pulses[i]))

    return rel_pos
