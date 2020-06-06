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
