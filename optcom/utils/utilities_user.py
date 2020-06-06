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

import optcom.field as fld
from optcom.utils.csv_fit import CSVFit
from optcom.utils.utilities import db_to_linear, linear_to_db

temporal_power = fld.Field.temporal_power
spectral_power = fld.Field.spectral_power
temporal_phase = fld.Field.temporal_phase
spectral_phase = fld.Field.temporal_phase
energy = fld.Field.energy
average_power = fld.Field.average_power
fwhm = fld.Field.fwhm
