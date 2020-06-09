import pytest

import numpy as np
from numpy.testing.utils import assert_array_equal

from optcom.components.gaussian import Gaussian
from optcom.components.load_field import LoadField
from optcom.components.save_field import SaveField
from optcom.domain import Domain
from optcom.field import Field
from optcom.layout import Layout

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.components
def test_load_field():
    """Should fail if the saved fields are not the same as the loaded
    fields.
    """
    lt = Layout()
    gssn_1 = Gaussian(channels=1, width=[5.0])
    field_saver_1 = SaveField()
    gssn_2 = Gaussian(channels=1, width=[10.0])
    field_saver_2 = SaveField()

    lt.add_links((gssn_1[0], field_saver_1[0]), (gssn_2[0], field_saver_2[0]))
    lt.run(gssn_1, gssn_2)

    fields = field_saver_1.fields + field_saver_2.fields

    lt_ = Layout()
    load_field = LoadField(fields=fields)

    lt.run(load_field)

    fields = load_field[0].fields

    assert (fields[0] == gssn_1[0].fields[0])
    assert (fields[1] == gssn_2[0].fields[0])
    assert_array_equal(fields[0].channels, gssn_1[0].fields[0].channels)
    assert_array_equal(fields[1].channels, gssn_2[0].fields[0].channels)
    assert_array_equal(fields[0].noise, gssn_1[0].fields[0].noise)
    assert_array_equal(fields[1].noise, gssn_2[0].fields[0].noise)
    assert_array_equal(fields[0].delays, gssn_1[0].fields[0].delays)
    assert_array_equal(fields[1].delays, gssn_2[0].fields[0].delays)
