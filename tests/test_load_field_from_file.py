import pytest
import pickle
import os

import numpy as np
from numpy.testing.utils import assert_array_equal

from optcom.components.gaussian import Gaussian
from optcom.components.load_field_from_file import LoadFieldFromFile
from optcom.components.save_field_to_file import SaveFieldToFile
from optcom.field import Field
from optcom.layout import Layout


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.components
def test_save_field_to_file():
    """Should fail if the saved fields are not the same as the loaded
    fields.
    """
    file_name = 'temp_file_for_save_field_to_file_test.pk1'
    if (os.path.isfile(file_name)):
        print("Can not perfom test because a file named '{}' already exist."
              .format(file_name))
        assert False
    else:
        lt = Layout()
        gssn_1 = Gaussian(channels=1, width=[5.0], save=True)
        field_saver_1 = SaveFieldToFile(file_name=file_name, add_fields=False)
        gssn_2 = Gaussian(channels=1, width=[10.0], save=True)
        field_saver_2 = SaveFieldToFile(file_name=file_name, add_fields=True)

        lt.link((gssn_1[0], field_saver_1[0]), (gssn_2[0], field_saver_2[0]))
        lt.run(gssn_1, gssn_2)

        lt_ = Layout()
        load_field = LoadFieldFromFile(file_name=file_name)

        lt_.run(load_field)
        fields = load_field[0].fields

        # Removing created file
        os.remove(file_name)

        # Tests
        assert (fields[0] == gssn_1[0].fields[0])
        assert (fields[1] == gssn_2[0].fields[0])
        assert_array_equal(fields[0].channels, gssn_1[0].fields[0].channels)
        assert_array_equal(fields[1].channels, gssn_2[0].fields[0].channels)
        assert_array_equal(fields[0].noise, gssn_1[0].fields[0].noise)
        assert_array_equal(fields[1].noise, gssn_2[0].fields[0].noise)
        assert_array_equal(fields[0].delays, gssn_1[0].fields[0].delays)
        assert_array_equal(fields[1].delays, gssn_2[0].fields[0].delays)
