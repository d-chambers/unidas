"""
Tests for core functionality of unidas.
"""

import dascore as dc
import daspy
import numpy as np
import pytest
import unidas
from lightguide.blast import Blast
from unidas import BaseDAS, adapter, convert, optional_import
from xdas.core.dataarray import DataArray

# A tuple of format names for testing generic conversions.
NAME_CLASS_MAP = {
    "dascore.Patch": dc.Patch,
    "xdas.DataArray": DataArray,
    "daspy.Section": daspy.Section,
    "lightguide.Blast": Blast,
}
BASE_FORMATS = tuple(NAME_CLASS_MAP)


# --- Tests for unidas utilities.


class TestOptionalImport:
    """Test suite for optional imports."""

    def test_import_installed_module(self):
        """Test to ensure an installed module imports."""
        import functools

        mod = optional_import("functools")
        assert mod is functools

    def test_missing_module_raises(self):
        """Ensure a module which is missing raises the appropriate Error."""
        with pytest.raises(ImportError, match="boblib4"):
            optional_import("boblib4")


class TestMisc:
    """Miscellaneous tests for unidas."""

    def test_version(self):
        """Simply ensure unidas has a version attribute."""
        assert hasattr(unidas, "__version__")
        assert isinstance(unidas.__version__, str)


# --------- Tests for unidas conversions.


class TestFormatConversionCombinations:
    """Tests for combinations of different formats."""

    # Note: we could also parametrize the base structure fixtures to make
    # all of these one test, but then it can get confusing to debug so
    # I am making one test for each format that then tests converting to
    # all other formats.
    @pytest.mark.parametrize("target_format", BASE_FORMATS)
    def test_convert_blast(self, lightguide_blast, target_format):
        """Test that the base blast can be converted to all formats."""
        out = convert(lightguide_blast, to=target_format)
        assert isinstance(out, NAME_CLASS_MAP[target_format])

    @pytest.mark.parametrize("target_format", BASE_FORMATS)
    def test_convert_patch(self, dascore_patch, target_format):
        """Test that the base patch can be converted to all formats."""
        out = convert(dascore_patch, to=target_format)
        assert isinstance(out, NAME_CLASS_MAP[target_format])

    @pytest.mark.parametrize("target_format", BASE_FORMATS)
    def test_convert_data_array(self, xdas_dataarray, target_format):
        """Test that the base data array can be converted to all formats."""
        out = convert(xdas_dataarray, to=target_format)
        assert isinstance(out, NAME_CLASS_MAP[target_format])

    @pytest.mark.parametrize("target_format", BASE_FORMATS)
    def test_convert_section(self, daspy_section, target_format):
        """Test that the base section can be converted to all formats."""
        out = convert(daspy_section, to=target_format)
        assert isinstance(out, NAME_CLASS_MAP[target_format])


class TestDASCorePatch:
    """Test suite for converting DASCore Patches."""

    @pytest.fixture(scope="class")
    def dascore_base_das(self, dascore_patch):
        """The converted DASCore patch."""
        return convert(dascore_patch, "unidas.BaseDAS")

    def test_to_base_das(self, dascore_base_das):
        """Ensure we can convert DASCore patch to BaseDAS."""
        assert isinstance(dascore_base_das, BaseDAS)

    def test_from_base_das(self, dascore_base_das, dascore_patch):
        """Test the conversion back to DASCore Patch from BaseDAS."""
        out = convert(dascore_base_das, "dascore.Patch")
        assert isinstance(out, dc.Patch)
        assert out == dascore_patch


class TestDASPySection:
    """Test suite for converting DASPy sections."""

    @pytest.fixture(scope="class")
    def daspy_base_das(self, daspy_section):
        """The default daspy section converted to BaseDAS instance."""
        return convert(daspy_section, "unidas.BaseDAS")

    def test_to_base_das(self, daspy_base_das):
        """Ensure the base section can be converted to BaseDAS."""
        assert isinstance(daspy_base_das, BaseDAS)

    def test_from_base_das(self, daspy_base_das, daspy_section):
        """Ensure the default section can round-trip."""
        out = convert(daspy_base_das, "daspy.Section")
        # TODO these objects aren't equal but their strings are.
        # Need to fix this.
        assert str(out) == str(daspy_section)
        assert np.all(out.data == daspy_section.data)


class TestXdasDataArray:
    """Tests for converting xdas DataArrays."""

    @pytest.fixture(scope="class")
    def xdas_base_das(self, xdas_dataarray):
        """Converted xdas section to BaseDAS."""
        return convert(xdas_dataarray, "unidas.BaseDAS")

    def test_to_base_das(self, xdas_base_das):
        """Ensure the example data_array can be converted to BaseDAS."""
        assert isinstance(xdas_base_das, BaseDAS)

    def test_from_base_das(self, xdas_base_das, xdas_dataarray):
        """Ensure xdas DataArray can round trip."""
        out = convert(xdas_base_das, "xdas.DataArray")
        assert np.all(out.data == xdas_dataarray.data)
        # TODO the str rep of coords are equal but not coords themselves.
        # Need to look into this.
        assert str(out.coords) == str(xdas_dataarray.coords)
        attr1, attr2 = out.attrs, xdas_dataarray.attrs
        assert attr1 == attr2 or (not attr1 and not attr2)
        assert out.dims == xdas_dataarray.dims


class TestLightGuideBlast:
    """Tests for Blast Conversions."""

    @pytest.fixture(scope="class")
    def lightguide_base_das(self, lightguide_blast):
        """Converted lightguide blast to BaseDAS."""
        return convert(lightguide_blast, "unidas.BaseDAS")

    def test_base_das(self, lightguide_base_das):
        """Ensure the example blast can be converted to BaseDAS."""
        assert isinstance(lightguide_base_das, BaseDAS)

    def test_from_base_das(self, lightguide_base_das, lightguide_blast):
        """Ensure lightguide Blast can round trip."""
        out = convert(lightguide_base_das, "lightguide.Blast")
        # TODO here the objects also do not compare equal. Need to figure out
        # why. For now just do weaker checks.
        # assert np.all(out.data == lightguide_blast.data)
        assert out.start_time == lightguide_blast.start_time
        assert np.all(out.data == lightguide_blast.data)
        assert out.unit == lightguide_blast.unit
        assert out.channel_spacing == lightguide_blast.channel_spacing
        assert out.start_channel == lightguide_blast.start_channel
        assert out.sampling_rate == lightguide_blast.sampling_rate


class TestConvert:
    """Generic tests for the convert function."""

    def test_bad_path_raises(self, dascore_patch):
        """Ensure a bad target raises a ValueError."""
        msg = "No conversion path"
        with pytest.raises(ValueError, match=msg):
            convert(dascore_patch, "notadaslibrary.NotAClass")


class TestAdapter:
    """Tests for adapter decorator."""

    def test_conversion(self, dascore_patch):
        """Simple conversion test."""

        @adapter("daspy.Section")
        def section_function(sec):
            """Dummy section function."""
            assert isinstance(sec, daspy.Section)
            return sec

        patch = dascore_patch.transpose("distance", "time")
        out = section_function(patch)
        assert isinstance(out, dc.Patch)

    def test_wrapping(self):
        """
        Ensure a function is only wrapped once for each target, and that
        the original function is accessible.
        """

        @adapter("dascore.Patch")
        def my_patch_func(patch):
            """A dummy patch function."""
            return patch

        assert hasattr(my_patch_func, "raw_function")
        assert my_patch_func is not my_patch_func.raw_function
        # This should simply return the original function
        new = adapter("dascore.Patch")(my_patch_func)
        assert new is my_patch_func
        # But this should wrap it again.
        new2 = adapter("daspy.Section")(my_patch_func)
        assert new2 is not my_patch_func
        # The raw function should remain unchanged.
        assert new2.raw_function is my_patch_func.raw_function
        assert new.raw_function is my_patch_func.raw_function
