"""
Tests for spinifex_gnss.parse_gnss module.

Tests GNSS data parsing without DCB dependencies.
"""

import pytest
import numpy as np
from pathlib import Path
from astropy.time import Time

from spinifex_gnss.parse_gnss import (
    GNSSData,
    dummy_gnss_data,
    get_gnss_data,
    process_all_rinex_parallel,
)


class TestGNSSData:
    """Test GNSSData NamedTuple."""
    
    def test_gnssdata_structure(self):
        """Test GNSSData has expected fields."""
        data = GNSSData(
            gnss={"G01": np.zeros((100, 4))},
            c1_str="C1W",
            c2_str="C2W",
            l1_str="L1W",
            l2_str="L2W",
            station="WSRT00NLD",
            is_valid=True,
            times=Time(np.arange(100), format='mjd'),
            constellation="G"
        )
        
        assert data.station == "WSRT00NLD"
        assert data.is_valid == True
        assert data.constellation == "G"
        assert data.c1_str == "C1W"
        
    def test_gnssdata_observation_array_shape(self):
        """Test observation array has correct shape."""
        obs_array = np.random.randn(100, 4)
        
        data = GNSSData(
            gnss={"G01": obs_array},
            c1_str="C1W",
            c2_str="C2W",
            l1_str="L1W",
            l2_str="L2W",
            station="TEST",
            is_valid=True,
            times=Time(np.arange(100), format='mjd'),
            constellation="G"
        )
        
        assert data.gnss["G01"].shape == (100, 4)
        # Columns are: C1, C2, L1, L2
        
    def test_dummy_gnss_data(self):
        """Test dummy_gnss_data creates invalid data."""
        dummy = dummy_gnss_data("TEST", "G")
        
        assert dummy.station == "TEST"
        assert dummy.constellation == "G"
        assert dummy.is_valid == False
        assert dummy.gnss is None
        assert dummy.times is None


class TestProcessAllRinexParallel:
    """Test parallel RINEX processing."""
    
    def test_function_signature(self):
        """Test function has correct signature (no DCB parameter)."""
        import inspect
        sig = inspect.signature(process_all_rinex_parallel)
        params = list(sig.parameters.keys())
        
        # Should have rinex_files and max_workers
        assert 'rinex_files' in params
        assert 'max_workers' in params
        
        # Should NOT have dcb parameter
        assert 'dcb' not in params
        
    def test_empty_list_returns_empty(self):
        """Test empty file list returns empty result."""
        result = process_all_rinex_parallel([])
        assert result == []
        
    def test_max_workers_parameter(self):
        """Test max_workers parameter is accepted."""
        # Should not raise error
        result = process_all_rinex_parallel([], max_workers=10)
        assert result == []


class TestObservationCodeSelection:
    """Test observation code selection logic."""
    
    def test_code_priorities_used(self):
        """Test that code priorities from config are used."""
        from spinifex_gnss.config import GNSS_OBS_PRIORITY
        
        # GPS should prefer W codes
        gps_c1 = GNSS_OBS_PRIORITY["G"]["C1"]
        assert gps_c1[0] in ["C1W", "C1P"]  # W or P preferred
        
    def test_all_constellations_have_priorities(self):
        """Test all supported constellations have observation priorities."""
        from spinifex_gnss.config import GNSS_OBS_PRIORITY
        
        required_constellations = ["G", "E", "R", "C", "J"]
        
        for const in required_constellations:
            assert const in GNSS_OBS_PRIORITY
            assert "C1" in GNSS_OBS_PRIORITY[const]
            assert "C2" in GNSS_OBS_PRIORITY[const]
            assert "L1" in GNSS_OBS_PRIORITY[const]
            assert "L2" in GNSS_OBS_PRIORITY[const]


class TestNoDCBDependency:
    """Test that module has no DCB dependencies."""
    
    def test_no_dcb_imports(self):
        """Test module doesn't import DCB-related functions."""
        import spinifex_gnss.parse_gnss as module
        
        # Should not have these attributes
        assert not hasattr(module, 'DCBdata')
        assert not hasattr(module, 'parse_dcb_sinex')
        
    def test_gnssdata_no_dcb_field(self):
        """Test GNSSData doesn't have has_dcb field."""
        from spinifex_gnss.parse_gnss import GNSSData
        
        fields = GNSSData._fields
        assert 'has_dcb' not in fields
        assert 'dcb' not in fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
