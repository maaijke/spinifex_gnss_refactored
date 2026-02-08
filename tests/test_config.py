"""
Tests for spinifex_gnss.config module.

Tests configuration constants and helper functions.
"""

import pytest
import astropy.units as u
import numpy as np

from spinifex_gnss.config import (
    FREQ,
    GNSS_OBS_PRIORITY,
    get_tec_coefficient,
    DISTANCE_KM_CUT,
    ELEVATION_CUT,
    DEFAULT_IONO_HEIGHT,
)


class TestFrequencyDefinitions:
    """Test GNSS frequency definitions."""
    
    def test_freq_structure(self):
        """Test FREQ dictionary has expected structure."""
        assert isinstance(FREQ, dict)
        assert len(FREQ) >= 5  # At least G, E, R, C, J
        
    def test_gps_frequencies(self):
        """Test GPS frequencies are correct."""
        assert "G" in FREQ
        assert FREQ["G"]["f1"] == 1575.42e6
        assert FREQ["G"]["f2"] == 1227.60e6


class TestObservationPriorities:
    """Test GNSS observation code priorities."""
    
    def test_priority_structure(self):
        """Test GNSS_OBS_PRIORITY has expected structure."""
        assert isinstance(GNSS_OBS_PRIORITY, dict)
        assert len(GNSS_OBS_PRIORITY) >= 5
        
    def test_gps_priorities(self):
        """Test GPS observation priorities."""
        assert "G" in GNSS_OBS_PRIORITY
        gps = GNSS_OBS_PRIORITY["G"]
        
        assert "C1" in gps
        assert "C2" in gps
        assert "L1" in gps
        assert "L2" in gps


class TestTECCoefficients:
    """Test TEC coefficient calculation."""
    
    def test_gps_coefficient(self):
        """Test GPS TEC coefficient calculation."""
        C12 = get_tec_coefficient("G")
        assert C12 < 0
        
        f1 = FREQ["G"]["f1"]
        f2 = FREQ["G"]["f2"]
        expected = 1e-16 / (40.3 * (1.0 / f1**2 - 1.0 / f2**2))
        
        assert np.isclose(C12, expected, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
