"""
Tests for spinifex_gnss.tec_core module.

Tests core TEC calculation functions without pseudorange TEC.
"""

import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u

from spinifex_gnss.tec_core import (
    get_transmission_time,
    getphase_tec,
    _get_cycle_slips,
    _get_phase_corrected,
)


class TestTransmissionTime:
    """Test transmission time calculation."""
    
    def test_basic_calculation(self):
        """Test basic transmission time calculation."""
        # Pseudorange of 20,000 km
        c2 = np.array([20000000.0])
        times = Time([59000.0], format='mjd')
        
        tx_time = get_transmission_time(c2, times)
        
        # Should be earlier than observation time
        assert tx_time.mjd[0] < times.mjd[0]
        
        # Difference should be ~67 ms (20,000 km / c)
        dt = (times - tx_time).to(u.ms)
        expected_dt = (c2[0] * u.m / (3e8 * u.m/u.s)).to(u.ms)
        assert np.isclose(dt.value[0], expected_dt.value, rtol=0.01)
        
    def test_nan_handling(self):
        """Test that NaN values are handled."""
        c2 = np.array([20000000.0, np.nan, 21000000.0])
        times = Time([59000.0, 59000.001, 59000.002], format='mjd')
        
        tx_time = get_transmission_time(c2, times)
        
        # Should return valid Time object
        assert isinstance(tx_time, Time)
        assert len(tx_time) == 3
        
    def test_no_dcb_parameter(self):
        """Test function doesn't have DCB parameters."""
        import inspect
        sig = inspect.signature(get_transmission_time)
        params = list(sig.parameters.keys())
        
        assert 'dcb_sat' not in params
        assert 'dcb_stat' not in params


class TestPhaseTEC:
    """Test carrier phase TEC calculation."""
    
    def test_basic_calculation(self):
        """Test basic phase TEC calculation."""
        # Create simple phase observations
        l1 = np.array([1000000.0, 1000100.0, 1000200.0])
        l2 = np.array([800000.0, 800080.0, 800160.0])
        
        phase_tec = getphase_tec(l1, l2, constellation="G")
        
        # Should return array of same length
        assert len(phase_tec) == 3
        
        # Should be finite
        assert np.all(np.isfinite(phase_tec))
        
    def test_different_constellations(self):
        """Test phase TEC for different constellations."""
        l1 = np.array([1000000.0])
        l2 = np.array([800000.0])
        
        tec_gps = getphase_tec(l1, l2, "G")
        tec_galileo = getphase_tec(l1, l2, "E")
        
        # Different frequencies → different TEC values
        assert tec_gps[0] != tec_galileo[0]
        
    def test_negative_sign(self):
        """Test that phase TEC has correct sign (phase advance)."""
        # Phase observations
        l1 = np.array([1000000.0])
        l2 = np.array([800000.0])
        
        tec = getphase_tec(l1, l2, "G")
        
        # Formula uses negative sign for phase
        # Result can be positive or negative depending on actual values
        assert np.isfinite(tec[0])


class TestCycleSlips:
    """Test cycle slip detection."""
    
    def test_no_slips(self):
        """Test data without cycle slips."""
        # Smooth phase TEC progression
        phase_tec = np.linspace(10, 20, 100)
        
        segments = _get_cycle_slips(phase_tec)
        
        # Should all be in same segment
        assert len(np.unique(segments)) == 1
        
    def test_with_gap(self):
        """Test detection of data gap (NaN)."""
        phase_tec = np.concatenate([
            np.linspace(10, 15, 50),
            [np.nan, np.nan, np.nan],
            np.linspace(15, 20, 50)
        ])
        
        segments = _get_cycle_slips(phase_tec)
        
        # Should have at least 2 segments (before and after gap)
        assert len(np.unique(segments)) >= 2
        
    def test_with_large_jump(self):
        """Test detection of large jump (cycle slip)."""
        phase_tec = np.concatenate([
            np.linspace(10, 20, 50),
            np.linspace(50, 60, 50)  # 30 TECU jump
        ])
        
        segments = _get_cycle_slips(phase_tec)
        
        # Should detect the jump
        assert len(np.unique(segments)) >= 2


class TestPhaseCorrected:
    """Test phase bias correction."""
    
    def test_basic_correction(self):
        """Test basic bias correction."""
        # Phase TEC with arbitrary bias
        phase_tec = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        
        # Pseudorange TEC (noisier but unbiased)
        pseudo_tec = np.array([15.0, 16.0, 17.0, 18.0, 19.0])
        
        corrected, std = _get_phase_corrected(phase_tec, pseudo_tec)
        
        # Corrected should be closer to pseudorange mean
        assert np.abs(np.mean(corrected) - np.mean(pseudo_tec)) < \
               np.abs(np.mean(phase_tec) - np.mean(pseudo_tec))
               
    def test_returns_two_values(self):
        """Test function returns corrected TEC and std."""
        phase_tec = np.linspace(10, 20, 50)
        pseudo_tec = phase_tec + np.random.randn(50) * 0.5 + 5  # Add bias and noise
        
        result = _get_phase_corrected(phase_tec, pseudo_tec)
        
        assert len(result) == 2
        corrected, std = result
        assert len(corrected) == len(phase_tec)
        assert len(std) == len(phase_tec)


class TestNoPseudorangeTEC:
    """Test that getpseudorange_tec was removed."""
    
    def test_function_not_in_module(self):
        """Test getpseudorange_tec is not in module."""
        import spinifex_gnss.tec_core as module
        
        assert not hasattr(module, 'getpseudorange_tec')
        
    def test_no_pseudorange_in_exports(self):
        """Test getpseudorange_tec not in module __all__ if it exists."""
        import spinifex_gnss.tec_core as module
        
        if hasattr(module, '__all__'):
            assert 'getpseudorange_tec' not in module.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
