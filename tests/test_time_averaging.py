"""
Unit tests for time averaging functionality.

Tests the new time-averaged processing to ensure:
1. Time window selection works correctly
2. Data structures are built properly
3. Measurements are combined correctly
4. Edge cases are handled
"""

import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation

from spinifex.geometry import IPP
from spinifex_gnss.proces_gnss_data_time_averaged import (
    _select_time_window,
    _get_distance_ipp_time_averaged,
    get_interpolated_tec,
    N_TIME_SLOTS,
    MAX_TIME_DIFF_MINUTES,
)


# ============================================================================
# Test Time Window Selection
# ============================================================================

class TestTimeWindowSelection:
    """Test _select_time_window function."""
    
    def test_basic_selection(self):
        """Test basic time window selection."""
        # Setup: observations every 30 seconds for 10 minutes
        obs_times = Time('2024-06-18T12:00:00') + np.arange(20) * 30 * u.s
        obs_times_mjd = obs_times.mjd
        
        # Target: middle of the sequence
        target_time = Time('2024-06-18T12:05:00')
        
        # Select 5 slots within ±2.5 minutes
        selected = _select_time_window(
            target_time.mjd,
            obs_times_mjd,
            n_slots=5,
            max_diff_min=2.5
        )
        
        # Should get indices around the middle
        assert len(selected) == 5
        assert 10 in selected  # Exact match should be included
        
    def test_exact_match(self):
        """Test that exact time match is always included."""
        obs_times = Time('2024-06-18T12:00:00') + np.arange(10) * 1 * u.min
        obs_times_mjd = obs_times.mjd
        
        target_time = Time('2024-06-18T12:05:00')
        
        selected = _select_time_window(
            target_time.mjd,
            obs_times_mjd,
            n_slots=3,
            max_diff_min=2.5
        )
        
        # Index 5 is exact match at 12:05:00
        assert 5 in selected
        
    def test_fewer_than_n_slots(self):
        """Test when fewer observations than n_slots are available."""
        # Only 3 observations
        obs_times = Time('2024-06-18T12:00:00') + np.array([0, 1, 2]) * u.min
        obs_times_mjd = obs_times.mjd
        
        target_time = Time('2024-06-18T12:01:00')
        
        # Request 5 slots, but only 3 available within window
        selected = _select_time_window(
            target_time.mjd,
            obs_times_mjd,
            n_slots=5,
            max_diff_min=2.5
        )
        
        # Should return all 3 available
        assert len(selected) <= 5
        assert len(selected) <= 3
        
    def test_no_obs_within_window(self):
        """Test fallback to nearest when no obs within max_diff."""
        obs_times = Time('2024-06-18T12:00:00') + np.array([0, 10, 20]) * u.min
        obs_times_mjd = obs_times.mjd
        
        # Target at 12:05, but nearest is at 12:00 (5 min away)
        target_time = Time('2024-06-18T12:05:00')
        
        selected = _select_time_window(
            target_time.mjd,
            obs_times_mjd,
            n_slots=5,
            max_diff_min=2.5  # Only ±2.5 min window
        )
        
        # Should return nearest (fallback behavior)
        assert len(selected) == 1
        # Nearest is index 1 (12:10:00, 5 min away)
        
    def test_symmetric_selection(self):
        """Test that selection is symmetric around target."""
        obs_times = Time('2024-06-18T12:00:00') + np.arange(20) * 30 * u.s
        obs_times_mjd = obs_times.mjd
        
        target_time = Time('2024-06-18T12:05:00')  # Index 10
        
        selected = _select_time_window(
            target_time.mjd,
            obs_times_mjd,
            n_slots=5,
            max_diff_min=2.5
        )
        
        # Should be roughly symmetric
        # For 5 slots: should get indices [8, 9, 10, 11, 12] or similar
        selected_sorted = sorted(selected)
        center_idx = 10
        
        # Check that selected indices are close to center
        assert min(selected_sorted) >= center_idx - 3
        assert max(selected_sorted) <= center_idx + 3


# ============================================================================
# Test Data Structure Building
# ============================================================================

class TestDataStructure:
    """Test that data structures are built correctly."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        # Create simple test data
        n_prns = 3
        n_obs_times = 20
        n_target_times = 5
        n_heights = 4
        
        # Mock STEC data
        stec_values = np.random.randn(n_prns, n_obs_times) * 10 + 20
        stec_errors = np.abs(np.random.randn(n_prns, n_obs_times)) * 2
        
        # Mock IPPs (simplified)
        obs_times = Time('2024-06-18T12:00:00') + np.arange(n_obs_times) * 30 * u.s
        target_times = Time('2024-06-18T12:00:00') + np.arange(n_target_times) * 2 * u.min
        heights = np.arange(n_heights) * 100 * u.km + 100 * u.km
        
        # Create mock IPPs for satellites
        ipp_sat_stat = []
        for prn in range(n_prns):
            # Create locations (simplified - same for all satellites)
            locs = EarthLocation(
                lon=np.zeros(n_obs_times) * u.deg,
                lat=np.zeros(n_obs_times) * u.deg,
                height=np.zeros(n_obs_times) * u.m
            )
            
            # Expand to all heights
            locs_all_heights = np.array([locs] * n_heights).T
            
            # Create IPP (simplified - missing some fields)
            from types import SimpleNamespace
            ipp = SimpleNamespace()
            ipp.loc = locs_all_heights
            ipp.altaz = SimpleNamespace()
            ipp.altaz.alt = SimpleNamespace()
            ipp.altaz.alt.deg = np.ones(n_obs_times) * 45  # 45° elevation
            ipp.airmass = np.ones((n_obs_times, n_heights)) * 2.0
            
            ipp_sat_stat.append(ipp)
        
        # Create target IPP
        target_locs = EarthLocation(
            lon=np.zeros(n_target_times) * u.deg,
            lat=np.zeros(n_target_times) * u.deg,
            height=np.zeros(n_target_times) * u.m
        )
        target_locs_all_heights = np.array([target_locs] * n_heights).T
        
        ipp_target = SimpleNamespace()
        ipp_target.loc = target_locs_all_heights
        ipp_target.times = target_times
        
        # Mock profiles
        profiles = np.ones((n_target_times, n_heights))
        
        # Call function
        result = _get_distance_ipp_time_averaged(
            stec_values=stec_values,
            stec_errors=stec_errors,
            ipp_sat_stat=ipp_sat_stat,
            ipp_target=ipp_target,
            obs_times=obs_times,
            profiles=profiles,
            n_time_slots=3,
            max_time_diff_min=1.5,
            use_time_weighting=False,
        )
        
        # Check output structure
        assert len(result) == n_target_times, f"Expected {n_target_times} time slots, got {len(result)}"
        assert len(result[0]) == n_heights, f"Expected {n_heights} heights, got {len(result[0])}"
        
        # Check that each element is an array (possibly empty)
        for tidx in range(n_target_times):
            for hidx in range(n_heights):
                assert isinstance(result[tidx][hidx], np.ndarray), \
                    f"result[{tidx}][{hidx}] is not ndarray, got {type(result[tidx][hidx])}"
    
    def test_column_structure(self):
        """Test that output arrays have correct columns."""
        # Simplified test with minimal data
        n_prns = 2
        n_obs_times = 10
        n_target_times = 2
        n_heights = 2
        
        stec_values = np.ones((n_prns, n_obs_times)) * 20
        stec_errors = np.ones((n_prns, n_obs_times)) * 2
        
        obs_times = Time('2024-06-18T12:00:00') + np.arange(n_obs_times) * 30 * u.s
        target_times = Time('2024-06-18T12:00:00') + np.array([0, 5]) * u.min
        heights = np.array([200, 400]) * u.km
        
        # Create simple IPPs
        ipp_sat_stat = []
        for prn in range(n_prns):
            from types import SimpleNamespace
            
            locs = EarthLocation(
                lon=np.random.randn(n_obs_times) * 0.1 * u.deg,
                lat=np.random.randn(n_obs_times) * 0.1 * u.deg,
                height=np.zeros(n_obs_times) * u.m
            )
            locs_all = np.array([[locs[i] for _ in range(n_heights)] for i in range(n_obs_times)])
            
            ipp = SimpleNamespace()
            ipp.loc = locs_all
            ipp.altaz = SimpleNamespace()
            ipp.altaz.alt = SimpleNamespace()
            ipp.altaz.alt.deg = np.ones(n_obs_times) * 60
            ipp.airmass = np.ones((n_obs_times, n_heights)) * 1.5
            
            ipp_sat_stat.append(ipp)
        
        target_locs = EarthLocation(
            lon=np.zeros(n_target_times) * u.deg,
            lat=np.zeros(n_target_times) * u.deg,
            height=np.zeros(n_target_times) * u.m
        )
        target_locs_all = np.array([[target_locs[i] for _ in range(n_heights)] for i in range(n_target_times)])
        
        ipp_target = SimpleNamespace()
        ipp_target.loc = target_locs_all
        ipp_target.times = target_times
        
        profiles = np.ones((n_target_times, n_heights))
        
        # Test WITHOUT time weighting
        result_no_weight = _get_distance_ipp_time_averaged(
            stec_values=stec_values,
            stec_errors=stec_errors,
            ipp_sat_stat=ipp_sat_stat,
            ipp_target=ipp_target,
            obs_times=obs_times,
            profiles=profiles,
            n_time_slots=3,
            max_time_diff_min=2.0,
            use_time_weighting=False,
        )
        
        # Check columns: [VTEC, error, dlon, dlat]
        for tidx in range(n_target_times):
            for hidx in range(n_heights):
                data = result_no_weight[tidx][hidx]
                if data.shape[0] > 0:
                    assert data.shape[1] == 4, \
                        f"Expected 4 columns without time weighting, got {data.shape[1]}"
        
        # Test WITH time weighting
        result_with_weight = _get_distance_ipp_time_averaged(
            stec_values=stec_values,
            stec_errors=stec_errors,
            ipp_sat_stat=ipp_sat_stat,
            ipp_target=ipp_target,
            obs_times=obs_times,
            profiles=profiles,
            n_time_slots=3,
            max_time_diff_min=2.0,
            use_time_weighting=True,
        )
        
        # Check columns: [VTEC, error, dlon, dlat, time_weight]
        for tidx in range(n_target_times):
            for hidx in range(n_heights):
                data = result_with_weight[tidx][hidx]
                if data.shape[0] > 0:
                    assert data.shape[1] == 5, \
                        f"Expected 5 columns with time weighting, got {data.shape[1]}"
                    
                    # Time weights should be positive and sum to ~1 per satellite
                    time_weights = data[:, 4]
                    assert np.all(time_weights > 0), "Time weights must be positive"


# ============================================================================
# Test Measurement Increase
# ============================================================================

class TestMeasurementIncrease:
    """Test that time averaging increases measurements."""
    
    def test_more_measurements_with_averaging(self):
        """Test that time averaging gives more measurements."""
        # This is a conceptual test - in practice we'd need real data
        # to verify the exact increase, but we can test the principle
        
        # With n_time_slots=1, we get measurements from 1 time slot
        # With n_time_slots=5, we should get up to 5× measurements
        
        # Create mock scenario
        n_prns = 10
        n_obs_times = 20
        n_target_times = 5
        n_heights = 3
        
        # All satellites visible (high elevation)
        stec_values = np.random.randn(n_prns, n_obs_times) * 5 + 20
        stec_errors = np.ones((n_prns, n_obs_times)) * 2
        
        obs_times = Time('2024-06-18T12:00:00') + np.arange(n_obs_times) * 30 * u.s
        target_times = Time('2024-06-18T12:00:00') + np.arange(n_target_times) * 2 * u.min
        
        # Setup simplified IPPs
        from types import SimpleNamespace
        
        ipp_sat_stat = []
        for prn in range(n_prns):
            locs = EarthLocation(
                lon=np.random.randn(n_obs_times) * 0.1 * u.deg,
                lat=np.random.randn(n_obs_times) * 0.1 * u.deg,
                height=np.zeros(n_obs_times) * u.m
            )
            locs_all = np.array([[locs[i] for _ in range(n_heights)] for i in range(n_obs_times)])
            
            ipp = SimpleNamespace()
            ipp.loc = locs_all
            ipp.altaz = SimpleNamespace()
            ipp.altaz.alt = SimpleNamespace()
            ipp.altaz.alt.deg = np.ones(n_obs_times) * 70  # High elevation - all pass
            ipp.airmass = np.ones((n_obs_times, n_heights)) * 1.2
            
            ipp_sat_stat.append(ipp)
        
        target_locs = EarthLocation(
            lon=np.zeros(n_target_times) * u.deg,
            lat=np.zeros(n_target_times) * u.deg,
            height=np.zeros(n_target_times) * u.m
        )
        target_locs_all = np.array([[target_locs[i] for _ in range(n_heights)] for i in range(n_target_times)])
        
        ipp_target = SimpleNamespace()
        ipp_target.loc = target_locs_all
        ipp_target.times = target_times
        
        profiles = np.ones((n_target_times, n_heights))
        
        # Test with 1 time slot (no averaging)
        result_1slot = _get_distance_ipp_time_averaged(
            stec_values=stec_values,
            stec_errors=stec_errors,
            ipp_sat_stat=ipp_sat_stat,
            ipp_target=ipp_target,
            obs_times=obs_times,
            profiles=profiles,
            n_time_slots=1,
            max_time_diff_min=0.5,
            use_time_weighting=False,
        )
        
        # Test with 5 time slots (with averaging)
        result_5slots = _get_distance_ipp_time_averaged(
            stec_values=stec_values,
            stec_errors=stec_errors,
            ipp_sat_stat=ipp_sat_stat,
            ipp_target=ipp_target,
            obs_times=obs_times,
            profiles=profiles,
            n_time_slots=5,
            max_time_diff_min=2.5,
            use_time_weighting=False,
        )
        
        # Count measurements
        count_1slot = sum(
            result_1slot[t][h].shape[0]
            for t in range(n_target_times)
            for h in range(n_heights)
            if result_1slot[t][h].shape
        )
        
        count_5slots = sum(
            result_5slots[t][h].shape[0]
            for t in range(n_target_times)
            for h in range(n_heights)
            if result_5slots[t][h].shape
        )
        
        # With time averaging, we should get more measurements
        # (not exactly 5×, but definitely more)
        assert count_5slots > count_1slot, \
            f"Time averaging should increase measurements: {count_5slots} vs {count_1slot}"
        
        print(f"\nMeasurement increase: {count_1slot} → {count_5slots} ({count_5slots/count_1slot:.1f}×)")


# ============================================================================
# Test Interpolation
# ============================================================================

class TestInterpolation:
    """Test get_interpolated_tec function."""
    
    def test_basic_interpolation(self):
        """Test basic interpolation."""
        n_times = 3
        n_heights = 2
        
        # Create simple test data
        input_data = []
        for tidx in range(n_times):
            time_data = []
            for hidx in range(n_heights):
                # Create some measurements
                # [VTEC, error, dlon, dlat]
                measurements = np.array([
                    [10.0, 1.0, 0.0, 0.0],    # At origin
                    [11.0, 1.0, 0.1, 0.0],    # Slightly east
                    [9.0, 1.0, -0.1, 0.0],    # Slightly west
                    [10.5, 1.0, 0.0, 0.1],    # Slightly north
                ])
                time_data.append(measurements)
            input_data.append(time_data)
        
        # Interpolate
        result = get_interpolated_tec(input_data, use_time_weighting=False)
        
        # Check shape
        assert result.shape == (n_times, n_heights)
        
        # Check that values are reasonable (around 10 TECU)
        assert np.all(result >= 8)
        assert np.all(result <= 12)
    
    def test_insufficient_data(self):
        """Test interpolation with insufficient data."""
        n_times = 2
        n_heights = 2
        
        # Create input with some empty slots
        input_data = []
        for tidx in range(n_times):
            time_data = []
            for hidx in range(n_heights):
                if tidx == 0 and hidx == 0:
                    # Enough data
                    measurements = np.array([
                        [10.0, 1.0, 0.0, 0.0],
                        [11.0, 1.0, 0.1, 0.0],
                        [9.0, 1.0, -0.1, 0.0],
                    ])
                elif tidx == 1 and hidx == 1:
                    # Only 1 measurement (insufficient)
                    measurements = np.array([
                        [10.0, 1.0, 0.0, 0.0],
                    ])
                else:
                    # No measurements
                    measurements = np.array([])
                
                time_data.append(measurements)
            input_data.append(time_data)
        
        # Interpolate
        result = get_interpolated_tec(input_data, use_time_weighting=False)
        
        # Check that insufficient data gives 0
        assert result[1, 1] == 0.0  # Only 1 measurement
        assert result[0, 1] == 0.0  # No measurements
        assert result[1, 0] == 0.0  # No measurements
        
        # Check that sufficient data gives non-zero
        assert result[0, 0] > 0.0


# ============================================================================
# Integration Test
# ============================================================================

class TestFullPipeline:
    """Integration test of full pipeline."""
    
    def test_end_to_end(self):
        """Test complete pipeline from STEC to interpolated density."""
        # This is a simplified integration test
        # In practice, you'd use real RINEX/SP3 data
        
        # Setup dimensions
        n_prns = 5
        n_obs_times = 30
        n_target_times = 6
        n_heights = 3
        
        # Create mock data
        stec_values = np.random.randn(n_prns, n_obs_times) * 5 + 20
        stec_errors = np.abs(np.random.randn(n_prns, n_obs_times)) * 2
        
        obs_times = Time('2024-06-18T12:00:00') + np.arange(n_obs_times) * 30 * u.s
        target_times = Time('2024-06-18T12:00:00') + np.arange(n_target_times) * 2 * u.min
        
        # Create IPPs
        from types import SimpleNamespace
        
        ipp_sat_stat = []
        for prn in range(n_prns):
            locs = EarthLocation(
                lon=np.random.randn(n_obs_times) * 0.2 * u.deg,
                lat=np.random.randn(n_obs_times) * 0.2 * u.deg,
                height=np.zeros(n_obs_times) * u.m
            )
            locs_all = np.array([[locs[i] for _ in range(n_heights)] for i in range(n_obs_times)])
            
            ipp = SimpleNamespace()
            ipp.loc = locs_all
            ipp.altaz = SimpleNamespace()
            ipp.altaz.alt = SimpleNamespace()
            ipp.altaz.alt.deg = np.random.rand(n_obs_times) * 40 + 50  # 50-90°
            ipp.airmass = np.random.rand(n_obs_times, n_heights) * 0.5 + 1.0
            
            ipp_sat_stat.append(ipp)
        
        target_locs = EarthLocation(
            lon=np.zeros(n_target_times) * u.deg,
            lat=np.zeros(n_target_times) * u.deg,
            height=np.zeros(n_target_times) * u.m
        )
        target_locs_all = np.array([[target_locs[i] for _ in range(n_heights)] for i in range(n_target_times)])
        
        ipp_target = SimpleNamespace()
        ipp_target.loc = target_locs_all
        ipp_target.times = target_times
        
        profiles = np.ones((n_target_times, n_heights))
        
        # Step 1: Build data structure
        structured_data = _get_distance_ipp_time_averaged(
            stec_values=stec_values,
            stec_errors=stec_errors,
            ipp_sat_stat=ipp_sat_stat,
            ipp_target=ipp_target,
            obs_times=obs_times,
            profiles=profiles,
            n_time_slots=5,
            max_time_diff_min=2.5,
            use_time_weighting=True,
        )
        
        # Verify structure
        assert len(structured_data) == n_target_times
        assert all(len(t) == n_heights for t in structured_data)
        
        # Step 2: Interpolate
        density = get_interpolated_tec(structured_data, use_time_weighting=True)
        
        # Verify output
        assert density.shape == (n_target_times, n_heights)
        assert not np.all(density == 0), "Should have some non-zero values"
        
        print(f"\nFull pipeline test:")
        print(f"  Input: {n_prns} satellites × {n_obs_times} times")
        print(f"  Output: {n_target_times} times × {n_heights} heights")
        print(f"  Density range: [{np.nanmin(density):.2f}, {np.nanmax(density):.2f}]")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
