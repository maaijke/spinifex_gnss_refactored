"""
Tests for spinifex_gnss.proces_gnss_data module.

Tests core processing functions.
"""

import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation

from spinifex_gnss.proces_gnss_data import (
    _get_distance_km,
    get_interpolated_tec,
)


class TestDistanceCalculation:
    """Test distance calculation function."""

    def test_same_location_zero_distance(self):
        """Test that same location gives zero distance."""
        loc = EarthLocation.from_geodetic(0 * u.deg, 50 * u.deg, 0 * u.m)

        dist = _get_distance_km(loc, loc)

        assert np.isclose(dist, 0.0, atol=1e-10)

    def test_distance_units(self):
        """Test distance is returned in km."""
        loc1 = EarthLocation.from_geodetic(0 * u.deg, 50 * u.deg, 0 * u.m)
        loc2 = EarthLocation.from_geodetic(1 * u.deg, 50 * u.deg, 0 * u.m)

        dist = _get_distance_km(loc1, loc2)

        # Should be ~70-90 km at this latitude
        assert 60 < dist < 100
        assert isinstance(dist, (float, np.ndarray))

    def test_array_input(self):
        """Test with array of locations."""
        locs1 = EarthLocation.from_geodetic(
            [0, 1, 2] * u.deg, [50, 50, 50] * u.deg, [0, 0, 0] * u.m
        )
        locs2 = EarthLocation.from_geodetic(0 * u.deg, 50 * u.deg, 0 * u.m)

        dist = _get_distance_km(locs1, locs2)

        assert dist.shape == (3,)
        assert dist[0] < dist[1] < dist[2]  # Increasing distance


class TestInterpolatedTEC:
    """Test spatial TEC interpolation."""



    def test_single_measurement_returns_value(self):
        """Test with single measurement at origin."""
        # Single measurement: [vtec, error, dlon, dlat]
        test_data = np.ones((5, 4), dtype=float)
        test_data[:, 0] = 10.0
        test_data[:, 2] = np.array((-0.2, -0.1, 0, 0.1, 0.2))
        test_data[:, 3] = np.array((-0.4, -0.1, 0, 0.1, 0.4))
        data = [[test_data]]  # 1 time, 1 height

        result = get_interpolated_tec(data)

        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 10.0)

    def test_insufficient_data_returns_zero(self):
        """Test with insufficient data points."""
        # Less than 2 measurements
        data = [[np.array([])]]  # No data

        result = get_interpolated_tec(data)

        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

    def test_multiple_times_heights(self):
        """Test with multiple times and heights."""
        # Create simple data structure
        # 2 times × 3 heights
        test_data = np.ones((5, 4), dtype=float)
        test_data[:, 0] = 10.0
        test_data[:, 2] = np.array((-0.2, -0.1, 0, 0.1, 0.2))
        test_data[:, 3] = np.array((-0.4, -0.1, 0, 0.1, 0.4))
        data = [
            [  # Time 1
                test_data,  # Height 1
                test_data,  # Height 2
                test_data,  # Height 3
            ],
            [  # Time 2
                test_data,  # Height 1
                test_data,  # Height 2
                test_data,  # Height 3
            ],
        ]

        result = get_interpolated_tec(data)

        assert result.shape == (2, 3)
        # Should have values for all times and heights
        assert np.all(result > 0)

    def test_weighted_by_inverse_variance(self):
        """Test that interpolation uses inverse variance weighting."""
        # Two measurements with different errors
        # Low error (high weight) at origin with value 10
        # High error (low weight) offset with value 50
        test_data = np.ones((5, 4), dtype=float)
        test_data[:, 0] = 10.0
        test_data[:, 2] = np.array((-0.2, -0.1, 0, 0.1, 0.2))
        test_data[:, 3] = np.array((-0.4, -0.1, 0, 0.1, 0.4))
        test_data[3:, 0] = 50.0
        test_data[3:, 1] = 50.0  # high error less influence
        data = [[test_data]]

        result = get_interpolated_tec(data)

        # Result should be closer to 10 than 50
        assert np.abs(result[0, 0] - 10.0) < np.abs(result[0, 0] - 50.0)


class TestNoDCBInModule:
    """Test that module has no DCB dependencies."""

    def test_no_dcb_parameters(self):
        """Test functions don't have DCB parameters."""
        from spinifex_gnss import proces_gnss_data

        # Check get_gnss_station_density signature
        import inspect

        sig = inspect.signature(proces_gnss_data.get_gnss_station_density)
        params = list(sig.parameters.keys())

        assert "dcb" not in params
        assert "gnss_data" in params
        assert "sp3_data" in params

    def test_no_dcb_imports(self):
        """Test module doesn't import DCB functions."""
        import spinifex_gnss.proces_gnss_data as module

        assert not hasattr(module, "parse_dcb_sinex")
        assert not hasattr(module, "DCBdata")
        assert not hasattr(module, "_get_dcb_value")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
