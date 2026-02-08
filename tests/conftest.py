"""
Pytest configuration and fixtures for spinifex_gnss tests.
"""

import pytest
import numpy as np
from pathlib import Path
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u


@pytest.fixture
def sample_times():
    """Generate sample observation times."""
    base_time = Time('2024-06-18T12:00:00', scale='utc')
    return base_time + np.arange(24) * 5*u.min


@pytest.fixture
def sample_station():
    """Generate sample GNSS station location."""
    return EarthLocation.from_geodetic(
        lon=6.6*u.deg,
        lat=52.9*u.deg,
        height=16*u.m
    )


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir
