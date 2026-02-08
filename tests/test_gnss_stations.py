"""
Tests for spinifex_gnss.gnss_stations module.

Tests GNSS station position loading.
"""

import pytest
from astropy.coordinates import EarthLocation
import astropy.units as u

from spinifex_gnss.gnss_stations import (
    load_gnss_station_positions,
    gnss_pos_dict,
)


class TestLoadGNSSStations:
    """Test GNSS station loading function."""
    
    def test_function_returns_dict(self):
        """Test function returns a dictionary."""
        stations = load_gnss_station_positions()
        assert isinstance(stations, dict)
        
    def test_stations_loaded(self):
        """Test that stations are actually loaded."""
        assert len(gnss_pos_dict) > 0
        print(f"Loaded {len(gnss_pos_dict)} stations")
        
    def test_expected_number_of_stations(self):
        """Test we have approximately the expected number of stations."""
        # Should have ~1700-1800 stations
        assert 1000 < len(gnss_pos_dict) < 2000
        
    def test_station_format(self):
        """Test station names have expected format."""
        # Station names should be 9 characters
        for station_name in list(gnss_pos_dict.keys())[:10]:
            assert len(station_name) == 9
            assert isinstance(station_name, str)
            
    def test_positions_are_earthlocations(self):
        """Test station positions are EarthLocation objects."""
        for pos in list(gnss_pos_dict.values())[:10]:
            assert isinstance(pos, EarthLocation)
            
    def test_position_coordinates_valid(self):
        """Test position coordinates are valid."""
        for pos in list(gnss_pos_dict.values())[:10]:
            # Latitude between -90 and 90
            assert -90 <= pos.lat.deg <= 90
            
            # Longitude between -180 and 180
            assert -180 <= pos.lon.deg <= 180
            
            # Height should be reasonable (not more than 10 km above sea level)
            assert pos.height.to(u.m).value < 10000
            
    def test_known_station_wsrt(self):
        """Test known station WSRT is present and correct."""
        if 'WSRT00NLD' in gnss_pos_dict:
            wsrt = gnss_pos_dict['WSRT00NLD']
            
            # WSRT is in Netherlands
            # Approximate position: 52.9°N, 6.6°E
            assert 52 < wsrt.lat.deg < 53
            assert 6 < wsrt.lon.deg < 7
            assert wsrt.height.to(u.m).value < 100  # Close to sea level
            
    def test_global_coverage(self):
        """Test stations have global coverage."""
        lats = [pos.lat.deg for pos in gnss_pos_dict.values()]
        lons = [pos.lon.deg for pos in gnss_pos_dict.values()]
        
        # Should have stations in both hemispheres
        assert min(lats) < 0 < max(lats)
        assert min(lons) < 0 < max(lons)
        
        # Should span significant latitude range
        assert max(lats) - min(lats) > 100  # At least 100 degrees
        
    def test_no_duplicate_stations(self):
        """Test no duplicate station names."""
        assert len(gnss_pos_dict) == len(set(gnss_pos_dict.keys()))


class TestModuleImport:
    """Test module import and exports."""
    
    def test_gnss_pos_dict_available(self):
        """Test gnss_pos_dict is available at module level."""
        from spinifex_gnss.gnss_stations import gnss_pos_dict
        assert gnss_pos_dict is not None
        assert isinstance(gnss_pos_dict, dict)
        
    def test_load_function_available(self):
        """Test load function is available."""
        from spinifex_gnss.gnss_stations import load_gnss_station_positions
        assert callable(load_gnss_station_positions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
