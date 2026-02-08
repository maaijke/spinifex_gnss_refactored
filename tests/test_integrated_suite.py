"""
Integrated test suite for spinifex_gnss.

This combines the maintainer's existing tests with the new refactored code.
Tests are updated to work with:
- Custom SP3 parser (no georinex)
- No DCB dependencies (where applicable)
- Refactored modules
"""

import pytest
import numpy as np
from pathlib import Path
import glob
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

# Import refactored modules
from spinifex_gnss.parse_sp3 import parse_sp3, concatenate_sp3_files, get_satellite_position
from spinifex_gnss.gnss_geometry import (
    get_sp3_data, 
    interpolate_satellite,
    get_sat_pos, 
    get_azel_sat,
    get_stat_sat_ipp,
    filter_by_elevation,
    get_slant_distance
)
from spinifex_gnss.parse_rinex import get_rinex_data
from spinifex_gnss.config import FREQ, GPS_TO_UTC_CORRECTION_DAYS


# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"

class TestParseRinex:
    """Tests for RINEX file parsing."""
    
    @pytest.mark.requires_data
    def test_get_rinex_data(self):
        """Test parsing a real RINEX file."""
        rinex_file = TEST_DATA_DIR / "WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz"
        
        if not rinex_file.exists():
            pytest.skip(f"Test data not found: {rinex_file}")
        
        rinex_data = get_rinex_data(rinex_file)
        
        assert rinex_data is not None
        assert rinex_data.times is not None
        assert len(rinex_data.times) > 0
        assert rinex_data.data is not None
        assert len(rinex_data.data) > 0
        
    @pytest.mark.requires_data
    def test_rinex_header_parsing(self):
        """Test that RINEX header is parsed correctly."""
        rinex_file = TEST_DATA_DIR / "WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz"
        
        if not rinex_file.exists():
            pytest.skip(f"Test data not found: {rinex_file}")
        
        rinex_data = get_rinex_data(rinex_file)
        
        assert rinex_data.header is not None
        assert rinex_data.header.version is not None
        assert rinex_data.header.datatypes is not None
        assert len(rinex_data.header.datatypes) > 0
        
    @pytest.mark.requires_data
    def test_rinex_data_structure(self):
        """Test that RINEX data has expected structure."""
        rinex_file = TEST_DATA_DIR / "WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz"
        
        if not rinex_file.exists():
            pytest.skip(f"Test data not found: {rinex_file}")
        
        rinex_data = get_rinex_data(rinex_file)
        
        # Check that data contains satellite observations
        for prn, obs_data in rinex_data.data.items():
            assert isinstance(prn, str)
            assert len(prn) == 3  # e.g., 'G01', 'E05'
            assert isinstance(obs_data, np.ndarray)
            assert obs_data.shape[0] == len(rinex_data.times)


class TestSP3ParsingWithRealData:
    """Tests for SP3 parsing with real data files."""
    
    @pytest.mark.requires_data
    def test_parse_real_sp3_file(self):
        """Test parsing a real SP3 file."""
        sp3_files = sorted(glob.glob(str(TEST_DATA_DIR / "*SP3.gz")))
        
        if not sp3_files:
            pytest.skip("No SP3 files found in test data")
        
        sp3_data = parse_sp3(Path(sp3_files[0]))
        
        assert sp3_data is not None
        assert len(sp3_data.times) > 0
        assert len(sp3_data.positions) > 0
        
    @pytest.mark.requires_data
    def test_sp3_satellite_coverage(self):
        """Test that SP3 file contains expected satellites."""
        sp3_files = sorted(glob.glob(str(TEST_DATA_DIR / "*SP3.gz")))
        
        if not sp3_files:
            pytest.skip("No SP3 files found in test data")
        
        sp3_data = parse_sp3(Path(sp3_files[0]))
        
        # Should have GPS satellites
        gps_sats = [sat for sat in sp3_data.positions.keys() if sat.startswith('G')]
        assert len(gps_sats) > 0
        
        # Common GPS satellites should be present
        for sat_id in ['G01', 'G02', 'G03']:
            if sat_id in sp3_data.header.satellite_ids:
                assert sat_id in sp3_data.positions
                
    @pytest.mark.requires_data
    def test_concatenate_real_sp3_files(self):
        """Test concatenating multiple real SP3 files."""
        sp3_files = sorted(glob.glob(str(TEST_DATA_DIR / "*SP3.gz")))
        
        if len(sp3_files) < 2:
            pytest.skip("Need at least 2 SP3 files for concatenation test")
        
        # Take first 3 files (or all if less than 3)
        files_to_concat = [Path(f) for f in sp3_files[:min(3, len(sp3_files))]]
        
        combined = concatenate_sp3_files(files_to_concat)
        
        assert combined is not None
        assert len(combined.times) > len(parse_sp3(files_to_concat[0]).times)


class TestGNSSGeometry:
    """Tests for GNSS geometry calculations with real data."""
    
    @pytest.mark.requires_data
    def test_get_sp3_data(self):
        """Test loading SP3 data using refactored function."""
        sp3_files = [Path(i) for i in sorted(glob.glob(str(TEST_DATA_DIR / "*SP3.gz")))]
        
        if len(sp3_files) < 3:
            pytest.skip("Need at least 3 SP3 files")
        
        sp3_data = get_sp3_data(sp3_files[:3])
        
        assert sp3_data is not None
        assert len(sp3_data.times) > 0
        assert len(sp3_data.positions) > 0
        
    @pytest.mark.requires_data
    def test_get_sat_pos_refactored(self):
        """Test getting satellite position with refactored code."""
        sp3_files = [Path(i) for i in sorted(glob.glob(str(TEST_DATA_DIR / "*SP3.gz")))]
        
        if len(sp3_files) < 3:
            pytest.skip("Need at least 3 SP3 files")
        
        sp3_data = get_sp3_data(sp3_files[:3])
        
        # Create test times
        times = sp3_data.times[0:1] + np.arange(4) * 13 * u.min
        
        # Get position for first available satellite
        sat_id = list(sp3_data.positions.keys())[0]
        sat_pos = get_sat_pos(sp3_data, times, sat_id)
        
        assert sat_pos is not None
        assert isinstance(sat_pos, EarthLocation)
        assert len(sat_pos.x) == len(times)
        
    @pytest.mark.requires_data
    def test_get_azel_sat(self):
        """Test azimuth/elevation calculation."""
        sp3_files = [Path(i) for i in sorted(glob.glob(str(TEST_DATA_DIR / "*SP3.gz")))]
        
        if len(sp3_files) < 3:
            pytest.skip("Need at least 3 SP3 files")
        
        sp3_data = get_sp3_data(sp3_files[:3])
        times = sp3_data.times[0:1] + np.arange(4) * 13 * u.min
        
        sat_id = list(sp3_data.positions.keys())[0]
        sat_pos = get_sat_pos(sp3_data, times, sat_id)
        
        # WSRT position
        receiver_pos = EarthLocation.from_geodetic(6.6 * u.deg, 52.9 * u.deg, 0 * u.m)
        
        azel = get_azel_sat(sat_pos, receiver_pos, times)
        
        assert azel is not None
        assert len(azel.az) == len(times)
        assert len(azel.alt) == len(times)
        
        # Azimuth should be 0-360 degrees
        assert np.all(azel.az.deg >= 0)
        assert np.all(azel.az.deg < 360)
        
        # Elevation can be negative (below horizon)
        assert np.all(azel.alt.deg >= -90)
        assert np.all(azel.alt.deg <= 90)
        
    @pytest.mark.requires_data
    def test_get_stat_sat_ipp(self):
        """Test ionospheric pierce point calculation."""
        sp3_files = [Path(i) for i in sorted(glob.glob(str(TEST_DATA_DIR / "*SP3.gz")))]
        
        if len(sp3_files) < 3:
            pytest.skip("Need at least 3 SP3 files")
        
        sp3_data = get_sp3_data(sp3_files[:3])
        times = sp3_data.times[0:1] + np.arange(4) * 13 * u.min
        
        sat_id = list(sp3_data.positions.keys())[0]
        sat_pos = get_sat_pos(sp3_data, times, sat_id)
        
        receiver_pos = EarthLocation.from_geodetic(6 * u.deg, 52 * u.deg, 0 * u.m)
        
        ipp = get_stat_sat_ipp(sat_pos, receiver_pos, times)
        
        assert ipp is not None
        assert ipp.loc is not None
        assert ipp.times is not None
        assert len(ipp.times) == len(times)
        
    @pytest.mark.requires_data
    def test_filter_by_elevation(self):
        """Test elevation filtering."""
        sp3_files = [Path(i) for i in sorted(glob.glob(str(TEST_DATA_DIR / "*SP3.gz")))]
        
        if len(sp3_files) < 3:
            pytest.skip("Need at least 3 SP3 files")
        
        sp3_data = get_sp3_data(sp3_files[:3])
        times = sp3_data.times[0:1] + np.arange(20) * 5 * u.min
        
        sat_id = list(sp3_data.positions.keys())[0]
        sat_pos = get_sat_pos(sp3_data, times, sat_id)
        
        receiver_pos = EarthLocation.from_geodetic(6 * u.deg, 52 * u.deg, 0 * u.m)
        azel = get_azel_sat(sat_pos, receiver_pos, times)
        
        # Filter for elevation > 20 degrees
        mask = filter_by_elevation(azel, min_elevation=20.0)
        
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(times)
        
        # All selected elevations should be >= 20 degrees
        if np.any(mask):
            assert np.all(azel.alt[mask].deg >= 20.0)


class TestTECCalculationsNoDCB:
    """Tests for TEC calculations without DCB dependencies."""
    
        
    def test_phase_tec_calculation(self):
        """Test carrier phase TEC calculation."""
        # Create sample data
        l1 = np.array([125000000.0, 125005000.0, 125010000.0])
        l2 = np.array([120000000.0, 120005000.0, 120010000.0])
        
        from astropy.constants import c as speed_light
        from spinifex_gnss.config import get_tec_coefficient
        
        C12 = get_tec_coefficient('G')
        f1 = FREQ['G']['f1']
        f2 = FREQ['G']['f2']
        WL1 = speed_light.value / f1
        WL2 = speed_light.value / f2
        
        phase_tec = -C12 * (l1 * WL1 - l2 * WL2)
        
        assert phase_tec is not None
        assert len(phase_tec) == 3
        assert np.all(np.isfinite(phase_tec))




class TestIntegrationWorkflow:
    """Integration tests for complete workflows."""
    
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_complete_sp3_workflow(self):
        """Test complete workflow from SP3 files to satellite positions."""
        sp3_files = [Path(i) for i in sorted(glob.glob(str(TEST_DATA_DIR / "*SP3.gz")))]
        
        if len(sp3_files) < 3:
            pytest.skip("Need at least 3 SP3 files")
        
        # Load SP3 data
        sp3_data = get_sp3_data(sp3_files[:3])
        
        # Define observation scenario
        times = sp3_data.times[100:110]  # 10 epochs
        receiver_pos = EarthLocation.from_geodetic(6 * u.deg, 52 * u.deg, 0 * u.m)
        
        # Get satellite positions
        sat_id = 'G01' if 'G01' in sp3_data.positions else list(sp3_data.positions.keys())[0]
        sat_pos = get_sat_pos(sp3_data, times, sat_id)
        
        # Calculate az/el
        azel = get_azel_sat(sat_pos, receiver_pos, times)
        
        # Calculate IPPs
        ipp = get_stat_sat_ipp(sat_pos, receiver_pos, times, height_array=np.array([450]) * u.km)
        
        # Verify complete chain
        assert sat_pos is not None
        assert azel is not None
        assert ipp is not None
        assert len(ipp.times) == len(times)
        
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_rinex_to_tec_workflow(self):
        """Test workflow from RINEX file to TEC values."""
        rinex_file = TEST_DATA_DIR / "WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz"
        
        if not rinex_file.exists():
            pytest.skip(f"Test data not found: {rinex_file}")
        
        # Parse RINEX
        rinex_data = get_rinex_data(rinex_file)
        
        # Extract some satellite data
        prn = list(rinex_data.data.keys())[0]
        sat_data = rinex_data.data[prn]
        
        # Calculate TEC (assuming data has C1, C2, L1, L2 in that order)
        if sat_data.shape[1] >= 4:
            c1 = sat_data[:, 0]
            c2 = sat_data[:, 1]
            
            from spinifex_gnss.config import get_tec_coefficient
            constellation = prn[0]
            
            if constellation in FREQ:
                C12 = get_tec_coefficient(constellation)
                tec = C12 * (c1 - c2)
                
                # Should have TEC values
                valid_tec = tec[~np.isnan(tec)]
                assert len(valid_tec) > 0




if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
