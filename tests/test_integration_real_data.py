"""
Integration tests using real RINEX and SP3 data.

These tests use actual observation files:
- WSRT station RINEX files (June 17-18, 2024)
- GFZ Multi-GNSS SP3 orbit files (3 days)

Tests marked with @pytest.mark.requires_data will be skipped if data files
are not present.
"""

import pytest
import numpy as np
from pathlib import Path
from astropy.time import Time
import astropy.units as u

# Check if test data is available
TEST_DATA_DIR = Path(__file__).parent / "data"
RINEX_FILES_PRESENT = (
    TEST_DATA_DIR / "WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz"
).exists() and (TEST_DATA_DIR / "WSRT00NLD_R_20241700000_01D_30S_MO_crx.gz").exists()
SP3_FILES_PRESENT = (
    (TEST_DATA_DIR / "GBM0MGXRAP_20241680000_01D_05M_ORB_SP3.gz").exists()
    and (TEST_DATA_DIR / "GBM0MGXRAP_20241690000_01D_05M_ORB_SP3.gz").exists()
    and (TEST_DATA_DIR / "GBM0MGXRAP_20241700000_01D_05M_ORB_SP3.gz").exists()
)

pytestmark = pytest.mark.requires_data


@pytest.fixture
def rinex_files():
    """Paths to RINEX test files."""
    return [
        TEST_DATA_DIR / "WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz",
        TEST_DATA_DIR / "WSRT00NLD_R_20241700000_01D_30S_MO_crx.gz",
    ]


@pytest.fixture
def sp3_files():
    """Paths to SP3 test files."""
    return [
        TEST_DATA_DIR / "GBM0MGXRAP_20241680000_01D_05M_ORB_SP3.gz",
        TEST_DATA_DIR / "GBM0MGXRAP_20241690000_01D_05M_ORB_SP3.gz",
        TEST_DATA_DIR / "GBM0MGXRAP_20241700000_01D_05M_ORB_SP3.gz",
    ]


@pytest.mark.skipif(not SP3_FILES_PRESENT, reason="SP3 test files not available")
class TestSP3ParsingRealData:
    """Test SP3 parsing with real GFZ orbit files."""

    def test_parse_single_sp3_file(self, sp3_files):
        """Test parsing a single SP3 file."""
        from spinifex_gnss import parse_sp3

        sp3_data = parse_sp3(sp3_files[1])  # Day 169

        # Check header
        assert sp3_data.header.version == "d"
        assert sp3_data.header.coordinate_system == "IGS20"
        assert sp3_data.header.gps_week == 2319

        # Check we have epochs (should be ~288 for 5-min intervals)
        assert len(sp3_data.times) > 200
        assert len(sp3_data.times) < 300

        # Check we have satellite data
        assert len(sp3_data.positions) > 50  # Should have 100+ satellites

        # Check GPS satellites present
        gps_sats = [sat for sat in sp3_data.positions.keys() if sat.startswith("G")]
        assert len(gps_sats) >= 30  # At least 30 GPS satellites

    def test_concatenate_sp3_files(self, sp3_files):
        """Test concatenating multiple SP3 files."""
        from spinifex_gnss import concatenate_sp3_files

        combined = concatenate_sp3_files(sp3_files)

        # Should have ~864 epochs (288 × 3 days)
        assert len(combined.times) > 800
        assert len(combined.times) < 900

        # Should have same satellites as individual files
        assert len(combined.positions) > 50

    def test_get_sp3_data(self, sp3_files):
        """Test get_sp3_data function (used in workflow)."""
        from spinifex_gnss import get_sp3_data

        sp3_data = get_sp3_data(sp3_files)

        # Should return combined SP3Data
        assert sp3_data.times is not None
        assert len(sp3_data.positions) > 0

    def test_satellite_position_interpolation(self, sp3_files):
        """Test satellite position interpolation."""
        from spinifex_gnss import parse_sp3, get_satellite_position

        sp3_data = parse_sp3(sp3_files[1])

        # Get position for GPS satellite at arbitrary time
        sat_id = "G01"
        if sat_id in sp3_data.positions:
            # Interpolate at middle of time range
            mid_time = sp3_data.times[len(sp3_data.times) // 2]

            pos = get_satellite_position(sp3_data, sat_id, mid_time)

            # Check position is reasonable (satellite orbit ~20,000 km altitude)
            radius = np.sqrt(pos.x.value**2 + pos.y.value**2 + pos.z.value**2)
            assert 20e6 < radius < 30e6  # meters

    def test_sp3_positions_are_meters(self, sp3_files):
        """Test that SP3 positions are in meters (not km)."""
        from spinifex_gnss import parse_sp3

        sp3_data = parse_sp3(sp3_files[0])

        # Get first satellite position
        first_sat = list(sp3_data.positions.keys())[0]
        first_pos = sp3_data.positions[first_sat][0]

        # Positions should be in meters (order of 10^7)
        assert np.abs(first_pos[0]) > 1e6  # More than 1000 km
        assert np.abs(first_pos[0]) < 1e8  # Less than 100,000 km


@pytest.mark.skipif(not RINEX_FILES_PRESENT, reason="RINEX test files not available")
class TestRINEXParsingRealData:
    """Test RINEX parsing with real WSRT data."""

    def test_parse_rinex_file(self, rinex_files):
        """Test parsing a RINEX file."""
        from spinifex_gnss.parse_rinex import get_rinex_data

        rinex_data = get_rinex_data(rinex_files[0])

        # Check we have data
        assert rinex_data.data is not None
        assert len(rinex_data.data) > 0

        # Check we have GPS satellites
        gps_sats = [sat for sat in rinex_data.data.keys() if sat.startswith("G")]
        assert len(gps_sats) > 5  # At least 5 GPS satellites

    def test_get_gnss_data(self, rinex_files):
        """Test get_gnss_data extracts observations correctly."""
        from spinifex_gnss import get_gnss_data

        gnss_data_list = get_gnss_data(rinex_files, "WSRT00NLD")

        # Should have at least GPS data
        assert len(gnss_data_list) > 0

        # Find GPS data
        gps_data = [d for d in gnss_data_list if d.constellation == "G" and d.is_valid]
        assert len(gps_data) > 0

        gps = gps_data[0]

        # Check data structure
        assert gps.station == "WSRT00NLD"
        assert gps.gnss is not None
        assert len(gps.gnss) > 0

        # Check observation codes were selected
        assert gps.c1_str.startswith("C1")
        assert gps.c2_str.startswith("C2")
        assert gps.l1_str.startswith("L1")
        assert gps.l2_str.startswith("L2")

        # Check observation array shape (C1, C2, L1, L2)
        first_sat = list(gps.gnss.keys())[0]
        assert gps.gnss[first_sat].shape[1] == 4

    def test_observation_codes_priority(self, rinex_files):
        """Test that best observation codes are selected."""
        from spinifex_gnss import get_gnss_data

        gnss_data_list = get_gnss_data(rinex_files, "WSRT00NLD")

        gps_data = [d for d in gnss_data_list if d.constellation == "G" and d.is_valid]
        if gps_data:
            gps = gps_data[0]

            # WSRT should have W codes (encrypted P-code) which are preferred
            # If not available, should fall back to C codes
            assert gps.c1_str in ["C1W", "C1P", "C1C", "C1Y"]


@pytest.mark.skipif(
    not (RINEX_FILES_PRESENT and SP3_FILES_PRESENT),
    reason="Both RINEX and SP3 files required",
)
class TestIntegratedWorkflow:
    """Test integrated workflow with real data."""

    def test_parse_and_process_workflow(self, rinex_files, sp3_files):
        """Test complete parsing workflow."""
        from spinifex_gnss import get_gnss_data, get_sp3_data

        # Parse GNSS data
        gnss_data_list = get_gnss_data(rinex_files, "WSRT00NLD")
        valid_data = [d for d in gnss_data_list if d.is_valid]

        assert len(valid_data) > 0

        # Parse SP3 data
        sp3_data = get_sp3_data(sp3_files)

        assert sp3_data.times is not None
        assert len(sp3_data.positions) > 0

        # Check time overlap
        rinex_time_range = (valid_data[0].times.min(), valid_data[0].times.max())
        sp3_time_range = (sp3_data.times.min(), sp3_data.times.max())

        # SP3 should cover RINEX time range
        assert sp3_time_range[0] - 5.0 * u.min <= rinex_time_range[0]
        assert sp3_time_range[1] + 5.0 * u.min >= rinex_time_range[1]

    def test_satellite_geometry_calculation(self, rinex_files, sp3_files):
        """Test calculating satellite geometry from real data."""
        from spinifex_gnss import get_gnss_data, get_sp3_data, get_sat_pos
        from spinifex_gnss.gnss_stations import gnss_pos_dict

        # Parse data
        gnss_data_list = get_gnss_data(rinex_files, "WSRT00NLD")
        gps_data = [d for d in gnss_data_list if d.constellation == "G" and d.is_valid]

        if not gps_data:
            pytest.skip("No GPS data available")

        sp3_data = get_sp3_data(sp3_files)

        # Get first GPS satellite
        first_sat = list(gps_data[0].gnss.keys())[0]

        # Get satellite position at first observation time
        obs_time = gps_data[0].times[0]

        sat_pos = get_sat_pos(sp3_data, obs_time, first_sat)

        # Check position is reasonable
        assert sat_pos is not None

        # Calculate distance to WSRT
        if "WSRT00NLD" in gnss_pos_dict:
            wsrt_pos = gnss_pos_dict["WSRT00NLD"]

            from spinifex_gnss import get_slant_distance

            distance = get_slant_distance(sat_pos, wsrt_pos)

            # Distance should be ~20,000-25,000 km
            assert 15000 * u.km < distance < 30000 * u.km

    def test_tec_calculation_workflow(self, rinex_files, sp3_files):
        """Test TEC calculation from real data."""
        from spinifex_gnss import get_gnss_data, getphase_tec
        from spinifex_gnss.tec_core import get_transmission_time

        # Parse GNSS data
        gnss_data_list = get_gnss_data(rinex_files, "WSRT00NLD")
        gps_data = [d for d in gnss_data_list if d.constellation == "G" and d.is_valid]

        if not gps_data:
            pytest.skip("No GPS data available")

        gps = gps_data[0]
        first_sat = list(gps.gnss.keys())[0]
        sat_obs = gps.gnss[first_sat]

        # Calculate phase TEC
        phase_tec = getphase_tec(
            sat_obs[:, 2], sat_obs[:, 3], constellation="G"  # L1  # L2
        )

        # Check we get TEC values
        valid_tec = phase_tec[~np.isnan(phase_tec)]
        assert len(valid_tec) > 0

        # TEC should be reasonable (0-100 TECU typically)
        assert np.all(np.abs(valid_tec) < 200)

        # Calculate transmission time (no DCB!)
        tx_time = get_transmission_time(sat_obs[:, 1], gps.times)

        # Should be earlier than observation time
        assert np.all(tx_time.mjd <= gps.times.mjd)


class TestDataFilesInfo:
    """Test to document what data files we expect."""

    def test_data_directory_structure(self):
        """Document expected data directory structure."""
        expected_files = [
            "WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz",  # RINEX day 169
            "WSRT00NLD_R_20241700000_01D_30S_MO_crx.gz",  # RINEX day 170
            "GBM0MGXRAP_20241680000_01D_05M_ORB_SP3.gz",  # SP3 day 168
            "GBM0MGXRAP_20241690000_01D_05M_ORB_SP3.gz",  # SP3 day 169
            "GBM0MGXRAP_20241700000_01D_05M_ORB_SP3.gz",  # SP3 day 170
        ]

        # Just document - don't fail if missing
        print(f"\nExpected test data files in {TEST_DATA_DIR}:")
        for filename in expected_files:
            filepath = TEST_DATA_DIR / filename
            exists = "✓" if filepath.exists() else "✗"
            print(f"  {exists} {filename}")

    def test_data_file_metadata(self):
        """Document metadata about test data files."""
        info = {
            "Station": "WSRT (Westerbork), Netherlands",
            "Coordinates": "52.9°N, 6.6°E",
            "Dates": "June 17-18, 2024 (DOY 169-170)",
            "RINEX interval": "30 seconds",
            "SP3 source": "GFZ Multi-GNSS rapid orbits",
            "SP3 interval": "5 minutes",
            "Coordinate system": "IGS20/ITRF2020",
            "Constellations": "GPS, Galileo, GLONASS, BeiDou, QZSS",
        }

        print("\nTest data metadata:")
        for key, value in info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "requires_data"])
