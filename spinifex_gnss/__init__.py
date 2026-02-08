"""
spinifex_gnss - GNSS-based ionospheric electron density calculations

Refactored version 2.0 - February 2026

Key improvements:
- Custom SP3 parser (replaced georinex)
- No DCB dependencies
- Removed unused pseudorange TEC function
- Cleaner, more maintainable code
- Comprehensive test coverage

Main Functions
--------------
get_electron_density_gnss : Calculate electron density from GNSS observations
select_gnss_stations : Find nearby GNSS stations
parse_sp3 : Parse satellite orbit files
get_rinex_data : Parse GNSS observation files
"""

__version__ = "2.0.0"
__author__ = "Maaijke Mevius"

# Main interface
from spinifex_gnss.gnss_tec import (
    get_electron_density_gnss,
    select_gnss_stations,
    get_min_distance,
)

# SP3 orbit file handling
from spinifex_gnss.parse_sp3 import (
    parse_sp3,
    concatenate_sp3_files,
    get_satellite_position,
    SP3Data,
    SP3Header,
)

# RINEX file handling  
from spinifex_gnss.parse_rinex import (
    get_rinex_data,
)

from spinifex_gnss.parse_gnss import (
    get_gnss_data,
    process_all_rinex_parallel,
    GNSSData,
)

# Geometry calculations
from spinifex_gnss.gnss_geometry import (
    get_sp3_data,
    get_sat_pos,
    get_azel_sat,
    get_stat_sat_ipp,
    interpolate_satellite,
    get_slant_distance,
)

# Core TEC calculations
from spinifex_gnss.tec_core import (
    getphase_tec,
    get_transmission_time,
)

# Processing
from spinifex_gnss.proces_gnss_data import (
    get_ipp_density,
    get_gnss_station_density,
    get_interpolated_tec,
)

# Configuration
from spinifex_gnss.config import (
    FREQ,
    GNSS_OBS_PRIORITY,
    get_tec_coefficient,
    GPS_TO_UTC_CORRECTION_DAYS,
)

__all__ = [
    "get_electron_density_gnss",
    "select_gnss_stations",
    "get_min_distance",
    "parse_sp3",
    "concatenate_sp3_files",
    "get_satellite_position",
    "get_rinex_data",
    "get_gnss_data",
    "process_all_rinex_parallel",
    "get_sp3_data",
    "get_sat_pos",
    "get_azel_sat",
    "get_stat_sat_ipp",
    "interpolate_satellite",
    "getphase_tec",
    "get_transmission_time",
    "get_ipp_density",
    "get_gnss_station_density",
    "get_interpolated_tec",
    "FREQ",
    "GNSS_OBS_PRIORITY",
    "get_tec_coefficient",
]
