"""
Configuration and constants for spinifex_gnss module.

This module centralizes all magic numbers, configuration values, and constants
used throughout the package.
"""

import astropy.units as u

# ============================================================================
# GNSS Observation Code Priorities
# ============================================================================

# Mapping of preferred observation codes for each constellation,
# listed in priority order (most preferred first)
GNSS_OBS_PRIORITY = {
    "G": {  # GPS
        "C1": ["C1W", "C1P", "C1C", "C1Y"],  # W/P(Y) > C/A
        "C2": ["C2W", "C2P", "C2Y", "C2L", "C2X"],
        "L1": ["L1W", "L1P", "L1Y", "L1C"],
        "L2": ["L2W", "L2P", "L2Y", "L2L", "L2X"],
    },
    "E": {  # Galileo
        "C1": ["C1C", "C1X"],
        "C2": ["C5Q", "C5X", "C7Q", "C7X"],
        "L1": ["L1C", "L1X"],
        "L2": ["L5Q", "L5X", "L7Q", "L7X"],
    },
    "R": {  # GLONASS
        "C1": ["C1P", "C1C"],
        "C2": ["C2P", "C2C"],
        "L1": ["L1P", "L1C"],
        "L2": ["L2P", "L2C"],
    },
    "C": {  # BeiDou
        "C1": ["C2I", "C2Q", "C2X"],
        "C2": ["C7I", "C7Q", "C7X", "C6I"],
        "L1": ["L2I", "L2Q", "L2X"],
        "L2": ["L7I", "L7Q", "L7X", "L6I"],
    },
    "J": {  # QZSS (same as GPS)
        "C1": ["C1C", "C1X"],
        "C2": ["C2L", "C2X"],
        "L1": ["L1C", "L1X"],
        "L2": ["L2L", "L2X"],
    },
}
# ============================================================================
# RINEX2 Observation Code Priorities (Legacy format)
# ============================================================================

GNSS_OBS_PRIORITY_RINEX2 = {
    "G": {  # GPS
        "C1": ["P1", "C1"],  # P(Y)-code preferred, C/A fallback
        "C2": ["P2"],  # P(Y)-code on L2 (required, no C/A on L2)
        "L1": ["L1"],  # L1 carrier phase
        "L2": ["L2"],  # L2 carrier phase
    },
    "R": {  # GLONASS
        "C1": ["P1", "C1"],  # Same structure as GPS
        "C2": ["P2"],
        "L1": ["L1"],
        "L2": ["L2"],
    },
    "E": {  # Galileo (limited support in RINEX2)
        "C1": ["C1"],  # E1 pseudorange
        "C2": ["C5", "C7"],  # E5a or E5b
        "L1": ["L1"],  # E1 carrier phase
        "L2": ["L5", "L7"],  # E5a or E5b phase
    },
    "C": {  # BeiDou (very limited support in RINEX2)
        # WARNING: BeiDou labels in RINEX2 are confusing!
        # "C2" means B1, not GPS L2!
        "C1": ["C2"],  # B1 pseudorange (confusing label)
        "C2": ["C7", "C6"],  # B2 or B3 pseudorange
        "L1": ["L2"],  # B1 carrier phase (confusing label)
        "L2": ["L7", "L6"],  # B2 or B3 phase
    },
    "J": {  # QZSS (rare in RINEX2)
        "C1": ["C1"],
        "C2": ["C2", "C5"],
        "L1": ["L1"],
        "L2": ["L2", "L5"],
    },
}


# ============================================================================
# GNSS Processing Constants
# ============================================================================

# Distance threshold for selecting GNSS stations (in km)
DISTANCE_KM_CUT = 200

# Number of distance points for interpolation
NDIST_POINTS = 300

# Minimum elevation angle for satellite observations (in degrees)
ELEVATION_CUT = 25

# Interpolation order for spatial interpolation
INTERPOLATION_ORDER = 3

# Default ionospheric pierce point height (in km)
DEFAULT_IONO_HEIGHT = 450 * u.km

# Minimum distance for GNSS station selection (in km)
MIN_DISTANCE_SELECT = 1500 * u.km

# ============================================================================
# Time Constants
# ============================================================================

# GPS time to UTC correction (in days)
# GPS time has been 18 seconds ahead of UTC since 2017-01-01
# This value is used for time conversions throughout the code
GPS_TO_UTC_CORRECTION_DAYS = 18.0 / (24.0 * 3600.0)

# Note: This is a simplification. The actual GPS-UTC offset changes with
# leap seconds. For precise work, use proper time conversion libraries.

# ============================================================================
# GNSS Frequency Definitions (in Hz)
# ============================================================================

FREQ = {
    "G": {  # GPS
        "f1": 1575.42e6,  # L1 frequency
        "f2": 1227.60e6,  # L2 frequency
    },
    "R": {  # GLONASS (nominal frequencies; actual frequencies vary by slot)
        "f1": 1602.00e6 + 9 * 0.5625e6,
        "f2": 1246.00e6 + 9 * 0.4375e6,
    },
    "E": {  # Galileo
        "f1": 1575.42e6,  # E1 frequency
        "f2": 1176.45e6,  # E5 frequency
    },
    "C": {  # BeiDou
        "f1": 1561.098e6,  # B1 frequency
        "f2": 1207.14e6,  # B2 frequency
    },
    "J": {  # QZSS (same as GPS)
        "f1": 1575.42e6,
        "f2": 1227.60e6,
    },
}

# ============================================================================
# Data Download Configuration
# ============================================================================

# Default server URLs for different data types
GNSS_SERVERS = {
    "satpos": "ftp://ftp.gfz-potsdam.de/GNSS/products/mgex/",
    "dcb": "https://data.bdsmart.cn/pub/product/bias/",
    "rinex": [
        "https://cddis.nasa.gov/archive/gnss/data/daily/",
        "https://www.epncb.oma.be/pub/obs/",
        "https://webring.gm.ingv.it:44324/rinex/RING/",
        "https://ga-gnss-data-rinex-v1.s3.amazonaws.com/index.html#public/daily/",
    ],
}

# Default data directory for downloads
DEFAULT_DATA_PATH = "../../GPS/data/"

# ============================================================================
# TEC Processing Constants
# ============================================================================

# Coefficient for TEC calculation from dual-frequency observations
# TEC_COEF = 1e-16 / (40.3 * (1/f1^2 - 1/f2^2))
# This is calculated dynamically based on constellation frequencies

# Cycle slip detection threshold (multiple of median difference)
CYCLE_SLIP_THRESHOLD = 15.0

# ============================================================================
# Parallel Processing Configuration
# ============================================================================

# Maximum number of worker processes for parallel RINEX processing
MAX_WORKERS_RINEX = 30

# Maximum number of worker processes for station density calculations
MAX_WORKERS_DENSITY = 30

# ============================================================================
# Data Quality Thresholds
# ============================================================================

# Minimum number of observations required per cycle slip segment
MIN_OBSERVATIONS_PER_SEGMENT = 10

# ============================================================================
# File Naming Conventions
# ============================================================================

# GNSS station position data file
GNSS_STATION_FILE = "data_gnss_pos.txt"

# SP3 file naming pattern (GBM0MGXRAP format)
SP3_FILE_PATTERN = "GBM0MGXRAP_{year}{doy:03d}0000_01D_05M_ORB.SP3.gz"

# Clock file naming pattern
CLK_FILE_PATTERN = "GBM0MGXRAP_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"

# DCB file naming pattern
DCB_FILE_PATTERN = "CAS0MGXRAP_{year}{doy:03d}0000_01D_01D_DCB.BSX.gz"

# RINEX file naming pattern
RINEX_FILE_PATTERN = "{station}_R_{year}{doy:03d}0000_01D_30S_MO.crx.gz"


def get_tec_coefficient(constellation: str) -> float:
    """
    Calculate the TEC coefficient for a given constellation.

    The coefficient is: 1e-16 / (40.3 * (1/f1^2 - 1/f2^2))

    Parameters
    ----------
    constellation : str
        Constellation identifier ('G', 'R', 'E', 'C', 'J')

    Returns
    -------
    float
        TEC coefficient for the constellation

    Raises
    ------
    KeyError
        If constellation is not recognized
    """
    if constellation not in FREQ:
        raise KeyError(f"Unknown constellation: {constellation}")

    f1 = FREQ[constellation]["f1"]
    f2 = FREQ[constellation]["f2"]

    return 1e-16 / (40.3 * (1.0 / f1**2 - 1.0 / f2**2))
