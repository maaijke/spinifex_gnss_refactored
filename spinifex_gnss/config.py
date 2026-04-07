"""
Configuration and constants for spinifex_gnss module.

This module centralizes all magic numbers, configuration values, and constants
used throughout the package.
"""

import astropy.units as u
from enum import Enum


class RinexStrategy(Enum):
    """
    Strategy for removing the carrier-phase integer ambiguity bias.

    DCB_ONLY
        Use DCB file corrections exclusively. Satellites without both a
        satellite and receiver DCB entry are dropped. No IONEX download
        needed — fastest option and gives the most accurate absolute TEC
        when DCB values exist.

    GIM_ONLY
        Align every arc to the GIM (IONEX) map. Works for any satellite
        but is slower (IONEX download + per-arc interpolation) and limited
        by GIM accuracy (~2-5 TECU RMS).

    DCB_WITH_GIM_FALLBACK
        Use DCB where available; fall back to GIM for satellites that lack
        DCB values. Requires both DCB and IONEX downloads. 
    DCB_WITH_GIM_FALLBACK_AND_BIAS
        Use DCB where available; fall back to GIM for satellites that lack
        DCB values. Requires both DCB and IONEX downloads. 
        Tries to scale gim values based on dcb corrected arcs
   """

    DCB_ONLY = "dcb_only"
    GIM_ONLY = "gim_only"
    DCB_WITH_GIM_FALLBACK = "dcb_with_gim_fallback"
    DCB_WITH_GIM_FALLBACK_AND_BIAS = "dcb_with_gim_fallback_and_bias"


# ============================================================================
# GNSS Observation Code Priorities
# ============================================================================

# Mapping of preferred observation codes for each constellation,
# listed in priority order (most preferred first)
GNSS_OBS_PRIORITY = {
    "G": {  # GPS
        "C1": ["C1W", "C1C", "C1P", "C1Y"],  # W/P(Y) > C/A
        "C2": ["C2W", "C2P", "C2Y", "C2L", "C2X"],
        "L1": ["L1C", "L1W", "L1Y", "L1P"],
        "L2": ["L2W", "L2P", "L2Y", "L2L", "L2X"],
    },
    "E": {  # Galileo
        "C1": ["C1C", "C1X"],
        "C2": ["C5Q", "C5X", "C7Q", "C7X"],
        "L1": ["L1C", "L1X"],
        "L2": ["L5Q", "L5X", "L7Q", "L7X"],
    },
    "R": {  # GLONASS         #quick fix, remove glonass because of different frequencies
        "C1": ["C1C", "C1P"],  # C1C for pseudorange is fine, most DCB coverage
        "C2": ["C2C", "C2P"],  # C2C for pseudorange also fine
        "L1": ["L1C", "L1P"],  # L1C most common, L1P where available
        "L2": ["L2C", "L2P"],  # L2C must be allowed — L2P absent at LICC and WSRT
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
    # "R": {  # GLONASS   # Ignore Glonass RX2 for now untill we understand the stucture
    #    "C1": ["P1", "C1"],  # Same structure as GPS
    #    "C2": ["P2"],
    #    "L1": ["L1"],
    #    "L2": ["L2"],
    # },
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

DCB_UNRELIABLE_SATELLITES = {}
DCB_UNRELIABLE_PAIRS = {
    "G": {"C1C-C2W", "C1C-C2P", "C1C-C2L", "C1C-C2X"},
    "R": {"C1C-C2C", "C1C-C2P", "C1P-C2P", "C1P-C2C"},
}
# ============================================================================
# GNSS Processing Constants
# ============================================================================
# Minimum error floor added to DCB-corrected STEC (TECU).
# Represents the accuracy limit of the DCB file itself (~0.1-0.5 ns).
DCB_ERROR_FLOOR_TECU = 0.5

# Minimum error floor added to GIM-corrected STEC (TECU).
# Represents the systematic GIM uncertainty that does not average down with N.
# Ensures GIM measurements are always downweighted relative to DCB.
GIM_ERROR_FLOOR_TECU = 2.0
# Maximum allowed std of (pseudo_tec - phase_tec) for a segment to be used.
# Above this threshold, pseudorange multipath is too severe for reliable
# bias estimation. Typical clean pseudorange: 3-5 TECU. Threshold at 6 TECU
# rejects arcs with severe multipath while keeping slightly noisy data.
MAX_PSEUDO_PHASE_STD_TECU = 6.0
# Distance threshold for selecting GNSS stations (in km)
DISTANCE_KM_CUT = 500

# Number of distance points for interpolation
NDIST_POINTS = 300

# # Elevation cut used when ESTIMATING the phase bias (DCB or GIM alignment).
# Higher cut reduces pseudorange multipath and GIM mapping errors.
# Pseudorange noise roughly doubles below 30 deg vs above 40 deg.
ELEVATION_CUT_BIAS = 35.0

# Elevation cut applied when SELECTING phase stec data for the final fit.
# Can be lower because the phase observable itself is clean at any elevation.
ELEVATION_CUT = 20.0


# Interpolation order for spatial interpolation
INTERPOLATION_ORDER = 2

# Default ionospheric pierce point height (in km)
DEFAULT_IONO_HEIGHT = 450 * u.km

# Minimum distance for GNSS station selection (in km)
MIN_DISTANCE_SELECT = 1500 * u.km
# Minimum number of epoch-level (DCB_stec - GIM_stec) differences needed
# to trust the daily GIM bias estimate.
MIN_DCB_ARCS_FOR_GIM_BIAS = 20
# ============================================================================
# Time Constants
# ============================================================================

# GPS time to TAI correction in seconds
# needed because astropy.time does not have a gps scale
GPS_TO_TAI_SECONDS = 19

# ============================================================================
# GNSS Frequency Definitions (in Hz)
# ============================================================================

FREQ = {
    "G": {  # GPS
        "f1": 1575.42e6,  # L1 frequency
        "f2": 1227.60e6,  # L2 frequency
    },
    "R": {  # GLONASS (nominal frequencies; actual frequencies vary by slot)
        "f1": 1602.00e6,  # + 9 * 0.5625e6,
        "f2": 1246.00e6,  # + 9 * 0.4375e6,
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
        # "https://webring.gm.ingv.it:44324/rinex/RING/",
        "https://ga-gnss-data-rinex-v1.s3.amazonaws.com/public/daily/",
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
MIN_OBSERVATIONS_PER_SEGMENT = 50

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

    return 1e-16 / (40.3 * (1.0 / f2**2 - 1.0 / f1**2))
