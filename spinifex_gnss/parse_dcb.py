"""
DCB (Differential Code Bias) file parser.

Parses BIAS-SINEX (BSX) format DCB files from CODE/IGS.
These contain satellite and receiver DCBs needed for absolute TEC calculation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import gzip
from typing import NamedTuple

_OBSERVABLE_PREFIXES = ("C", "L", "S", "P", "D")  # RINEX3 code observable first chars


def _is_observable(s: str) -> bool:
    """True if s looks like a RINEX3 observable code (e.g. C1W, L2P)."""
    return len(s) >= 2 and s[0] in _OBSERVABLE_PREFIXES and s[1].isdigit()


class DCBData(NamedTuple):
    """
    Container for DCB (Differential Code Bias) data.

    Attributes
    ----------
    satellite_dcb : Dict[str, Dict[str, float]]
        Satellite DCB values by PRN and observable pair.
        Format: satellite_dcb['G01']['C1C-C2W'] = dcb_nanoseconds
    receiver_dcb : Dict[str, float]
        Receiver DCB values by station and observable pair.
        Format: receiver_dcb['WSRT_C1C_C2W'] = dcb_nanoseconds
    """

    satellite_dcb: Dict[str, Dict[str, float]]
    receiver_dcb: Dict[str, float]


def parse_dcb_file(dcb_file: Path) -> DCBData:
    """
    Parse DCB file in BIAS-SINEX format.

    Format example:
    +BIAS/SOLUTION
    *BIAS SVN_ PRN STATION__ OBS1 OBS2 BIAS_START____ BIAS_END______ UNIT __ESTIMATED_VALUE____ _STD_DEV___
     DSB  G001 G01           C1W  C2W  2020:001:00000 2020:002:00000 ns                 -1.234        0.123
     DSB  G002 G02           C1W  C2W  2020:001:00000 2020:002:00000 ns                  0.567        0.089
     OSB       G01 WSRT00NLD C1W       2020:001:00000 2020:002:00000 ns                  2.345        0.234

    Parameters
    ----------
    dcb_file : Path
        Path to DCB file (.BSX or .BSX.gz)

    Returns
    -------
    satellite_dcb : Dict[str, float]
        Satellite DCBs in nanoseconds, key = "G01", "E05", etc.
    receiver_dcb : Dict[str, float]
        Receiver DCBs in nanoseconds, key = station name

    Notes
    -----
    - DSB = Differential Signal Bias (satellite, between two signals)
    - OSB = Observable-Specific Bias (receiver, single signal)
    - We need C1-C2 differential bias (DSB for satellites)
    - Units are converted from nanoseconds to meters for TEC calculation
    """
    satellite_dcb = {}
    receiver_dcb = {}

    # Open file (handle gzip)
    if dcb_file.suffix == ".gz":
        with gzip.open(dcb_file, "rt") as f:
            lines = f.readlines()
    else:
        with open(dcb_file, "r") as f:
            lines = f.readlines()

    # Find BIAS/SOLUTION block
    in_bias_block = False

    for line in lines:
        # Start of bias block
        if "+BIAS/SOLUTION" in line:
            in_bias_block = True
            continue

        # End of bias block
        if "-BIAS/SOLUTION" in line:
            in_bias_block = False
            continue

        # Skip header lines
        if not in_bias_block or line.startswith("*") or line.startswith("%"):
            continue

        # Parse bias line
        try:
            parts = line.split()
            if len(parts) < 10:
                continue

            bias_type = parts[0]  # DSB or OSB

            if bias_type == "DSB":
                # Differential Signal Bias
                # Two formats:
                # 1. Satellite: DSB SVN PRN (empty) OBS1 OBS2 ...
                #    After split(): parts[2]=PRN, parts[3]=OBS1, parts[4]=OBS2
                # 2. Receiver:  DSB SVN PRN STATION OBS1 OBS2 ...
                #    After split(): parts[3]=STATION, parts[4]=OBS1, parts[5]=OBS2

                # Detect which format by checking if parts[3] looks like a station name
                # Stations are typically 4-9 uppercase chars (e.g., WSRT, ONSA00SWE)
                # Observables start with C, L, S, etc.

                if len(parts) >= 10:
                    # Check if parts[3] is a station name (doesn't start with C/L/S/D)
                    if not _is_observable(parts[3]) and not parts[3].startswith("@"):
                        # Receiver DCB format
                        svn = parts[1]
                        prn = parts[2]
                        station = parts[3]
                        obs1 = parts[4]
                        obs2 = parts[5]
                        # Skip time fields parts[6], parts[7]
                        unit = parts[8]
                        value = float(parts[9])

                        # Store receiver DCB
                        if obs1[0] == "C" and obs2[0] == "C":
                            # Store as station_obs1_obs2 to allow multiple combinations per station
                            constellation = parts[1]  # 'E', 'G', 'C', 'R' etc.
                            key = f"{constellation}_{station}_{obs1}_{obs2}"
                            receiver_dcb[key] = value
                    else:
                        # Satellite DCB format (STATION field is empty)
                        svn = parts[1]
                        prn = parts[2]
                        obs1 = parts[3]
                        obs2 = parts[4]
                        # Skip time fields parts[5], parts[6]
                        unit = parts[7]
                        value = float(parts[8])

                        # Store satellite DCB
                        if obs1[0] == "C" and obs2[0] == "C":
                            if prn not in satellite_dcb:
                                satellite_dcb[prn] = {}
                            obs_pair = f"{obs1}-{obs2}"
                            satellite_dcb[prn][obs_pair] = value

            elif bias_type == "OSB":
                # Receiver observable-specific bias
                # Format: OSB _ PRN STATION OBS _ START END UNIT VALUE STD
                prn = parts[2]
                station = parts[3]
                obs = parts[4]
                value = float(parts[8])

                # For receiver, we need both C1 and C2 OSB to compute C1-C2
                # Store as station_OBS
                key = f"{station}_{obs}"
                receiver_dcb[key] = value  # nanoseconds

        except (ValueError, IndexError) as e:
            # Skip malformed lines
            continue

    return DCBData(satellite_dcb=satellite_dcb, receiver_dcb=receiver_dcb)


def get_satellite_dcb(
    satellite_dcb: Dict[str, Dict[str, float]], prn: str, obs1: str, obs2: str
) -> Optional[float]:
    """
    Get satellite DCB for specific observable pair.

    Matches EXACT observables used in RINEX.

    Parameters
    ----------
    satellite_dcb : Dict[str, Dict[str, float]]
        Satellite DCB dict from parse_dcb_file
    prn : str
        Satellite PRN (e.g., 'G01', 'E05', 'R12')
    obs1 : str
        First observable from RINEX (e.g., 'C1C', 'C1P')
    obs2 : str
        Second observable from RINEX (e.g., 'C2P', 'C2C')

    Returns
    -------
    float or None
        DCB in nanoseconds for this exact combination, or None

    Examples
    --------
    >>> # GLONASS satellite R12 with C1C-C2P combination
    >>> dcb = get_satellite_dcb(sat_dcb, 'R12', 'C1C', 'C2P')
    >>> # Returns DCB for C1C-C2P specifically
    >>>
    >>> # If station uses C1C-C2C instead:
    >>> dcb = get_satellite_dcb(sat_dcb, 'R12', 'C1C', 'C2C')
    >>> # Returns different DCB value (or None if not in file)

    Notes
    -----
    NO fallback - returns exact match only. This is important because:
    - GLONASS: C1C-C2P ≠ C1C-C2C ≠ C1P-C2P (different biases!)
    - GPS: C1W-C2W ≠ C1C-C2L (different tracking modes)
    - Galileo: C1C-C5Q ≠ C1X-C7Q (different signals)

    Using wrong DCB is worse than no DCB (falls back to GIM method).
    """
    if prn not in satellite_dcb:
        return None

    obs_pair = f"{obs1}-{obs2}"
    if obs_pair in satellite_dcb[prn]:
        return satellite_dcb[prn][obs_pair]

    # Reverse pair (file may store obs2-obs1)
    rev_pair = f"{obs2}-{obs1}"
    if rev_pair in satellite_dcb[prn]:
        return -satellite_dcb[prn][rev_pair]

    # Chain derivation: find a pivot X such that DSB(obs1,X) and DSB(X,obs2) exist
    # DSB(obs1, obs2) = DSB(obs1, X) + DSB(X, obs2)
    #                 = DSB(obs1, X) - DSB(obs2, X)
    for pair, val in satellite_dcb[prn].items():
        pivot1, pivot2 = pair.split('-')
        # Case: DSB(obs1, pivot1) exists and DSB(obs2, pivot1) exists
        # -> DSB(obs1,obs2) = DSB(obs1,pivot1) - DSB(obs2,pivot1)
        if pivot1 == obs1:   # stored as obs1-X: we have DSB(obs1, X)
            dsb_obs2_x = satellite_dcb[prn].get(f"{obs2}-{pivot2}")
            if dsb_obs2_x is not None:
                return val - dsb_obs2_x
        if pivot2 == obs1:   # stored as X-obs1: we have -DSB(obs1, X)
            dsb_obs2_x = satellite_dcb[prn].get(f"{obs2}-{pivot1}")
            if dsb_obs2_x is not None:
                return -val - dsb_obs2_x   # -DSB(obs1,X) - DSB(obs2,X) ... wrong
    # Simpler explicit chain: DSB(A,C) = DSB(A,B) - DSB(C,B)
    # Try all stored pairs as pivot
    stored = satellite_dcb[prn]
    for pivot in set(o for p in stored for o in p.split('-')):
        dsb_obs1_pivot = stored.get(f"{obs1}-{pivot}") or (
            -stored[f"{pivot}-{obs1}"] if f"{pivot}-{obs1}" in stored else None
        )
        dsb_obs2_pivot = stored.get(f"{obs2}-{pivot}") or (
            -stored[f"{pivot}-{obs2}"] if f"{pivot}-{obs2}" in stored else None
        )
        if dsb_obs1_pivot is not None and dsb_obs2_pivot is not None:
            return dsb_obs1_pivot - dsb_obs2_pivot
    return None


def get_receiver_dcb_c1c2(
    receiver_dcb: Dict[str, float],
    station: str,
    obs1: str = "C1W",
    obs2: str = "C2W",
    constellation: str = "G",
) -> Optional[float]:
    """
    Get receiver C1-C2 DCB.

    In the new format, receiver DCBs are stored directly as DSB entries,
    not as separate OSB entries that need to be subtracted.

    Parameters
    ----------
    receiver_dcb : Dict[str, float]
        Receiver DCB dictionary from parse_dcb_file
    station : str
        Station name (e.g., 'WSRT00NLD', 'WSRT', 'ONSA')
    obs1 : str
        First observable code (e.g., 'C1W', 'C1C')
    obs2 : str
        Second observable code (e.g., 'C2W', 'C2P')

    Returns
    -------
    float or None
        DCB in nanoseconds, or None if not available

    Notes
    -----
    The function tries multiple station name formats:
    - Exact match: WSRT00NLD
    - Short form: WSRT (first 4 chars)
    - This handles both 4-char and 9-char station names in DCB files
    """
    # Try exact match first
    key = f"{constellation}_{station}_{obs1}_{obs2}"
    if key in receiver_dcb:
        return receiver_dcb[key]

    # Try short station name
    if len(station) > 4:
        short_station = station[:4]
        key_short = f"{constellation}_{short_station}_{obs1}_{obs2}"
        if key_short in receiver_dcb:
            return receiver_dcb[key_short]

    # OSB fallback
    for sta in (station, station[:4] if len(station) > 4 else None):
        if sta is None:
            continue
        osb1 = receiver_dcb.get(f"{constellation}_{sta}_{obs1}")
        osb2 = receiver_dcb.get(f"{constellation}_{sta}_{obs2}")
        if osb1 is not None and osb2 is not None:
            return osb1 - osb2

    prefix = f"{constellation}_{station[:4] if len(station) > 4 else station}_"
    stored = {
        k[len(prefix):]: v
        for k, v in receiver_dcb.items()
        if k.startswith(prefix) and '_' in k[len(prefix):]
    }
    # stored keys are now like 'C1C_C2C', 'C1C_C1P'
    pivots = set(o for pair in stored for o in pair.split('_'))
    for pivot in pivots:
        dsb_obs1_pivot = stored.get(f"{obs1}_{pivot}") or (
            -stored[f"{pivot}_{obs1}"] if f"{pivot}_{obs1}" in stored else None
        )
        dsb_obs2_pivot = stored.get(f"{obs2}_{pivot}") or (
            -stored[f"{pivot}_{obs2}"] if f"{pivot}_{obs2}" in stored else None
        )
        if dsb_obs1_pivot is not None and dsb_obs2_pivot is not None:
            return dsb_obs1_pivot - dsb_obs2_pivot

    return None

def convert_dcb_to_tec(dcb_ns: float, f1_hz: float, f2_hz: float) -> float:
    """
    Convert DCB from nanoseconds to TEC units (TECU).

    The relationship:
    DCB affects pseudorange: P = ρ + I + DCB
    where I = 40.3/f² × TEC

    DCB contribution to differential pseudorange:
    ΔP = (40.3 × TEC / f1²) - (40.3 × TEC / f2²) + DCB

    So: TEC_offset = DCB / (40.3 × (1/f1² - 1/f2²))

    Parameters
    ----------
    dcb_ns : float
        DCB in nanoseconds
    f1_hz : float
        First frequency in Hz
    f2_hz : float
        Second frequency in Hz

    Returns
    -------
    float
        Equivalent TEC offset in TECU
    """
    # Convert nanoseconds to meters
    c = 299792458.0  # m/s
    dcb_m = dcb_ns * 1e-9 * c

    # Convert to TEC
    # TEC = DCB / (40.3 × (1/f1² - 1/f2²))
    tec_offset = dcb_m / (40.3 * (1.0 / f1_hz**2 - 1.0 / f2_hz**2))

    # Convert to TECU
    return tec_offset / 1e16
