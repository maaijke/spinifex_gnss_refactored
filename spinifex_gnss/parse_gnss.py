"""
GNSS RINEX file parsing.

This module handles parsing of GNSS observation files and extracting
dual-frequency measurements for TEC calculation.

Refactored to remove:
- DCB parsing and dependencies
- Clock file parsing (not used)
"""

import numpy as np
from astropy.time import Time
from pathlib import Path
from typing import NamedTuple
import concurrent.futures

from spinifex_gnss.parse_rinex_enhanced import get_rinex_data_auto as get_rinex_data
from spinifex_gnss.parse_rinex import RinexData
from spinifex_gnss.config import (
    GNSS_OBS_PRIORITY,
    GNSS_OBS_PRIORITY_RINEX2,
    MAX_WORKERS_RINEX,
)


class GNSSData(NamedTuple):
    """Parsed GNSS observation data for one constellation."""

    gnss: dict[str, np.ndarray] | None
    """Dictionary mapping satellite PRN to observation array [n_epochs, 4] (C1, C2, L1, L2)"""
    c1_str: str
    """Label for pseudorange frequency 1"""
    c2_str: str
    """Label for pseudorange frequency 2"""
    l1_str: str
    """Label for carrier phase frequency 1"""
    l2_str: str
    """Label for carrier phase frequency 2"""
    station: str
    """GNSS station identifier"""
    is_valid: bool
    """Whether this contains valid observation data"""
    times: Time
    """Observation times (GPS time)"""
    constellation: str
    """Constellation identifier (G, E, R, C, J, I)"""
    tec_coefficients: dict[str, float] | None
    """Per-satellite TEC coefficients (for GLONASS FDMA). PRN -> C12"""


def dummy_gnss_data(station: str, constellation: str) -> GNSSData:
    """
    Create empty GNSSData for failed parsing.

    Parameters
    ----------
    station : str
        Station identifier
    constellation : str
        Constellation identifier

    Returns
    -------
    GNSSData
        Empty GNSSData with is_valid=False
    """
    return GNSSData(
        gnss=None,
        c1_str="",
        c2_str="",
        l1_str="",
        l2_str="",
        station=station,
        is_valid=False,
        times=None,
        constellation=constellation,
        tec_coefficients=None,
    )


def calculate_glonass_tec_coefficient(freq_channel: int) -> tuple[float, float, float]:
    """
    Calculate GLONASS TEC coefficient for specific frequency channel.

    Parameters
    ----------
    freq_channel : int
        GLONASS frequency channel k (-7 to +6)

    Returns
    -------
    tuple[float, float, float]
        (C12 coefficient, f1 in Hz, f2 in Hz)

    Notes
    -----
    GLONASS uses FDMA:
    - L1 = 1602.0 + k × 0.5625 MHz
    - L2 = 1246.0 + k × 0.4375 MHz

    Where k ranges from -7 to +6
    """
    # Calculate frequencies for this channel
    f1 = 1602.0e6 + freq_channel * 0.5625e6  # Hz
    f2 = 1246.0e6 + freq_channel * 0.4375e6  # Hz

    # Calculate TEC coefficient
    # C12 = (f1² × f2²) / (40.3e16 × (f1² - f2²))
    C12 = 1e-16 / (40.3 * (1.0 / f1**2 - 1.0 / f2**2))
    return C12, f1, f2


def _get_obs_code(rinex_data: RinexData, constellation: str, rxlabels: list[str]):
    try:

        # Filter for C1, C2, C5, L1, L2, L5 observations
        labels = [code for code in sorted(rxlabels) if code[1] in ["1", "2", "5"]]
        if rinex_data.header.version[0] == "3":
            c1c2_labels = GNSS_OBS_PRIORITY[constellation]
            # Find best available C1/C2 pair with corresponding L1/L2
            c_tracking = None

            # Try each priority combination
            for c1 in c1c2_labels["C1"]:
                for c2 in c1c2_labels["C2"]:
                    l1 = f"L{c1[-2:]}"
                    l2 = f"L{c2[-2:]}"

                    if c1 in labels and c2 in labels and l1 in labels and l2 in labels:
                        c_tracking = (c1, c2)
                        break
                if c_tracking:
                    c1_str, c2_str = c_tracking
                    l1_str = f"L{c1_str[-2:]}"
                    l2_str = f"L{c2_str[-2:]}"
                    break

            if not c_tracking:
                return None, None, None, None
        else:
            c1c2_labels = GNSS_OBS_PRIORITY_RINEX2[constellation]
            c1_str = [i for i in rxlabels if i in c1c2_labels["C1"]][0]
            c2_str = [i for i in rxlabels if i in c1c2_labels["C2"]][0]
            l1_str = [i for i in rxlabels if i in c1c2_labels["L1"]][0]
            l2_str = [i for i in rxlabels if i in c1c2_labels["L2"]][0]

        # Extract observation codes

        return c1_str, c2_str, l1_str, l2_str
    except:
        return None, None, None, None


def get_glonass_tec_coefficients(
    satellites: list[str], glonass_channels: dict[int, int]
) -> dict[str, float]:
    """
    Get TEC coefficients for GLONASS satellites.

    Parameters
    ----------
    satellites : list[str]
        List of GLONASS PRNs (e.g., ['R01', 'R02', 'R08'])
    glonass_channels : dict[int, int]
        Mapping from slot number to frequency channel k
        (from RINEX header "GLONASS SLOT / FRQ #")

    Returns
    -------
    dict[str, float]
        Mapping from PRN to TEC coefficient C12

    Notes
    -----
    Assumes PRN number = slot number (RINEX 3 convention):
    - R01 → slot 1
    - R02 → slot 2
    - R24 → slot 24

    Examples
    --------
    >>> glonass_channels = {1: -4, 2: -3, 8: 6}
    >>> sats = ['R01', 'R02', 'R08']
    >>> coeffs = get_glonass_tec_coefficients(sats, glonass_channels)
    >>> print(coeffs)
    {'R01': 0.162..., 'R02': 0.162..., 'R08': 0.162...}
    """
    tec_coefficients = {}

    for prn in satellites:
        # Extract slot number from PRN (R01 → 1, R02 → 2, etc.)
        slot = prn

        # Get frequency channel for this slot
        if slot in glonass_channels:
            freq_channel = glonass_channels[slot]
            C12, f1, f2 = calculate_glonass_tec_coefficient(freq_channel)
            tec_coefficients[prn] = (C12, f1,f2)
        else:
            # Fallback: use k=0 (average) if slot not in header
            C12, f1, f2 = calculate_glonass_tec_coefficient(0)
            tec_coefficients[prn] = (C12, f1,f2)
            print(f"  Warning: {prn} (slot {slot}) not in GLONASS channels, using k=0")

    return tec_coefficients


def get_gnss_data(gnss_file: list[Path], station: str) -> list[GNSSData]:
    """
    Parse GNSS RINEX files and extract dual-frequency observations.

    This function:
    1. Parses two consecutive days of RINEX data
    2. Identifies observation codes for each constellation
    3. Extracts C1, C2, L1, L2 observations
    4. Returns one GNSSData object per constellation

    Parameters
    ----------
    gnss_file : list[Path]
        List of two RINEX files for consecutive days
    station : str
        Station identifier

    Returns
    -------
    list[GNSSData]
        List of GNSSData objects, one per constellation found

    Notes
    -----
    Observation codes are selected based on GNSS_OBS_PRIORITY from config.
    This ensures we use the best available codes (e.g., GPS L1C/L2W preferred over L1C/L2C).

    No DCB corrections are applied - these are handled later if needed.

    Examples
    --------
    >>> files = [
    ...     Path("WSRT00NLD_R_20241690000_01D_30S_MO.crx.gz"),
    ...     Path("WSRT00NLD_R_20241700000_01D_30S_MO.crx.gz")
    ... ]
    >>> gnss_data_list = get_gnss_data(files, "WSRT00NLD")
    >>> for data in gnss_data_list:
    ...     if data.is_valid:
    ...         print(f"{data.constellation}: {len(data.gnss)} satellites")
    G: 12 satellites
    E: 8 satellites
    """
    # Parse RINEX files
    try:
        rinex_data = get_rinex_data(gnss_file[0])
        rinex_data_next_day = get_rinex_data(gnss_file[1])
    except Exception as e:
        print(f"RINEX parsing failed for station {station}: {e}")
        return []

    constellations = rinex_data.header.datatypes.keys()
    gnss_data_list = []
    # Get GLONASS frequency channels from header (if available)
    glonass_channels = (
        rinex_data.header.glonass_channels
        if hasattr(rinex_data.header, "glonass_channels")
        else {}
    )

    if glonass_channels:
        print(f"  Found GLONASS frequency channels: {len(glonass_channels)} slots")

    # Process each constellation separately
    for constellation in constellations:
        if not constellation in GNSS_OBS_PRIORITY.keys():
            continue
        try:
            # Get observation code priorities for this constellation
            rxlabels = rinex_data.header.datatypes[constellation]
            c1_str, c2_str, l1_str, l2_str = _get_obs_code(
                rinex_data, constellation, rxlabels
            )
            if c1_str is None:
                print(
                    f"No consistent observation codes found for {station} {constellation}"
                )
                gnss_data_list.append(dummy_gnss_data(station, constellation))
                continue
            # Get indices in observation array
            idx_c1 = rxlabels.index(c1_str)
            idx_c2 = rxlabels.index(c2_str)
            idx_l1 = rxlabels.index(l1_str)
            idx_l2 = rxlabels.index(l2_str)

            # Extract data for each satellite
            data = {}
            for key, rxdata in rinex_data.data.items():
                if key[0] == constellation:
                    # Concatenate two days of data
                    data[key] = np.concatenate(
                        (
                            rxdata[:, (idx_c1, idx_c2, idx_l1, idx_l2)],
                            rinex_data_next_day.data[key][
                                :, (idx_c1, idx_c2, idx_l1, idx_l2)
                            ],
                        ),
                        axis=0,
                    )
            # Calculate TEC coefficients (GLONASS only)
            tec_coefficients = None
            if constellation == "R" and glonass_channels:
                satellites = list(data.keys())
                tec_coefficients = get_glonass_tec_coefficients(
                    satellites, glonass_channels
                )
                print(
                    f"  GLONASS: Calculated TEC coefficients for {len(tec_coefficients)} satellites"
                )

            # Create GNSSData object
            gnss_data_list.append(
                GNSSData(
                    c1_str=c1_str,
                    c2_str=c2_str,
                    l1_str=l1_str,
                    l2_str=l2_str,
                    gnss=data,
                    station=station,
                    is_valid=True,
                    times=Time(
                        np.concatenate(
                            (rinex_data.times.mjd, rinex_data_next_day.times.mjd)
                        ),
                        format="mjd",
                    ),
                    constellation=constellation,
                    tec_coefficients=tec_coefficients,
                )
            )

        except Exception as e:
            print(f"Failed to process {station} {constellation}: {e}")
            gnss_data_list.append(dummy_gnss_data(station, constellation))

    return gnss_data_list


def process_all_rinex_parallel(
    rinex_files: list[tuple[Path, Path]], max_workers: int = MAX_WORKERS_RINEX
) -> list[GNSSData]:
    """
    Parse multiple RINEX file pairs in parallel.

    Parameters
    ----------
    rinex_files : list[tuple[Path, Path]]
        List of file pairs (day1, day2) for consecutive days
    max_workers : int, optional
        Maximum parallel workers, by default 20

    Returns
    -------
    list[GNSSData]
        Combined list of GNSSData from all stations

    Notes
    -----
    Uses ProcessPoolExecutor for parallel processing.
    Each station is processed independently.
    No DCB parameter needed!

    Examples
    --------
    >>> file_pairs = [
    ...     (Path("WSRT_day1.crx.gz"), Path("WSRT_day2.crx.gz")),
    ...     (Path("IJMU_day1.crx.gz"), Path("IJMU_day2.crx.gz"))
    ... ]
    >>> all_data = process_all_rinex_parallel(file_pairs)
    >>> valid_data = [d for d in all_data if d.is_valid]
    >>> print(f"Valid data from {len(valid_data)} constellations")
    """
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all parsing tasks
        futures = {
            executor.submit(get_gnss_data, rf, rf[0].stem[:9]): rf for rf in rinex_files
        }

        # Collect results as they complete
        for fut in concurrent.futures.as_completed(futures):
            try:
                results += fut.result()
            except Exception as e:
                print(f"Parallel processing error: {e}")

    return results
