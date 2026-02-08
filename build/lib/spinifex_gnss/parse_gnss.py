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

from spinifex_gnss.parse_rinex import get_rinex_data, RinexData
from spinifex_gnss.config import GNSS_OBS_PRIORITY, MAX_WORKERS_RINEX


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
    )


def get_gnss_data(
    gnss_file: list[Path],
    station: str
) -> list[GNSSData]:
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
    
    # Process each constellation separately
    for constellation in constellations:
        try:
            # Get observation code priorities for this constellation
            c1c2_labels = GNSS_OBS_PRIORITY[constellation]
            rxlabels = rinex_data.header.datatypes[constellation]
            
            # Filter for C1, C2, C5, L1, L2, L5 observations
            labels = [
                code for code in sorted(rxlabels)
                if code[1] in ['1', '2', '5']
            ]
            
            # Find best available C1/C2 pair with corresponding L1/L2
            c_tracking = None
            
            # Try each priority combination
            for c1 in c1c2_labels["C1"]:
                for c2 in c1c2_labels["C2"]:
                    l1 = f"L{c1[-2:]}"
                    l2 = f"L{c2[-2:]}"
                    
                    if (c1 in labels and c2 in labels and
                        l1 in labels and l2 in labels):
                        c_tracking = (c1, c2)
                        break
                if c_tracking:
                    break
            
            if not c_tracking:
                print(f"No consistent observation codes found for {station} {constellation}")
                gnss_data_list.append(dummy_gnss_data(station, constellation))
                continue
            
            # Extract observation codes
            c1_str, c2_str = c_tracking
            l1_str = f"L{c1_str[-2:]}"
            l2_str = f"L{c2_str[-2:]}"
            
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
                            rinex_data_next_day.data[key][:, (idx_c1, idx_c2, idx_l1, idx_l2)],
                        ),
                        axis=0,
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
                )
            )
            
        except Exception as e:
            print(f"Failed to process {station} {constellation}: {e}")
            gnss_data_list.append(dummy_gnss_data(station, constellation))
    
    return gnss_data_list


def process_all_rinex_parallel(
    rinex_files: list[tuple[Path, Path]],
    max_workers: int = MAX_WORKERS_RINEX
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
            executor.submit(get_gnss_data, rf, rf[0].stem[:9]): rf
            for rf in rinex_files
        }
        
        # Collect results as they complete
        for fut in concurrent.futures.as_completed(futures):
            try:
                results += fut.result()
            except Exception as e:
                print(f"Parallel processing error: {e}")
    
    return results
