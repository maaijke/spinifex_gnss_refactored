"""
RINEX2 parser to complement the existing RINEX3 parser.

This module adds support for the older RINEX2 format with automatic
format detection and unified output.
"""

import hatanaka
from pathlib import Path
import numpy as np
from typing import NamedTuple, Any
from astropy.time import Time


# Import existing RINEX3 structures
from spinifex_gnss.parse_rinex import RinexHeader, RinexData, get_rinex_data as get_rinex3_data


def _read_rinex2_header(raw_rinex_lines: list[str]) -> tuple[RinexHeader, int]:
    """
    Read header information from RINEX2 file.
    
    RINEX2 format differences from RINEX3:
    - No 'SYS / # / OBS TYPES' - uses '# / TYPES OF OBSERV'
    - Observation types are global (not per constellation)
    - Different header structure
    
    Parameters
    ----------
    raw_rinex_lines : list[str]
        All lines in the file
        
    Returns
    -------
    RinexHeader
        Object with RINEX version and datatypes
    int
        Line number where header ends
    """
    version = ""
    obs_types = []
    
    for line_number, line in enumerate(raw_rinex_lines):
        label = line[60:].strip()
        
        if "END OF HEADER" in label:
            # In RINEX2, observation types are global for all constellations
            # We'll assign them to common constellation types
            obs_map = {}
            if obs_types:
                # Assign same types to all common constellations
                for constellation in ['G', 'R', 'E', 'C', 'J', 'S', 'I']:
                    obs_map[constellation] = obs_types
            
            return RinexHeader(version=version, datatypes=obs_map), line_number + 1
        
        if "RINEX VERSION / TYPE" in label:
            version = line[:20].strip()
        
        elif "# / TYPES OF OBSERV" in label:
            # RINEX2 format: number of types, then type codes
            n_types = int(line[:6].strip())
            types = line[6:60].split()
            obs_types.extend(types)
            
            # Check for continuation lines
            if len(obs_types) < n_types:
                # Read continuation lines
                continue_line_num = line_number + 1
                while len(obs_types) < n_types and continue_line_num < len(raw_rinex_lines):
                    cont_line = raw_rinex_lines[continue_line_num]
                    if "# / TYPES OF OBSERV" in cont_line[60:]:
                        obs_types.extend(cont_line[6:60].split())
                        continue_line_num += 1
                    else:
                        break
    
    return None, None


def get_rinex2_data(fname: Path) -> RinexData:
    """
    Parse RINEX2 file.
    
    RINEX2 epoch format:
     YY MM DD HH MM SS.SSSSSSS  0  N (where N = number of satellites)
     [satellite list]
     [observations for each satellite]
    
    Parameters
    ----------
    fname : Path
        Path to RINEX2 file (Hatanaka compressed or regular)
        
    Returns
    -------
    RinexData
        Object with data, times (GPS time) and header
        
    Notes
    -----
    RINEX2 differences:
    - 2-digit year (requires century handling)
    - Epoch flag 0 = OK, 1 = power failure, >1 = special events
    - Observations formatted differently
    - 14-character field width (vs 16 in RINEX3)
    """
    # Decompress if needed
    rinex_lines = hatanaka.decompress(fname).decode().split("\n")
    
    # Parse header
    header, end_of_header = _read_rinex2_header(rinex_lines)
    
    if header is None:
        raise ValueError(f"Failed to parse RINEX2 header in {fname}")
    
    # Determine field width (typically 14 for RINEX2, but can vary)
    width = 14  # Standard RINEX2 field width
    
    # Parse data
    cur_time = None
    all_times = []
    data = {}
    
    i = end_of_header
    while i < len(rinex_lines):
        line = rinex_lines[i]
        
        # Check for epoch line
        # RINEX2 epoch format: " YY MM DD HH MM SS.SSSSSSS  0  N"
        if len(line) >= 26 and line[0:2].strip().isdigit():
            try:
                # Parse epoch time
                year = int(line[0:3].strip())
                month = int(line[3:6].strip())
                day = int(line[6:9].strip())
                hour = int(line[9:12].strip())
                minute = int(line[12:15].strip())
                second = float(line[15:26].strip())
                
                # Handle 2-digit year (assume 1980-2079)
                if year < 80:
                    year += 2000
                else:
                    year += 1900
                
                # Epoch flag (usually 0 for OK)
                epoch_flag = int(line[26:29].strip())
                
                # Number of satellites
                n_sat = int(line[29:32].strip())
                
                # Skip if not normal epoch
                if epoch_flag != 0:
                    i += 1
                    continue
                
                # Create time
                cur_time = Time(f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:011.8f}")
                all_times.append(cur_time.mjd)
                
                # Read satellite list (may span multiple lines)
                sat_list = []
                sat_line = line[32:68]  # First 12 satellites on epoch line
                sat_list.extend([sat_line[j:j+3].strip() for j in range(0, len(sat_line), 3) if sat_line[j:j+3].strip()])
                
                # Check for continuation lines (if more than 12 satellites)
                while len(sat_list) < n_sat:
                    i += 1
                    if i >= len(rinex_lines):
                        break
                    cont_line = rinex_lines[i]
                    sat_line = cont_line[32:68]
                    sat_list.extend([sat_line[j:j+3].strip() for j in range(0, len(sat_line), 3) if sat_line[j:j+3].strip()])
                
                # Ensure we have correct number of satellites
                sat_list = sat_list[:n_sat]
                
                # Read observations for each satellite
                i += 1
                for sat_id in sat_list:
                    # Determine number of observation types for this constellation
                    constellation = sat_id[0] if sat_id else 'G'
                    if constellation not in header.datatypes:
                        constellation = 'G'  # Default to GPS
                    
                    n_types = len(header.datatypes[constellation])
                    
                    # Observations may span multiple lines (usually 5 obs per line)
                    obs_vals = []
                    n_lines = (n_types + 4) // 5  # Round up
                    
                    for _ in range(n_lines):
                        if i >= len(rinex_lines):
                            break
                        obs_line = rinex_lines[i]
                        
                        # Parse observations (14-character fields in RINEX2)
                        # Format: XXXXXX.XXX  (10 digits + decimal + 3 decimals)
                        for j in range(0, min(5, n_types - len(obs_vals))):
                            field_start = j * width
                            field_end = field_start + width
                            field = obs_line[field_start:field_end].strip()
                            
                            if field:
                                try:
                                    # Extract value (first part before LLI and signal strength)
                                    val_str = field.split()[0] if field.split() else ''
                                    obs_vals.append(float(val_str) if val_str else np.nan)
                                except:
                                    obs_vals.append(np.nan)
                            else:
                                obs_vals.append(np.nan)
                        
                        i += 1
                    
                    # Pad with NaN if needed
                    while len(obs_vals) < n_types:
                        obs_vals.append(np.nan)
                    
                    # Store data
                    if sat_id not in data:
                        data[sat_id] = {"time": [cur_time], "data": [obs_vals]}
                    else:
                        data[sat_id]["time"].append(cur_time)
                        data[sat_id]["data"].append(obs_vals)
            
            except Exception as e:
                print(f"Error parsing epoch at line {i}: {e}")
                i += 1
                continue
        else:
            i += 1
    
    # Convert to uniform time grid (same as RINEX3 parser)
    all_times = np.array(all_times)
    newdata = {}
    
    for prn, prndata in data.items():
        if not prndata["data"]:
            continue
        
        alldata = np.empty((len(all_times), len(prndata["data"][0])))
        alldata.fill(np.nan)
        
        for tm, dt in zip(prndata["time"], prndata["data"]):
            tm_idx = np.argmin(np.abs(all_times - tm.mjd))
            alldata[tm_idx] = dt
        
        newdata[prn] = alldata
    
    return RinexData(header=header, times=Time(all_times, format="mjd"), data=newdata)


def detect_rinex_version(fname: Path) -> int:
    """
    Detect RINEX version from file.
    
    Parameters
    ----------
    fname : Path
        Path to RINEX file
        
    Returns
    -------
    int
        RINEX version (2 or 3)
        
    Raises
    ------
    ValueError
        If version cannot be determined
    """
    try:
        # Decompress and read first few lines
        rinex_lines = hatanaka.decompress(fname).decode().split("\n")
        
        # Look for RINEX VERSION line in header
        for line in rinex_lines[:50]:  # Check first 50 lines
            if "RINEX VERSION" in line[60:]:
                version_str = line[:20].strip()
                if version_str.startswith('2'):
                    return 2
                elif version_str.startswith('3'):
                    return 3
                else:
                    # Try to parse as float
                    version_float = float(version_str.split()[0])
                    if version_float < 3.0:
                        return 2
                    else:
                        return 3
        
        raise ValueError("RINEX VERSION not found in header")
    
    except Exception as e:
        raise ValueError(f"Failed to detect RINEX version: {e}")


def get_rinex_data_auto(fname: Path) -> RinexData:
    """
    Parse RINEX file with automatic version detection.
    
    This is the recommended function to use - it automatically
    detects RINEX2 vs RINEX3 and uses the appropriate parser.
    
    Parameters
    ----------
    fname : Path
        Path to RINEX file (any version, Hatanaka compressed or not)
        
    Returns
    -------
    RinexData
        Parsed RINEX data with uniform structure
        
    Examples
    --------
    >>> from pathlib import Path
    >>> # Works with RINEX2
    >>> data = get_rinex_data_auto(Path('wsrt1690.24o.gz'))
    >>> 
    >>> # Works with RINEX3
    >>> data = get_rinex_data_auto(Path('WSRT00NLD_R_20241690000_01D_30S_MO.crx.gz'))
    >>> 
    >>> # Same output structure for both!
    >>> print(data.times)
    >>> print(data.header.datatypes)
    >>> print(list(data.data.keys()))
    """
    version = detect_rinex_version(fname)
    
    if version == 2:
        print(f"  Detected RINEX2: {fname.name}")
        return get_rinex2_data(fname)
    elif version == 3:
        print(f"  Detected RINEX3: {fname.name}")
        return get_rinex3_data(fname)
    else:
        raise ValueError(f"Unsupported RINEX version: {version}")


# Re-export for convenience
__all__ = [
    'RinexHeader',
    'RinexData', 
    'get_rinex_data_auto',  # Recommended - auto-detects version
    'get_rinex2_data',      # Explicit RINEX2
    'get_rinex3_data',      # Explicit RINEX3 (from original module)
    'detect_rinex_version',
]
