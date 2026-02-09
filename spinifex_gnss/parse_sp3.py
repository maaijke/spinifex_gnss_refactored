"""
Enhanced SP3 parser with support for legacy formats.

Handles SP3 format variations across different eras:
- SP3-a (1991): Original format
- SP3-b (1999): Extended format  
- SP3-c (2001): Multi-GNSS format
- SP3-d (2016): Latest format

Key differences in older formats:
- Different header spacing
- Varying date/time field positions
- Different coordinate system labels
"""

import gzip
from pathlib import Path
import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation
from typing import NamedTuple
from datetime import datetime, timedelta


class SP3Header(NamedTuple):
    """SP3 file header information."""
    
    version: str
    pos_vel_flag: str
    start_time: Time
    num_epochs: int
    data_used: str
    coordinate_system: str
    orbit_type: str
    agency: str
    gps_week: int
    seconds_of_week: float
    epoch_interval: float
    mjd_start: int
    fractional_day: float
    satellite_ids: list[str]


class SP3Data(NamedTuple):
    """Parsed SP3 satellite orbit data."""
    
    header: SP3Header
    times: Time
    positions: dict[str, np.ndarray]
    clock_corrections: dict[str, np.ndarray]
    position_stds: dict[str, np.ndarray] | None
    clock_stds: dict[str, np.ndarray] | None


def _parse_sp3_header_flexible(lines: list[str]) -> tuple[SP3Header, int]:
    """
    Parse SP3 file header with flexible spacing for legacy formats.
    
    Handles differences between SP3-a/b/c/d versions.
    
    Parameters
    ----------
    lines : list[str]
        Lines from SP3 file
        
    Returns
    -------
    tuple[SP3Header, int]
        Parsed header and line number where data starts
        
    Notes
    -----
    Old format (SP3-c, 2016):
        #cP2016  6 16  0  0  0.00000000     288   u+U UNDEF FIT  GFZ
        
    New format (SP3-d, 2024):
        #dP2024  6 16  0  0  0.00000000     288   u+U IGS20 FIT  GFZ
        
    Differences:
    - Spacing in date fields varies
    - Coordinate system may be "UNDEF" vs "IGS20"
    - Version character: c vs d
    """
    first_line = lines[0]
    
    # Version (character 1)
    version = first_line[1]
    
    # Position/velocity flag (character 2)
    pos_vel_flag = first_line[2]
    
    # Parse date/time - use flexible parsing to handle spacing variations
    # Split by whitespace and take first 6 numbers after version
    parts = first_line[3:].split()
    
    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        hour = int(parts[3])
        minute = int(parts[4])
        second = float(parts[5])
    except (IndexError, ValueError) as e:
        # Fallback to fixed positions if split fails
        try:
            year = int(first_line[3:8].strip())
            month = int(first_line[9:11].strip())
            day = int(first_line[12:14].strip())
            hour = int(first_line[15:17].strip())
            minute = int(first_line[18:20].strip())
            second = float(first_line[21:32].strip())
        except ValueError as e2:
            raise ValueError(f"Could not parse date/time from first line: {first_line}") from e2
    
    start_time = Time(datetime(year, month, day, hour, minute, int(second)) +
                     timedelta(seconds=second - int(second)))
    
    # Number of epochs - usually around position 32-40
    try:
        num_epochs = int(first_line[32:40].strip())
    except ValueError:
        # Try finding it in the parts
        for i, part in enumerate(parts):
            if i > 5 and part.isdigit():
                num_epochs = int(part)
                break
        else:
            num_epochs = 0
    
    # Data used, coordinate system, orbit type, agency
    # These have flexible positions, extract from remaining parts
    remaining = first_line[40:].strip()
    remaining_parts = remaining.split()
    
    if len(remaining_parts) >= 4:
        data_used = remaining_parts[0]
        coordinate_system = remaining_parts[1]
        orbit_type = remaining_parts[2]
        agency = remaining_parts[3]
    else:
        # Fallback
        data_used = first_line[40:46].strip()
        coordinate_system = first_line[46:52].strip()
        orbit_type = first_line[52:56].strip()
        agency = first_line[56:60].strip()
    
    # Second line: GPS week, seconds, interval, MJD
    second_line = lines[1]
    second_parts = second_line.split()
    
    try:
        gps_week = int(second_parts[1])
        seconds_of_week = float(second_parts[2])
        epoch_interval = float(second_parts[3])
        mjd_start = int(second_parts[4])
        fractional_day = float(second_parts[5])
    except (IndexError, ValueError):
        # Fallback to fixed positions
        gps_week = int(second_line[3:8].strip())
        seconds_of_week = float(second_line[9:23].strip())
        epoch_interval = float(second_line[24:38].strip())
        mjd_start = int(second_line[39:44].strip())
        fractional_day = float(second_line[45:60].strip())
    
    # Satellite IDs: lines starting with '+'
    satellite_ids = []
    data_start_line = None
    
    for i, line in enumerate(lines[2:], start=2):
        if line.startswith('+'):
            # Extract satellite IDs
            # Format: "+   78   C01C03C04..." or "+        E14E18E19..."
            # Satellites are 3 characters each, can start at various positions
            sat_line = line[9:].strip()  # Skip "+ nnn" part
            
            # Split into 3-character chunks
            j = 0
            while j < len(sat_line):
                sat_id = sat_line[j:j+3].strip()
                if sat_id and sat_id != '00':  # Skip padding
                    satellite_ids.append(sat_id)
                j += 3
                
        elif line.startswith('++'):
            # Accuracy exponents - skip
            continue
        elif line.startswith('%'):
            # File metadata - skip
            continue
        elif line.startswith('/*'):
            # Comment - skip
            continue
        elif line.startswith('*'):
            # Start of epoch data
            data_start_line = i
            break
    
    if data_start_line is None:
        raise ValueError("Could not find start of epoch data (line starting with '*')")
    
    header = SP3Header(
        version=version,
        pos_vel_flag=pos_vel_flag,
        start_time=start_time,
        num_epochs=num_epochs,
        data_used=data_used,
        coordinate_system=coordinate_system,
        orbit_type=orbit_type,
        agency=agency,
        gps_week=gps_week,
        seconds_of_week=seconds_of_week,
        epoch_interval=epoch_interval,
        mjd_start=mjd_start,
        fractional_day=fractional_day,
        satellite_ids=satellite_ids,
    )
    
    return header, data_start_line


def _parse_sp3_epoch_flexible(line: str) -> Time:
    """
    Parse epoch time with flexible spacing.
    
    Format: *  2016  6 16  0  0  0.00000000
            *  2024  6 16  0  0  0.00000000
    
    Parameters
    ----------
    line : str
        Epoch header line
        
    Returns
    -------
    Time
        Parsed epoch time
    """
    # Split by whitespace for flexibility
    parts = line.split()
    
    try:
        year = int(parts[1])
        month = int(parts[2])
        day = int(parts[3])
        hour = int(parts[4])
        minute = int(parts[5])
        second = float(parts[6])
    except (IndexError, ValueError):
        # Fallback to fixed positions
        year = int(line[3:7].strip())
        month = int(line[7:11].strip())
        day = int(line[11:13].strip())
        hour = int(line[13:16].strip())
        minute = int(line[16:19].strip())
        second = float(line[19:32].strip())
    
    return Time(datetime(year, month, day, hour, minute, int(second)) + 
                timedelta(seconds=second - int(second)))


def _parse_sp3_position_line(line: str) -> tuple[str, np.ndarray, float]:
    """
    Parse satellite position line.
    
    Format: PG01  22440.953477  14400.571300  -1297.025565     23.768370
    
    Works for both old and new formats (same structure).
    """
    sat_id = line[1:4].strip()
    
    try:
        x = float(line[4:18].strip())
        y = float(line[18:32].strip())
        z = float(line[32:46].strip())
        clock = float(line[46:60].strip())
    except ValueError:
        # Try flexible parsing
        parts = line[4:].split()
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        clock = float(parts[3]) if len(parts) > 3 else 999999.999999
    
    # Convert to meters
    position = np.array([x, y, z]) * 1000.0
    
    # Convert clock (999999.999999 = no data)
    if abs(clock - 999999.999999) < 1e-6:
        clock = np.nan
    else:
        clock = clock * 1e-6  # microseconds to seconds
    
    return sat_id, position, clock


def parse_sp3(filepath: Path, include_stds: bool = False) -> SP3Data:
    """
    Parse an SP3 orbit file (supports all versions: a, b, c, d).
    
    Automatically handles format variations across different eras.
    
    Parameters
    ----------
    filepath : Path
        Path to SP3 file (can be gzipped or .Z compressed)
    include_stds : bool, optional
        Whether to parse position/clock standard deviations
        
    Returns
    -------
    SP3Data
        Parsed satellite orbit data
        
    Examples
    --------
    >>> # Modern file
    >>> sp3_data = parse_sp3(Path("GBM0MGXRAP_20243360000_01D_05M_ORB.SP3.gz"))
    >>> 
    >>> # Legacy file
    >>> sp3_data = parse_sp3(Path("gbm19014.sp3.Z"))
    >>> 
    >>> # Both work the same way!
    >>> print(sp3_data.header.satellite_ids)
    >>> print(sp3_data.positions['G01'].shape)
    """
    # Read file (handle gzip and .Z compression)
    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    elif filepath.suffix == '.Z':
        # Unix compress format - decompress first
        import subprocess
        result = subprocess.run(['uncompress', '-c', str(filepath)], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Failed to decompress .Z file: {result.stderr}")
        lines = result.stdout.split('\n')
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    # Parse header with flexible spacing
    header, data_start = _parse_sp3_header_flexible(lines)
    
    # Initialize data structures
    times = []
    positions = {sat_id: [] for sat_id in header.satellite_ids}
    clock_corrections = {sat_id: [] for sat_id in header.satellite_ids}
    
    if include_stds:
        position_stds = {sat_id: [] for sat_id in header.satellite_ids}
        clock_stds = {sat_id: [] for sat_id in header.satellite_ids}
    else:
        position_stds = None
        clock_stds = None
    
    # Parse data
    current_epoch = None
    epoch_data_collected = {sat_id: False for sat_id in header.satellite_ids}
    
    for line in lines[data_start:]:
        if not line.strip() or line.startswith('EOF'):
            break
            
        if line.startswith('*'):
            # New epoch
            if current_epoch is not None:
                # Fill in NaN for satellites with no data
                for sat_id in header.satellite_ids:
                    if not epoch_data_collected[sat_id]:
                        positions[sat_id].append(np.array([np.nan, np.nan, np.nan]))
                        clock_corrections[sat_id].append(np.nan)
                        if include_stds:
                            position_stds[sat_id].append(np.array([np.nan, np.nan, np.nan]))
                            clock_stds[sat_id].append(np.nan)
            
            current_epoch = _parse_sp3_epoch_flexible(line)
            times.append(current_epoch)
            epoch_data_collected = {sat_id: False for sat_id in header.satellite_ids}
            
        elif line.startswith('P') or line.startswith('V'):
            if line.startswith('P'):
                sat_id, position, clock = _parse_sp3_position_line(line)
                
                if sat_id in positions:
                    positions[sat_id].append(position)
                    clock_corrections[sat_id].append(clock)
                    epoch_data_collected[sat_id] = True
                    
                    if include_stds:
                        position_stds[sat_id].append(np.array([np.nan, np.nan, np.nan]))
                        clock_stds[sat_id].append(np.nan)
    
    # Handle last epoch
    if current_epoch is not None:
        for sat_id in header.satellite_ids:
            if not epoch_data_collected[sat_id]:
                positions[sat_id].append(np.array([np.nan, np.nan, np.nan]))
                clock_corrections[sat_id].append(np.nan)
                if include_stds:
                    position_stds[sat_id].append(np.array([np.nan, np.nan, np.nan]))
                    clock_stds[sat_id].append(np.nan)
    
    # Convert to numpy arrays
    times = Time([t.datetime for t in times])
    for sat_id in header.satellite_ids:
        positions[sat_id] = np.array(positions[sat_id])
        clock_corrections[sat_id] = np.array(clock_corrections[sat_id])
        if include_stds:
            position_stds[sat_id] = np.array(position_stds[sat_id])
            clock_stds[sat_id] = np.array(clock_stds[sat_id])
    
    return SP3Data(
        header=header,
        times=times,
        positions=positions,
        clock_corrections=clock_corrections,
        position_stds=position_stds,
        clock_stds=clock_stds,
    )


def get_satellite_position(
    sp3_data: SP3Data, 
    satellite_id: str, 
    times: Time
) -> EarthLocation:
    """
    Get interpolated satellite position at specific times.
    
    This is a convenience wrapper that combines SP3 parsing with interpolation.
    
    Parameters
    ----------
    sp3_data : SP3Data
        Parsed SP3 data
    satellite_id : str
        Satellite identifier (e.g., 'G01', 'E05')
    times : Time
        Times at which to get satellite position
        
    Returns
    -------
    EarthLocation
        Satellite positions at requested times
        
    Examples
    --------
    >>> sp3_data = parse_sp3(sp3_file)
    >>> times = Time(['2024-12-01T00:00:00', '2024-12-01T01:00:00'])
    >>> positions = get_satellite_position(sp3_data, 'G01', times)
    """
    from scipy.interpolate import CubicSpline
    
    if satellite_id not in sp3_data.positions:
        raise ValueError(f"Satellite {satellite_id} not found in SP3 data")
    
    # Get satellite data
    sat_positions = sp3_data.positions[satellite_id]
    
    # SP3 times in MJD
    sp3_times_mjd = sp3_data.times.mjd
    target_times_mjd = times.mjd
    
    # Remove NaN values for interpolation
    valid_mask = ~np.isnan(sat_positions[:, 0])
    if not np.any(valid_mask):
        raise ValueError(f"No valid position data for satellite {satellite_id}")
    
    valid_times = sp3_times_mjd[valid_mask]
    valid_positions = sat_positions[valid_mask]
    
    # Interpolate each coordinate
    x_interp = CubicSpline(valid_times, valid_positions[:, 0])(target_times_mjd)
    y_interp = CubicSpline(valid_times, valid_positions[:, 1])(target_times_mjd)
    z_interp = CubicSpline(valid_times, valid_positions[:, 2])(target_times_mjd)
    
    # Return as EarthLocation
    return EarthLocation(
        x=x_interp * u.m,
        y=y_interp * u.m,
        z=z_interp * u.m
    )


def concatenate_sp3_files(sp3_files: list[Path]) -> SP3Data:
    """
    Concatenate multiple SP3 files in chronological order.
    
    This is useful for getting continuous coverage across multiple days.
    
    Parameters
    ----------
    sp3_files : list[Path]
        List of SP3 files to concatenate (should be consecutive days)
        
    Returns
    -------
    SP3Data
        Combined SP3 data from all files
        
    Examples
    --------
    >>> files = [Path("day1.SP3"), Path("day2.SP3"), Path("day3.SP3")]
    >>> combined = concatenate_sp3_files(files)
    """
    if not sp3_files:
        raise ValueError("No SP3 files provided")
    
    # Parse all files
    parsed_files = [parse_sp3(f) for f in sp3_files]
    
    # Use first file's header as base
    header = parsed_files[0].header
    
    # Concatenate times
    all_times = Time(np.concatenate([data.times.mjd for data in parsed_files]), format='mjd')
    
    # Get union of all satellite IDs
    all_sat_ids = set()
    for data in parsed_files:
        all_sat_ids.update(data.positions.keys())
    
    # Concatenate positions and clocks for each satellite
    combined_positions = {}
    combined_clocks = {}
    
    for sat_id in all_sat_ids:
        sat_positions = []
        sat_clocks = []
        
        for data in parsed_files:
            if sat_id in data.positions:
                sat_positions.append(data.positions[sat_id])
                sat_clocks.append(data.clock_corrections[sat_id])
            else:
                # Fill with NaN if satellite not in this file
                n_epochs = len(data.times)
                sat_positions.append(np.full((n_epochs, 3), np.nan))
                sat_clocks.append(np.full(n_epochs, np.nan))
        
        combined_positions[sat_id] = np.concatenate(sat_positions, axis=0)
        combined_clocks[sat_id] = np.concatenate(sat_clocks)
    
    return SP3Data(
        header=header,
        times=all_times,
        positions=combined_positions,
        clock_corrections=combined_clocks,
        position_stds=None,
        clock_stds=None,
    )
