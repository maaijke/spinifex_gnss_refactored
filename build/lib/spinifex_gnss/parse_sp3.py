"""
Custom SP3 file parser for satellite orbit data.

This module replaces the georinex dependency with a pure Python implementation
for parsing SP3 (Standard Product 3) orbit files.

SP3 format specification: https://files.igs.org/pub/data/format/sp3c.txt
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
    """SP3 version (a, c, d)"""
    pos_vel_flag: str
    """Position and/or velocity flag (P=positions only, V=velocities, PV=both)"""
    start_time: Time
    """Data start time"""
    num_epochs: int
    """Number of epochs in file"""
    data_used: str
    """Data used indicator"""
    coordinate_system: str
    """Coordinate system (e.g., IGS14, ITRF2014)"""
    orbit_type: str
    """Orbit type (e.g., FIT, BCT, HLM)"""
    agency: str
    """Agency code"""
    gps_week: int
    """GPS week number"""
    seconds_of_week: float
    """Seconds of GPS week at start"""
    epoch_interval: float
    """Epoch interval in seconds"""
    mjd_start: int
    """Modified Julian Day at start"""
    fractional_day: float
    """Fractional part of day"""
    satellite_ids: list[str]
    """List of satellite IDs in file"""


class SP3Data(NamedTuple):
    """Parsed SP3 satellite orbit data."""
    
    header: SP3Header
    """File header information"""
    times: Time
    """Array of epoch times"""
    positions: dict[str, np.ndarray]
    """Dictionary mapping satellite ID to position array [n_epochs, 3] in meters"""
    clock_corrections: dict[str, np.ndarray]
    """Dictionary mapping satellite ID to clock correction array [n_epochs] in seconds"""
    position_stds: dict[str, np.ndarray] | None
    """Optional position standard deviations [n_epochs, 3] in meters"""
    clock_stds: dict[str, np.ndarray] | None
    """Optional clock standard deviations [n_epochs] in seconds"""


def _parse_sp3_header(lines: list[str]) -> tuple[SP3Header, int]:
    """
    Parse SP3 file header.
    
    Parameters
    ----------
    lines : list[str]
        Lines from SP3 file
        
    Returns
    -------
    tuple[SP3Header, int]
        Parsed header and line number where data starts
    """
    # First line: #cP2024 12  1  0  0  0.00000000      96 ORBIT IGS14 HLM  IGS
    first_line = lines[0]
    version = first_line[1]
    pos_vel_flag = first_line[2]
    
    year = int(first_line[3:8])
    month = int(first_line[9:11])
    day = int(first_line[12:14])
    hour = int(first_line[15:17])
    minute = int(first_line[18:20])
    second = float(first_line[21:32])
    
    start_time = Time(datetime(year, month, day, hour, minute, int(second)))
    num_epochs = int(first_line[32:40])
    data_used = first_line[40:46].strip()
    coordinate_system = first_line[46:52].strip()
    orbit_type = first_line[52:56].strip()
    agency = first_line[56:60].strip()
    
    # Second line: ## 2333 604800.00000000   900.00000000 59917 0.0000000000000
    second_line = lines[1]
    gps_week = int(second_line[3:8])
    seconds_of_week = float(second_line[9:23])
    epoch_interval = float(second_line[24:38])
    mjd_start = int(second_line[39:44])
    fractional_day = float(second_line[45:60])
    
    # Satellite IDs: lines starting with '+'
    satellite_ids = []
    for line in lines[2:]:
        if line.startswith('+'):
            # Extract satellite IDs (3 characters each, starting at position 9)
            sats = [line[i:i+3].strip() for i in range(9, len(line), 3) if line[i:i+3].strip()]
            satellite_ids.extend(sats)
        elif line.startswith('++'):
            # Accuracy exponents - skip for now
            continue
        elif line.startswith('%c') or line.startswith('%f') or line.startswith('%i'):
            # File type, time system, etc. - skip for now
            continue
        elif line.startswith('/*'):
            # Comment line - skip
            continue
        elif line.startswith('*'):
            # Start of epoch data
            data_start_line = lines.index(line)
            break
    
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


def _parse_sp3_epoch(line: str) -> Time:
    """
    Parse epoch time from SP3 epoch header line.
    
    Format: *  2024 12  1  0  0  0.00000000
    
    Parameters
    ----------
    line : str
        Epoch header line
        
    Returns
    -------
    Time
        Parsed epoch time
    """
    year = int(line[3:7])
    month = int(line[7:11])
    day = int(line[11:13])
    hour = int(line[13:16])
    minute = int(line[16:19])
    second = float(line[19:32])
    
    return Time(datetime(year, month, day, hour, minute, int(second)) + 
                timedelta(seconds=second - int(second)))


def _parse_sp3_position_line(line: str) -> tuple[str, np.ndarray, float]:
    """
    Parse satellite position line from SP3 file.
    
    Format: PG01  20000000.000  15000000.000  10000000.000    999999.999999
    
    Parameters
    ----------
    line : str
        Position line
        
    Returns
    -------
    tuple[str, np.ndarray, float]
        Satellite ID, position [x, y, z] in km, clock correction in microseconds
    """
    sat_id = line[1:4].strip()
    x = float(line[4:18])  # km
    y = float(line[18:32])  # km
    z = float(line[32:46])  # km
    clock = float(line[46:60])  # microseconds
    
    # Convert to meters
    position = np.array([x, y, z]) * 1000.0  # km to meters
    
    # Convert clock to seconds (999999.999999 indicates no clock data)
    if abs(clock - 999999.999999) < 1e-6:
        clock = np.nan
    else:
        clock = clock * 1e-6  # microseconds to seconds
    
    return sat_id, position, clock


def parse_sp3(filepath: Path, include_stds: bool = False) -> SP3Data:
    """
    Parse an SP3 orbit file.
    
    Parameters
    ----------
    filepath : Path
        Path to SP3 file (can be gzipped)
    include_stds : bool, optional
        Whether to parse position/clock standard deviations, by default False
        
    Returns
    -------
    SP3Data
        Parsed satellite orbit data
        
    Examples
    --------
    >>> sp3_data = parse_sp3(Path("GBM0MGXRAP_20243360000_01D_05M_ORB.SP3.gz"))
    >>> print(sp3_data.header.satellite_ids)
    ['G01', 'G02', 'G03', ...]
    >>> print(sp3_data.positions['G01'].shape)
    (288, 3)  # 288 epochs, xyz coordinates
    """
    # Read file (handle gzip)
    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    # Parse header
    header, data_start = _parse_sp3_header(lines)
    
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
        if line.startswith('EOF'):
            break
            
        if line.startswith('*'):
            # New epoch
            if current_epoch is not None:
                # Fill in NaN for satellites with no data at this epoch
                for sat_id in header.satellite_ids:
                    if not epoch_data_collected[sat_id]:
                        positions[sat_id].append(np.array([np.nan, np.nan, np.nan]))
                        clock_corrections[sat_id].append(np.nan)
                        if include_stds:
                            position_stds[sat_id].append(np.array([np.nan, np.nan, np.nan]))
                            clock_stds[sat_id].append(np.nan)
            
            current_epoch = _parse_sp3_epoch(line)
            times.append(current_epoch)
            epoch_data_collected = {sat_id: False for sat_id in header.satellite_ids}
            
        elif line.startswith('P') or line.startswith('V'):
            # Position (or velocity) line
            if line.startswith('P'):
                sat_id, position, clock = _parse_sp3_position_line(line)
                
                if sat_id in positions:
                    positions[sat_id].append(position)
                    clock_corrections[sat_id].append(clock)
                    epoch_data_collected[sat_id] = True
                    
                    if include_stds:
                        # Standard deviations would be on following lines
                        # Not implemented yet - set to NaN
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
