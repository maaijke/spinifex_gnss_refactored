"""
GNSS satellite geometry calculations.

This module handles satellite position interpolation, azimuth/elevation calculations,
and ionospheric pierce point computations.
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation, ITRS, AltAz
from spinifex.geometry import IPP, get_ipp_from_altaz
from pathlib import Path
from scipy.interpolate import CubicSpline
from typing import Literal
from spinifex_gnss.parse_sp3 import parse_sp3, concatenate_sp3_files, SP3Data


def get_sp3_data(sp3_files: list[Path]) -> SP3Data:
    """
    Load and combine SP3 satellite orbit files.
    
    This replaces the old georinex-based get_sat_pos_object function.
    
    Parameters
    ----------
    sp3_files : list[Path]
        List of SP3 files, should contain 3 consecutive days for interpolation
        
    Returns
    -------
    SP3Data
        Combined satellite ephemeris data
        
    Notes
    -----
    Files are automatically decompressed if gzipped. The custom parser handles
    .gz files transparently without needing external gunzip calls.
    
    Examples
    --------
    >>> sp3_files = [Path("day1.SP3.gz"), Path("day2.SP3.gz"), Path("day3.SP3.gz")]
    >>> sp3_data = get_sp3_data(sp3_files)
    >>> print(len(sp3_data.times))
    864  # 3 days * 288 epochs/day
    """
    if len(sp3_files) < 1:
        raise ValueError("At least one SP3 file is required")
    
    # Combine all SP3 files
    return concatenate_sp3_files(sp3_files)


def interpolate_satellite(
    sp3_data: SP3Data,
    satellite_id: str,
    time_target: Time,
    method: Literal["linear", "cubicspline"] = "cubicspline"
) -> u.Quantity:
    """
    Interpolate satellite positions to requested times.
    
    Parameters
    ----------
    sp3_data : SP3Data
        Satellite ephemeris data from SP3 files
    satellite_id : str
        Satellite identifier (e.g., 'G01', 'E05', 'R24')
    time_target : Time
        Times to interpolate to
    method : {"linear", "cubicspline"}, optional
        Interpolation method, by default "cubicspline"
        
    Returns
    -------
    u.Quantity
        Interpolated ITRF positions [n_times, 3] with units of meters
        
    Raises
    ------
    ValueError
        If satellite_id is not in the SP3 data
    ValueError
        If no valid position data exists for the satellite
        
    Examples
    --------
    >>> sp3_data = get_sp3_data(sp3_files)
    >>> target_times = Time(['2024-12-01T12:00:00', '2024-12-01T13:00:00'])
    >>> positions = interpolate_satellite(sp3_data, 'G01', target_times)
    >>> print(positions.shape)
    (2, 3)  # 2 times, xyz coordinates
    """
    if satellite_id not in sp3_data.positions:
        raise ValueError(
            f"Satellite {satellite_id} not found in SP3 data. "
            f"Available satellites: {list(sp3_data.positions.keys())}"
        )
    
    # Get satellite position data
    sat_positions = sp3_data.positions[satellite_id]
    
    # Filter out NaN values for interpolation
    valid_mask = ~np.isnan(sat_positions[:, 0])
    
    if not np.any(valid_mask):
        raise ValueError(f"No valid position data for satellite {satellite_id}")
    
    # Time arrays for interpolation
    x_values = sp3_data.times[valid_mask].mjd
    target_x = time_target.mjd
    
    # Extract valid positions
    valid_positions = sat_positions[valid_mask]
    x, y, z = valid_positions.T
    
    # Interpolate each coordinate
    if method == "linear":
        x_interp = np.interp(target_x, x_values, x)
        y_interp = np.interp(target_x, x_values, y)
        z_interp = np.interp(target_x, x_values, z)
    elif method == "cubicspline":
        spl_x = CubicSpline(x_values, x)
        spl_y = CubicSpline(x_values, y)
        spl_z = CubicSpline(x_values, z)
        
        x_interp = spl_x(target_x)
        y_interp = spl_y(target_x)
        z_interp = spl_z(target_x)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Return as quantity with shape [n_times, 3]
    return np.column_stack([x_interp, y_interp, z_interp]) * u.m


def get_sat_pos(sp3_data: SP3Data, times: Time, sat_name: str) -> EarthLocation:
    """
    Get satellite positions at requested times for a specific satellite.
    
    Parameters
    ----------
    sp3_data : SP3Data
        Ephemeris data from SP3 files
    times : Time
        Times at which to get satellite position
    sat_name : str
        Satellite PRN (e.g., 'G01', 'E05')
        
    Returns
    -------
    EarthLocation
        Satellite location(s) at requested time(s)
        
    Examples
    --------
    >>> sp3_data = get_sp3_data(sp3_files)
    >>> times = Time('2024-12-01T12:00:00')
    >>> sat_pos = get_sat_pos(sp3_data, times, 'G01')
    >>> print(sat_pos.x, sat_pos.y, sat_pos.z)
    """
    positions = interpolate_satellite(sp3_data, sat_name, times)
    return EarthLocation(
        x=positions[:, 0],
        y=positions[:, 1], 
        z=positions[:, 2]
    )


def get_azel_sat(satpos: EarthLocation, gnsspos: EarthLocation, times: Time) -> AltAz:
    """
    Calculate azimuth and elevation of satellite from receiver position.
    
    Parameters
    ----------
    satpos : EarthLocation
        Satellite position(s)
    gnsspos : EarthLocation
        GNSS receiver position
    times : Time
        Observation times
        
    Returns
    -------
    AltAz
        Azimuth and elevation at requested times
        
    Notes
    -----
    Azimuth is measured clockwise from North (0-360°).
    Elevation is measured from horizon (0°) to zenith (90°).
    
    Examples
    --------
    >>> sat_pos = get_sat_pos(sp3_data, times, 'G01')
    >>> receiver_pos = EarthLocation(lat=52*u.deg, lon=5*u.deg, height=0*u.m)
    >>> azel = get_azel_sat(sat_pos, receiver_pos, times)
    >>> print(f"Azimuth: {azel.az.deg:.1f}°, Elevation: {azel.alt.deg:.1f}°")
    """
    # Convert satellite position to ITRS
    itrs_geo = satpos.itrs
    
    # Calculate topocentric (relative to receiver) position
    topo_itrs_repr = itrs_geo.cartesian.without_differentials() - gnsspos.itrs.cartesian
    itrs_topo = ITRS(topo_itrs_repr, obstime=times, location=gnsspos)
    
    # Transform to AltAz frame
    aa = itrs_topo.transform_to(AltAz(obstime=times, location=gnsspos))
    
    return aa


def get_stat_sat_ipp(
    satpos: EarthLocation,
    gnsspos: EarthLocation,
    times: Time,
    height_array: u.Quantity = np.array([450]) * u.km,
) -> IPP:
    """
    Get ionospheric pierce points for satellite-receiver combination.
    
    Parameters
    ----------
    satpos : EarthLocation
        Satellite position(s)
    gnsspos : EarthLocation
        GNSS receiver position
    times : Time
        Observation times (should be in GPS time)
    height_array : u.Quantity, optional
        Altitudes of ionospheric pierce points, by default [450 km]
        
    Returns
    -------
    IPP
        Ionospheric pierce point data structure containing:
        - loc: Pierce point locations at each height
        - times: Observation times
        - los: Line-of-sight vectors
        - airmass: Obliquity factors
        - altaz: Azimuth/elevation data
        - station_loc: Receiver location
        
    Notes
    -----
    The ionospheric pierce point is where the line of sight from the receiver
    to the satellite intersects a spherical shell at the specified height(s).
    
    Examples
    --------
    >>> sat_pos = get_sat_pos(sp3_data, times, 'G01')
    >>> receiver_pos = EarthLocation(lat=52*u.deg, lon=5*u.deg, height=0*u.m)
    >>> ipp = get_stat_sat_ipp(sat_pos, receiver_pos, times, height_array=[450]*u.km)
    >>> print(ipp.loc.lat, ipp.loc.lon, ipp.loc.height)
    """
    # Calculate azimuth and elevation
    azel = get_azel_sat(satpos, gnsspos, times)
    
    # Calculate pierce points at specified heights
    return get_ipp_from_altaz(gnsspos, azel, height_array)


def filter_by_elevation(
    azel: AltAz,
    min_elevation: float = 20.0
) -> np.ndarray:
    """
    Create mask for observations above minimum elevation.
    
    Parameters
    ----------
    azel : AltAz
        Azimuth/elevation data
    min_elevation : float, optional
        Minimum elevation in degrees, by default 20.0
        
    Returns
    -------
    np.ndarray
        Boolean mask, True for elevations >= min_elevation
        
    Examples
    --------
    >>> azel = get_azel_sat(sat_pos, receiver_pos, times)
    >>> mask = filter_by_elevation(azel, min_elevation=15.0)
    >>> high_el_times = times[mask]
    """
    return azel.alt.deg >= min_elevation


def get_slant_distance(
    satpos: EarthLocation,
    gnsspos: EarthLocation
) -> u.Quantity:
    """
    Calculate slant distance from receiver to satellite.
    
    Parameters
    ----------
    satpos : EarthLocation
        Satellite position(s)
    gnsspos : EarthLocation
        GNSS receiver position
        
    Returns
    -------
    u.Quantity
        Slant distance(s) with units
        
    Examples
    --------
    >>> distance = get_slant_distance(sat_pos, receiver_pos)
    >>> print(f"Distance: {distance.to(u.km):.1f}")
    Distance: 20183.5 km
    """
    dx = satpos.x - gnsspos.x
    dy = satpos.y - gnsspos.y
    dz = satpos.z - gnsspos.z
    
    return np.sqrt(dx**2 + dy**2 + dz**2)
