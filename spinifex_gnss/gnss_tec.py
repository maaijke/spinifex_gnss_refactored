"""
GNSS-based electron density calculations.

This module provides the main interface for calculating electron density
from GNSS observations. It has been refactored to:
- Remove DCB dependencies
- Use custom SP3 parser instead of georinex
- Download IONEX data centrally and pass to processing functions
- Improve code organization and documentation
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation
import gc
from pathlib import Path
from typing import Optional

from spinifex.geometry import IPP, R_EARTH_MEAN
from spinifex.times import get_unique_days, get_indexlist_unique_days
from spinifex.ionospheric.tec_data import ElectronDensity, IonexOptions
from spinifex.ionospheric.ionex_manipulation import _download_ionex, read_ionex

from spinifex_gnss.download_gnss import (
    download_rinex,
    download_satpos_files,
    download_dcb_file,
)
from spinifex_gnss.parse_gnss import process_all_rinex_parallel
from spinifex_gnss.parse_dcb import parse_dcb_file, DCBData
from spinifex_gnss.proces_gnss_data import get_ipp_density
from spinifex_gnss.gnss_stations import gnss_pos_dict
from spinifex_gnss.gnss_geometry import get_sp3_data, _convert_ipp_lonlatr_to_xyz
from spinifex_gnss.config import MIN_DISTANCE_SELECT, DEFAULT_IONO_HEIGHT, RinexStrategy


def get_min_distance(ipp: IPP, gnss_pos: EarthLocation) -> u.Quantity:
    """
    Calculate minimum distance from IPP to a GNSS station.

    Parameters
    ----------
    ipp : IPP
        Ionospheric pierce points
    gnss_pos : EarthLocation
        GNSS station position

    Returns
    -------
    u.Quantity
        Minimum distance with units

    Examples
    --------
    >>> ipp = get_ipp_from_skycoord(...)
    >>> station_pos = EarthLocation.from_geodetic(6*u.deg, 52*u.deg, 0*u.m)
    >>> min_dist = get_min_distance(ipp, station_pos)
    >>> print(f"Minimum distance: {min_dist.to(u.km)}")
    """

    # Convert IPP geocentric lon/lat/radius to XYZ

    ipp_xyz = _convert_ipp_lonlatr_to_xyz(ipp)

    distance = np.linalg.norm(
        ipp_xyz.to(u.m).value - gnss_pos.itrs.cartesian.xyz.to(u.m).value, axis=-1
    )

    return np.min(distance) * u.m


def select_gnss_stations(
    ipp_location: IPP,
    gnss_positions: Optional[dict] = None,
    max_distance: u.Quantity = MIN_DISTANCE_SELECT,
    reference_height: u.Quantity = DEFAULT_IONO_HEIGHT,
) -> list[str]:
    """
    Select GNSS stations near the ionospheric pierce points.

    This function finds GNSS stations within a specified distance of the
    target IPPs to use for electron density calculations.

    Parameters
    ----------
    ipp_location : IPP
        IPP locations (times × heights)
    gnss_positions : dict, optional
        Dictionary of GNSS station positions. If None, uses global gnss_pos_dict
    max_distance : u.Quantity, optional
        Maximum distance for station selection, by default MIN_DISTANCE_SELECT
    reference_height : u.Quantity, optional
        Reference ionospheric height for selection, by default DEFAULT_IONO_HEIGHT

    Returns
    -------
    list[str]
        List of selected GNSS station names

    Notes
    -----
    The function projects IPPs to Earth's surface at the reference height
    to calculate horizontal distances to stations.

    Examples
    --------
    >>> ipp = get_ipp_from_skycoord(source, loc, times, heights)
    >>> stations = select_gnss_stations(ipp.loc)
    >>> print(f"Selected {len(stations)} stations: {stations[:5]}")
    """
    if gnss_positions is None:
        gnss_positions = gnss_pos_dict

    # Find height index closest to reference height
    hidx = np.argmin(np.abs(ipp_location.height[0] - R_EARTH_MEAN - reference_height))

    # Project IPPs to Earth's surface at reference height
    ipp_earth = IPP(
        lon=ipp_location.lon[:, hidx],
        lat=ipp_location.lat[:, hidx],
        height=np.ones_like(ipp_location.lon[:, hidx].value) * R_EARTH_MEAN,
        times=None,  # Not needed for distance calculation
        los=None,
        airmass=None,
        altaz=None,
        station_loc=None,
    )
    # Select stations within maximum distance
    gnss_list = []
    for gnss_name, gnss_pos in gnss_positions.items():
        if (
            get_min_distance(
                ipp_earth,
                gnss_pos,
            )
            < max_distance
        ):
            gnss_list.append(gnss_name)

    return gnss_list


def _select_times_from_ipp(ipp: IPP, indices: np.ndarray) -> IPP:
    """
    Select a subset of IPP data by time indices.

    Parameters
    ----------
    ipp : IPP
        Full IPP data structure
    indices : np.ndarray
        Indices of times to select

    Returns
    -------
    IPP
        IPP data for selected times only
    """
    return IPP(
        lon=ipp.lon[indices],
        lat=ipp.lat[indices],
        height=ipp.height[indices],
        times=ipp.times[indices],
        los=ipp.los[indices],
        airmass=ipp.airmass[indices],
        altaz=ipp.altaz[indices],
        station_loc=ipp.station_loc,
    )


def get_electron_density_gnss(
    ipp: IPP,
    data_directory: Optional[Path] = None,
    max_workers: int = 20,
    n_time_slots: int = 5,
    max_time_diff_min: float = 2.5,
    use_time_weighting: bool = True,
    strategy: RinexStrategy = RinexStrategy.DCB_WITH_GIM_FALLBACK,
) -> ElectronDensity:
    """
    Calculate electron density from GNSS observations.

    This is the main function for obtaining electron density measurements
    at specified ionospheric pierce points using nearby GNSS stations.

    The workflow:
    1. Group target times by day
    2. For each day:
       a. Select nearby GNSS stations
       b. Download RINEX observation files (no DCB needed!)
       c. Download SP3 satellite orbit files
       d. Download IONEX files for GIM bias correction
       e. Parse RINEX data
       f. Calculate electron density (with IONEX passed as parameter)
    3. Combine results from all days

    Parameters
    ----------
    ipp : IPP
        Target ionospheric pierce points where electron density is needed
    data_directory : Path, optional
        Directory for downloaded data files. If None, uses default.
    max_workers : int, optional
        Maximum parallel workers for RINEX processing, by default 20

    Returns
    -------
    ElectronDensity
        Electron density and uncertainties at the target IPPs

    Notes
    -----
    This refactored version:
    - Does NOT use DCB corrections (removed dependency)
    - Uses custom SP3 parser (no georinex)
    - Uses parallel processing for efficiency
    - Downloads IONEX data centrally and passes to processing functions

    Examples
    --------
    >>> from spinifex.geometry import get_ipp_from_skycoord
    >>> from astropy.coordinates import SkyCoord
    >>>
    >>> # Define observation scenario
    >>> source = SkyCoord.from_name("CasA")
    >>> station = EarthLocation.from_geodetic(6*u.deg, 52*u.deg, 0*u.m)
    >>> times = Time('2024-06-18T12:00:00') + np.arange(48) * 5*u.min
    >>> heights = np.arange(100, 1500, 30) * u.km
    >>>
    >>> # Get IPPs
    >>> ipp = get_ipp_from_skycoord(source, station, times, heights)
    >>>
    >>> # Calculate electron density
    >>> density = get_electron_density_gnss(ipp)
    >>> print(f"Density shape: {density.electron_density.shape}")
    """
    # Group times by unique days
    unique_days = get_unique_days(ipp.times)
    unique_days_indices = get_indexlist_unique_days(unique_days, ipp.times)

    all_data = []

    # Process each day separately
    for day, indices in zip(unique_days, unique_days_indices):
        print(f"Processing day: {day.iso}")

        # Select IPPs for this day
        selected_ipp = _select_times_from_ipp(ipp, indices)
        dcb_data = None    
        if strategy in (RinexStrategy.DCB_ONLY, RinexStrategy.DCB_WITH_GIM_FALLBACK):
            # Download DCB files for bias correction
            print(f"  Downloading DCB files...")
            dcb_file = download_dcb_file(date=day.to_datetime(), datapath=data_directory)
            # parse DCB files
            dcb_data = parse_dcb_file(dcb_file)

        # Find nearby GNSS stations
        gnss_list = select_gnss_stations(selected_ipp)
        print(f"  Selected {len(gnss_list)} GNSS stations")

        if len(gnss_list) == 0:
            print(f"  Warning: No GNSS stations found near IPPs for {day.iso}")
            # Create empty density for this day
            n_times = len(indices)
            n_heights = ipp.lon.shape[1]
            all_data.append(
                ElectronDensity(
                    electron_density=np.zeros((n_times, n_heights)),
                    electron_density_error=np.zeros((n_times, n_heights)),
                )
            )
            continue

        # Download RINEX files for this day and next day
        # (Need consecutive days for 24-hour coverage)
        print(f"  Downloading RINEX files...")
        gnss_file_list = sorted(
            download_rinex(
                date=day.to_datetime(), stations=gnss_list, datapath=data_directory
            )
        )

        # Get station names from downloaded files
        st_list = sorted([f.name[:9] for f in gnss_file_list])
        print(f"  Downloaded {len(st_list)} RINEX files for day 1")

        # Download next day's files for same stations
        gnss_file_list_next_day = sorted(
            download_rinex(
                date=(day + 1 * u.day).to_datetime(),
                stations=st_list,
                datapath=data_directory,
            )
        )

        st_list2 = sorted([f.name[:4] for f in gnss_file_list_next_day])
        print(f"  Downloaded {len(st_list2)} RINEX files for day 2")

        # Only keep stations with data for both days #truncate to first four for rinex2 compatibility
        gnss_file_list = [f for f in gnss_file_list if f.name[:4] in st_list2]

        # Pair up files from consecutive days
        gnss_file_pairs = list(zip(gnss_file_list, gnss_file_list_next_day))

        # Parse RINEX files (no DCB needed!)
        print(f"  Parsing RINEX files...")
        gnss_data_list = process_all_rinex_parallel(
            gnss_file_pairs, max_workers=max_workers
        )
        # Filter for valid data only
        gnss_data_list = [data for data in gnss_data_list if data.is_valid]
        print(f"  Valid GNSS data from {len(gnss_data_list)} stations")

        # Download SP3 satellite orbit files
        print(f"  Downloading SP3 files...")
        sp3_files = download_satpos_files(
            date=day.to_datetime(), datapath=data_directory
        )

        # Parse SP3 files (using custom parser - no georinex!)
        print(f"  Parsing SP3 files...")
        sp3_data = get_sp3_data(sp3_files[:3])  # Use 3 days for interpolation

        # Download IONEX only when GIM alignment is needed
        ionex = None
        if strategy in (RinexStrategy.GIM_ONLY, RinexStrategy.DCB_WITH_GIM_FALLBACK):
            print(f"  Downloading IONEX files...")
            default_options = IonexOptions(remove_midnight_jumps=True)
            sorted_ionex_paths = _download_ionex(
                times=selected_ipp.times, options=default_options
            )
            sorted_next_day_paths = (
                _download_ionex(
                    times=selected_ipp.times + 1 * u.day, options=default_options
                )
                if default_options.remove_midnight_jumps
                else [None] * len(sorted_ionex_paths)
            )
            print(f"  Reading IONEX files...")
            ionex = read_ionex(
                sorted_ionex_paths[0],
                sorted_next_day_paths[0],
                options=default_options,
                concatenate=True,
            )


        # Calculate electron density for this day
        print(f"  Calculating electron density...")
        day_density = get_ipp_density(
            gnss_data_list=gnss_data_list,
            ipp_target=selected_ipp,
            sp3_data=sp3_data,
            ionex=ionex,
            dcb_data=dcb_data,
            n_time_slots=n_time_slots,
            max_time_diff_min=max_time_diff_min,
            use_time_weighting=use_time_weighting,
            max_workers=max_workers,
            strategy=strategy,
        )

        all_data.append(day_density)
        print(f"  ✓ Day complete\n")

        del gnss_data_list, sp3_data
        gc.collect()  # Force garbage collection
    # Combine results from all days
    print(f"Combining results from {len(all_data)} days...")
    combined_density = ElectronDensity(
        electron_density=np.concatenate(
            [data.electron_density for data in all_data], axis=0
        ),
        electron_density_error=np.concatenate(
            [data.electron_density_error for data in all_data], axis=0
        ),
    )

    print(f"✓ Complete! Final shape: {combined_density.electron_density.shape}")
    return combined_density


# Backward compatibility alias
get_electron_density = get_electron_density_gnss
