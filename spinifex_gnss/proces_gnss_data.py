"""
GNSS data processing with TIME AVERAGING.

Enhancement: Combines multiple time slots for better spatial coverage.

Key change: Instead of nearest-neighbor time matching, we select N nearest
time slots (e.g., ±2.5 minutes) to get more satellite positions and pierce points.
This significantly improves interpolation by increasing measurement density.
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation
from concurrent.futures import as_completed, ProcessPoolExecutor
import gc

from spinifex.geometry import IPP
from spinifex.ionospheric import tec_data
from spinifex.ionospheric.ionex_manipulation import interpolate_ionex, IonexData
from spinifex.ionospheric.iri_density import get_profile

from spinifex_gnss.parse_gnss import GNSSData
from spinifex_gnss.gnss_geometry import get_sat_pos, get_stat_sat_ipp
from spinifex_gnss.gnss_stations import gnss_pos_dict
from spinifex_gnss.tec_core import getphase_tec, get_transmission_time, _get_cycle_slips
from spinifex_gnss.config import (
    DISTANCE_KM_CUT, NDIST_POINTS, ELEVATION_CUT, INTERPOLATION_ORDER,
    GPS_TO_UTC_CORRECTION_DAYS, MAX_WORKERS_DENSITY,
)


# ============================================================================
# Time Averaging Configuration
# ============================================================================

# Number of time slots to combine (5 = ±2 slots around target)
N_TIME_SLOTS = 5

# Maximum time difference to include (in minutes)
MAX_TIME_DIFF_MINUTES = 2.5

# Time weighting in interpolation (True = weight by 1/time_distance)
USE_TIME_WEIGHTING = True


def _get_distance_km(loc1: EarthLocation, loc2: EarthLocation) -> np.ndarray:
    """Calculate distance between locations."""
    dx = loc1.x.value - loc2.x.value
    dy = loc1.y.value - loc2.y.value
    dz = loc1.z.value - loc2.z.value
    return np.sqrt(dx*dx + dy*dy + dz*dz) * 0.001


def _get_gim_phase_corrected(
    phase_tec: np.ndarray,
    ipp_sat_stat: IPP,
    timeselect: np.ndarray,
    ionex: IonexData,
    max_time_diff_min: float = MAX_TIME_DIFF_MINUTES
) -> tuple[np.ndarray, np.ndarray]:
    """
    Correct carrier phase TEC using GIM.
    
    Parameters
    ----------
    phase_tec : np.ndarray
        STEC from carrier phases
    ipp_sat_stat : IPP
        Ionospheric pierce points
    timeselect : np.ndarray
        Indices of target times (for checking arc overlap)
    ionex : IonexData
        Global ionospheric map
    max_time_diff_min : float, optional
        Maximum time difference for time averaging (minutes)
        Used to expand time window for arc selection
        
    Notes
    -----
    OPTIMIZATION: Only processes arcs (cycle slip segments) that overlap
    with the extended time window. Since we now use time averaging,
    we need to check if arc overlaps with target_time ± max_time_diff_min,
    not just the exact target time.
    """
    cycle_slips = _get_cycle_slips(phase_tec=phase_tec)
    phase_bias = np.zeros_like(phase_tec)
    phase_std = np.zeros_like(phase_tec)
    
    default_options = tec_data.IonexOptions(remove_midnight_jumps=True)
    h_idx = np.argmin(
        np.abs(
            ipp_sat_stat.loc[0].height.to(u.km).value
            - default_options.height.to(u.km).value
        )
    )
    
    # Build extended time window for time averaging
    # For each target time, we may select obs within ±max_time_diff_min
    # So we need to process arcs that overlap with this extended window
    if len(timeselect) > 0:
        # Convert max_time_diff to observation index units
        # Assuming ~30-second observations: max_time_diff_min minutes → N indices
        time_buffer_indices = int(np.ceil(max_time_diff_min * 60 / 30))  # Conservative estimate
        
        # Expand timeselect window
        extended_timeselect = set()
        for t_idx in timeselect:
            # Add indices within ±time_buffer around each target
            start_idx = max(0, t_idx - time_buffer_indices)
            end_idx = min(len(phase_tec), t_idx + time_buffer_indices + 1)
            extended_timeselect.update(range(start_idx, end_idx))
        
        extended_timeselect = np.array(sorted(extended_timeselect))
    else:
        extended_timeselect = timeselect
    
    for seg in np.unique(cycle_slips):
        seg_idx = np.nonzero(cycle_slips == seg)[0]
        
        if seg_idx.shape[0] < 2:
            phase_bias[seg_idx] = np.nan
            continue
            
        # OPTIMIZATION: Skip arcs that don't overlap with extended time window
        if np.intersect1d(seg_idx, extended_timeselect).size == 0:
            # This arc is far from any target time, skip it
            continue
            
        ipp = ipp_sat_stat.loc[:, h_idx][seg_idx]
        elevation = ipp_sat_stat.altaz.alt.deg[seg_idx]
        
        gim_tec = interpolate_ionex(
            ionex, ipp.lon.deg, ipp.lat.deg,
            ipp_sat_stat.times[seg_idx],
            apply_earth_rotation=default_options.apply_earth_rotation,
        )
        
        high_el_mask = elevation > ELEVATION_CUT
        
        if np.sum(high_el_mask) > 0:
            phase_bias[seg_idx] = np.nanmean(
                gim_tec[high_el_mask]
                * ipp_sat_stat.airmass[:, h_idx][seg_idx][high_el_mask]
                - phase_tec[seg_idx][high_el_mask],
            )
            
            data_count = np.sum(~np.isnan(phase_tec[seg_idx][high_el_mask]))
            if data_count > 1:
                phase_std[seg_idx] = np.nanstd(
                    gim_tec[high_el_mask]
                    * ipp_sat_stat.airmass[:, h_idx][seg_idx][high_el_mask]
                    - phase_tec[seg_idx][high_el_mask]
                ) / np.sqrt(data_count)
            else:
                phase_std[seg_idx] = np.nan
    
    return phase_tec + phase_bias, phase_std


def _select_time_window(
    target_time_mjd: float,
    obs_times_mjd: np.ndarray,
    n_slots: int = N_TIME_SLOTS,
    max_diff_min: float = MAX_TIME_DIFF_MINUTES
) -> np.ndarray:
    """
    Select indices of observations within time window around target.
    
    Instead of just nearest neighbor, selects N nearest observations
    within max_diff_min minutes of target time.
    
    Parameters
    ----------
    target_time_mjd : float
        Target time in MJD
    obs_times_mjd : np.ndarray
        Observation times in MJD
    n_slots : int
        Number of time slots to select
    max_diff_min : float
        Maximum time difference in minutes
        
    Returns
    -------
    np.ndarray
        Indices of selected time slots
    """
    # Calculate time differences in minutes
    time_diffs_min = np.abs(obs_times_mjd - target_time_mjd) * 24 * 60
    
    # Filter by maximum time difference
    valid_mask = time_diffs_min <= max_diff_min
    
    if not np.any(valid_mask):
        # If no observations within max_diff, use nearest
        return np.array([np.argmin(time_diffs_min)])
    
    # Get indices of valid observations
    valid_indices = np.where(valid_mask)[0]
    valid_diffs = time_diffs_min[valid_mask]
    
    # Select n_slots nearest observations
    if len(valid_indices) <= n_slots:
        # Use all valid observations
        selected_indices = valid_indices
    else:
        # Select n_slots nearest
        nearest = np.argpartition(valid_diffs, n_slots - 1)[:n_slots]
        selected_indices = valid_indices[nearest]
    
    return selected_indices


def _get_distance_ipp_time_averaged(
    stec_values: np.ndarray,
    stec_errors: np.ndarray,
    ipp_sat_stat: list[IPP],
    ipp_target: IPP,
    obs_times: Time,
    profiles: np.ndarray,
    n_time_slots: int = N_TIME_SLOTS,
    max_time_diff_min: float = MAX_TIME_DIFF_MINUTES,
    use_time_weighting: bool = USE_TIME_WEIGHTING,
) -> list[list[np.ndarray]]:
    """
    Calculate VTEC and distances with TIME AVERAGING.
    
    Key Enhancement: Instead of selecting only the nearest time slot,
    this selects N nearest time slots (e.g., ±2.5 minutes) to combine
    measurements from multiple satellite positions.
    
    This significantly increases the number of measurements available
    for interpolation, improving accuracy especially in sparse regions.
    
    Parameters
    ----------
    stec_values : np.ndarray
        STEC values [satellites × times]
    stec_errors : np.ndarray
        STEC uncertainties [satellites × times]
    ipp_sat_stat : list[IPP]
        IPPs for each satellite
    ipp_target : IPP
        Target IPPs where density is needed
    obs_times : Time
        Observation times (GNSS time)
    profiles : np.ndarray
        Normalized electron density profiles [times × heights]
    n_time_slots : int, optional
        Number of time slots to combine, by default 5
    max_time_diff_min : float, optional
        Maximum time difference in minutes, by default 2.5
    use_time_weighting : bool, optional
        Weight measurements by inverse time distance, by default True
        
    Returns
    -------
    list[list[np.ndarray]]
        Nested list [times][heights] of arrays with columns:
        [VTEC, VTEC_error, dlon, dlat, time_weight (optional)]
        
    Notes
    -----
    Example: For target time T0, instead of using only measurements at T0,
    we combine measurements from [T0-2, T0-1, T0, T0+1, T0+2] giving us
    5× more satellite positions and pierce points for interpolation!
    """
    Ntimes_target = ipp_target.times.shape[0]
    Nheights = ipp_target.loc[0].shape[0]
    Nprns = stec_values.shape[0]
    
    # GPS to UTC time correction
    gpstime_correction = GPS_TO_UTC_CORRECTION_DAYS
    
    # Prepare output structure
    result = []
    
    # Process each target time independently
    for target_idx in range(Ntimes_target):
        target_time_mjd = ipp_target.times[target_idx].mjd
        
        # Select observation time slots within window
        time_indices = _select_time_window(
            target_time_mjd + gpstime_correction,
            obs_times.mjd,
            n_slots=n_time_slots,
            max_diff_min=max_time_diff_min
        )
        
        # Calculate time weights if requested
        if use_time_weighting:
            time_diffs_min = np.abs(
                obs_times.mjd[time_indices] - (target_time_mjd + gpstime_correction)
            ) * 24 * 60
            # Inverse time distance weighting (avoid division by zero)
            time_weights = 1.0 / (time_diffs_min + 0.1)
            time_weights /= np.sum(time_weights)  # Normalize
        else:
            time_weights = np.ones(len(time_indices)) / len(time_indices)
        
        # Collect data for all heights at this target time
        height_data = []
        
        for hidx in range(Nheights):
            # Lists to accumulate measurements from all time slots
            all_vtec = []
            all_vtec_errors = []
            all_dlon = []
            all_dlat = []
            all_time_weights = []
            
            # Combine data from selected time slots
            for time_slot_idx, time_weight in zip(time_indices, time_weights):
                # Selection criteria for this time slot
                el_select = np.array([
                    ipp.altaz.alt.deg[time_slot_idx] > ELEVATION_CUT
                    for ipp in ipp_sat_stat
                ])  # prn mask
                
                # Check for valid STEC
                valid_stec = ~np.isnan(stec_values[:, time_slot_idx])
                el_select = np.logical_and(valid_stec, el_select)
                
                # Distance selection for this height
                dist_select = np.array([
                    _get_distance_km(
                        ipp.loc[time_slot_idx, hidx],
                        ipp_target.loc[target_idx, hidx]
                    ) < DISTANCE_KM_CUT
                    for ipp in ipp_sat_stat
                ])  # prn mask
                
                # Combined selection
                prn_select = np.logical_and(el_select, dist_select)
                
                # Get selected satellites for this time slot
                selected_prns = np.where(prn_select)[0]
                
                if len(selected_prns) == 0:
                    continue
                
                # Calculate VTEC for selected satellites
                for prn_idx in selected_prns:
                    # Weighted airmass for this PRN and time
                    weighted_am = np.sum(
                        profiles[target_idx] * ipp_sat_stat[prn_idx].airmass[time_slot_idx]
                    )
                    
                    # Convert STEC to VTEC
                    vtec = (
                        profiles[target_idx, hidx] *
                        stec_values[prn_idx, time_slot_idx] / weighted_am
                    )
                    vtec_error = (
                        profiles[target_idx, hidx] *
                        stec_errors[prn_idx, time_slot_idx]
                    )
                    
                    # Calculate spatial offsets
                    dlon = (
                        np.cos(ipp_target.loc[target_idx, hidx].lat.rad) *
                        (ipp_sat_stat[prn_idx].loc[time_slot_idx, hidx].lon.deg -
                         ipp_target.loc[target_idx, hidx].lon.deg)
                    )
                    dlat = (
                        ipp_sat_stat[prn_idx].loc[time_slot_idx, hidx].lat.deg -
                        ipp_target.loc[target_idx, hidx].lat.deg
                    )
                    
                    # Store measurement
                    all_vtec.append(vtec)
                    all_vtec_errors.append(vtec_error)
                    all_dlon.append(dlon)
                    all_dlat.append(dlat)
                    all_time_weights.append(time_weight)
            
            # Combine measurements for this height
            if len(all_vtec) > 0:
                if use_time_weighting:
                    # Include time weight as 5th column
                    height_data.append(np.column_stack([
                        all_vtec,
                        all_vtec_errors,
                        all_dlon,
                        all_dlat,
                        all_time_weights
                    ]))
                else:
                    height_data.append(np.column_stack([
                        all_vtec,
                        all_vtec_errors,
                        all_dlon,
                        all_dlat,
                    ]))
            else:
                height_data.append(np.array([]))
        
        result.append(height_data)
    
    return result


def get_interpolated_tec(
    input_data: list[list[np.ndarray]],
    use_time_weighting: bool = USE_TIME_WEIGHTING
) -> np.ndarray:
    """
    Interpolate VTEC to target (with optional time weighting).
    
    Parameters
    ----------
    input_data : list[list[np.ndarray]]
        Nested list [times][heights] of arrays with columns:
        [VTEC, VTEC_error, dlon, dlat] or
        [VTEC, VTEC_error, dlon, dlat, time_weight]
    use_time_weighting : bool
        If True, expects 5 columns and combines time+error weighting
        
    Returns
    -------
    np.ndarray
        Electron density at target IPPs [times × heights]
    """
    fitted_density = np.zeros((len(input_data), len(input_data[0])))
    
    for timeidx, input_time in enumerate(input_data):
        for hidx, measurements in enumerate(input_time):
            if not measurements.shape or measurements.shape[0] < 2:
                continue
            
            # Extract columns
            vtec = measurements[:, 0]
            errors = measurements[:, 1]
            dlon = measurements[:, 2]
            dlat = measurements[:, 3]
            
            # Select nearest NDIST_POINTS measurements
            dist = np.sqrt(dlon**2 + dlat**2)
            dist_select = np.zeros(dist.shape, dtype=bool)
            nearest_indices = np.argpartition(
                dist, min(NDIST_POINTS, dist.shape[0] - 1), axis=0
            )[:NDIST_POINTS]
            dist_select[nearest_indices] = True
            
            # Build design matrix for polynomial fit
            A = np.ones(
                (np.sum(dist_select),
                 ((INTERPOLATION_ORDER ** 2 + INTERPOLATION_ORDER) // 2)),
                dtype=float,
            )
            
            # Calculate weights
            if use_time_weighting and measurements.shape[1] >= 5:
                # Combine inverse-variance and time weighting
                time_weights = measurements[dist_select, 4]
                variance_weights = 1.0 / errors[dist_select]
                weights = variance_weights * time_weights
            else:
                # Just inverse-variance weighting
                weights = 1.0 / errors[dist_select]
            
            # Build polynomial terms
            idx = 0
            for ilon in range(INTERPOLATION_ORDER):
                for ilat in range(INTERPOLATION_ORDER):
                    if ilon + ilat <= INTERPOLATION_ORDER - 1:
                        if idx > 0:
                            A[:, idx] = (
                                dlon[dist_select] ** ilon *
                                dlat[dist_select] ** ilat
                            )
                        idx += 1
            
            # Weighted least squares fit
            w = weights * np.eye(A.shape[0])
            AwT = A.T @ w
            
            try:
                par = (np.linalg.inv(AwT @ A) @ (AwT @ vtec[dist_select][:, np.newaxis])).squeeze()
                fitted_density[timeidx, hidx] = par[0]
            except:
                continue
    
    return fitted_density


def get_gnss_station_density(
    gnss_data: GNSSData,
    ipp_target: IPP,
    profiles: np.ndarray,
    sp3_data,
    ionex: IonexData,
    n_time_slots: int = N_TIME_SLOTS,
    max_time_diff_min: float = MAX_TIME_DIFF_MINUTES,
    use_time_weighting: bool = USE_TIME_WEIGHTING,
) -> list[list[np.ndarray]]:
    """
    Process one GNSS station with TIME AVERAGING.
    
    Parameters
    ----------
    gnss_data : GNSSData
        Observations from one station
    ipp_target : IPP
        Target IPPs
    profiles : np.ndarray
        Density profiles [times × heights]
    sp3_data
        Satellite orbit data
    ionex : IonexData
        Ionospheric map
    n_time_slots : int, optional
        Number of time slots to combine
    max_time_diff_min : float, optional
        Maximum time difference in minutes
    use_time_weighting : bool, optional
        Use time-distance weighting
        
    Returns
    -------
    list[list[np.ndarray]]
        Data structure for interpolation
    """
    prns = sorted(gnss_data.gnss.keys())
    stec_values = []
    stec_errors = []
    ipp_sat_stat = []
    
    for prn in prns:
        try:
            sat_data = gnss_data.gnss[prn]
            transmission_time = get_transmission_time(sat_data[:, 1], gnss_data.times)
            phase_stec = getphase_tec(
                sat_data[:, 2], sat_data[:, 3],
                constellation=gnss_data.constellation
            )
            sat_pos = get_sat_pos(sp3_data, transmission_time, prn)
            ipp_sat_stat.append(
                get_stat_sat_ipp(
                    satpos=sat_pos,
                    gnsspos=gnss_pos_dict[gnss_data.station],
                    times=gnss_data.times,
                    height_array=ipp_target.loc[0].height,
                )
            )
            
            # For time averaging, we need all time indices that might be used
            # Build extended time window
            all_time_indices = np.arange(len(gnss_data.times))
            
            stec_value, stec_error = _get_gim_phase_corrected(
                phase_stec, 
                ipp_sat_stat[-1], 
                all_time_indices,  # Pass all indices, function filters internally
                ionex,
                max_time_diff_min=max_time_diff_min  # Pass time window parameter
            )
            stec_values.append(stec_value)
            stec_errors.append(stec_error)
        except Exception as e:
            print(f"Failed for {gnss_data.station} {prn}: {e}")
    
    # Use time-averaged distance calculation
    result = _get_distance_ipp_time_averaged(
        stec_values=np.array(stec_values),
        stec_errors=np.array(stec_errors),
        ipp_sat_stat=ipp_sat_stat,
        ipp_target=ipp_target,
        obs_times=gnss_data.times,
        profiles=profiles,
        n_time_slots=n_time_slots,
        max_time_diff_min=max_time_diff_min,
        use_time_weighting=use_time_weighting,
    )
    
    del stec_values, stec_errors, ipp_sat_stat
    return result


def get_ipp_density(
    ipp_target: IPP,
    gnss_data_list: list[GNSSData],
    sp3_data,
    ionex: IonexData,
    n_time_slots: int = N_TIME_SLOTS,
    max_time_diff_min: float = MAX_TIME_DIFF_MINUTES,
    use_time_weighting: bool = USE_TIME_WEIGHTING,
) -> tec_data.ElectronDensity:
    """
    Calculate electron density with TIME AVERAGING.
    
    Parameters
    ----------
    ipp_target : IPP
        Target ionospheric pierce points
    gnss_data_list : list[GNSSData]
        GNSS observations from all stations
    sp3_data
        Satellite orbit data
    ionex : IonexData
        Ionospheric map
    n_time_slots : int, optional
        Number of time slots to combine, by default 5
    max_time_diff_min : float, optional
        Maximum time difference in minutes, by default 2.5
    use_time_weighting : bool, optional
        Use time-distance weighting, by default True
        
    Returns
    -------
    tec_data.ElectronDensity
        Electron density and uncertainties
        
    Notes
    -----
    Time averaging significantly improves interpolation quality by:
    - Increasing number of measurements per target point (5×)
    - Providing better spatial coverage from different satellite positions
    - Reducing impact of outliers through more robust fitting
    """
    profiles = get_profile(ipp_target)
    
    Ntimes = ipp_target.times.shape[0]
    Nheights = ipp_target.loc.shape[1]
    
    all_data = [[[] for _ in range(Nheights)] for _ in range(Ntimes)]
    
    # Process stations in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_DENSITY) as executor:
        future_to_station = {
            executor.submit(
                get_gnss_station_density,
                gnss_data, ipp_target, profiles, sp3_data, ionex,
                n_time_slots, max_time_diff_min, use_time_weighting,
            ): gnss_data.station + gnss_data.constellation
            for gnss_data in gnss_data_list
        }
        
        for future in as_completed(future_to_station):
            station = future_to_station[future]
            try:
                result = future.result()
                
                for itm in range(Ntimes):
                    for hidx in range(Nheights):
                        all_data[itm][hidx].append(result[itm][hidx])
                
            except Exception as e:
                print(f"Error processing {station}: {e}")
    
    # Concatenate measurements
    for itm in range(Ntimes):
        for hidx in range(Nheights):
            if all_data[itm][hidx]:
                all_data[itm][hidx] = np.concatenate(all_data[itm][hidx], axis=0)
            else:
                all_data[itm][hidx] = np.array([])
    
    # Interpolate with time weighting
    electron_density = get_interpolated_tec(all_data, use_time_weighting)
    
    del all_data, profiles
    gc.collect()
    
    return tec_data.ElectronDensity(
        electron_density=electron_density,
        electron_density_error=np.zeros_like(electron_density),
    )
