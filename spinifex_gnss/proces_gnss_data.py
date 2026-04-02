"""
GNSS data processing with optional time averaging.

This module provides electron density calculation from GNSS observations
with optional time averaging for improved spatial coverage.

Key features:
- Nearest-neighbor mode (fast, default)
- Time averaging mode (better coverage, still fast with vectorization)
- GIM bias correction with arc optimization
- Memory-efficient processing
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation
from concurrent.futures import as_completed, ProcessPoolExecutor
import gc

from spinifex.geometry import IPP, R_EARTH_MEAN
from spinifex.ionospheric import tec_data
from spinifex.ionospheric.ionex_manipulation import interpolate_ionex, IonexData
from spinifex.ionospheric.iri_density import get_profile
from spinifex_gnss.parse_dcb import DCBData, get_satellite_dcb, get_receiver_dcb_c1c2
from spinifex_gnss.parse_sp3 import SP3Data
from spinifex_gnss.parse_gnss import GNSSData
from spinifex_gnss.gnss_geometry import (
    get_sat_pos,
    get_stat_sat_ipp,
    _convert_ipp_lonlatr_to_xyz,
)
from spinifex_gnss.gnss_stations import gnss_pos_dict
from spinifex_gnss.tec_core import (
    getphase_tec,
    get_transmission_time,
    get_cycle_slips,
    getpseudorange_tec,
)
from spinifex_gnss.config import (
    DISTANCE_KM_CUT,
    NDIST_POINTS,
    ELEVATION_CUT,
    ELEVATION_CUT_BIAS,
    INTERPOLATION_ORDER,
    MAX_WORKERS_DENSITY,
    MIN_OBSERVATIONS_PER_SEGMENT,
    RinexStrategy,
    DCB_ERROR_FLOOR_TECU,
    GIM_ERROR_FLOOR_TECU,
    MAX_PSEUDO_PHASE_STD_TECU,
)

# ============================================================================
# Helper Functions
# ============================================================================


def _get_distance_km(loc1: u.Quantity, loc2: u.Quantity) -> np.ndarray:
    """Calculate distance between two sets of locations in km."""

    return np.linalg.norm(loc1.to(u.km).value - loc2.to(u.km).value, axis=-1)


def _get_phase_corrected_with_dcb(
    phase_tec: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    ipp_sat_stat: IPP,
    constellation: str = "G",
    tec_coefficient: tuple = None,
    satellite_dcb_ns: float = None,
    receiver_dcb_ns: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove phase bias using DCB-corrected pseudorange TEC.

    Parameters
    ----------
    phase_tec : np.ndarray
        STEC from carrier phases (has bias)
    c1 : np.ndarray
        Pseudorange for frequency f1
    c2 : np.ndarray
        Pseudorange for frequency f2
    constellation : str
        Satellite constellation
    tec_coefficient : tuple
        (C12, f1, f2) for GLONASS FDMA
    satellite_dcb_ns : float
        Satellite DCB in nanoseconds
    receiver_dcb_ns : float
        Receiver DCB in nanoseconds

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - Bias-corrected phase TEC
        - Standard deviation of bias estimate per segment
    """
    # Calculate DCB-corrected pseudorange TEC
    pseudo_tec = getpseudorange_tec(
        c1=c1,
        c2=c2,
        constellation=constellation,
        tec_coefficient=tec_coefficient,
        satellite_dcb_ns=satellite_dcb_ns,
        receiver_dcb_ns=receiver_dcb_ns,
    )

    # Detect cycle slips
    cycle_slips = get_cycle_slips(phase_tec)
    phase_bias = np.full_like(phase_tec, np.nan)
    phase_std = np.full_like(phase_tec, np.nan)

    for seg in np.unique(cycle_slips):
        seg_idx = np.nonzero(cycle_slips == seg)[0]
        diff = pseudo_tec[seg_idx] - phase_tec[seg_idx]
        elevation = ipp_sat_stat.altaz.alt.deg[seg_idx]
        valid = ~np.isnan(diff) & (elevation >= ELEVATION_CUT_BIAS)
        n_valid = np.sum(valid)
        if n_valid < MIN_OBSERVATIONS_PER_SEGMENT:
            # Too few pseudorange points — noise dominates, bias unreliable.
            # Leave as NaN so this arc is excluded from the fit.
            continue
        if np.nanstd(diff) > MAX_PSEUDO_PHASE_STD_TECU:
            continue
        bias = np.nanmean(diff)
        std = np.nanstd(diff[valid]) / np.sqrt(n_valid) + DCB_ERROR_FLOOR_TECU
        phase_bias[seg_idx] = bias
        phase_std[seg_idx] = std

    return phase_tec + phase_bias, phase_std


def _get_gim_phase_corrected(
    phase_tec: np.ndarray,
    ipp_sat_stat: IPP,
    timeselect: np.ndarray,
    ionex: IonexData,
    max_time_diff_min: float = 2.5,
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
    with the extended time window for time averaging.
    """
    cycle_slips = get_cycle_slips(phase_tec=phase_tec)
    phase_bias = np.full_like(phase_tec, np.nan)
    phase_std = np.full_like(phase_tec, np.nan)

    default_options = tec_data.IonexOptions(remove_midnight_jumps=True)
    h_idx = np.argmin(
        np.abs(
            (ipp_sat_stat.height[0] - R_EARTH_MEAN).to(u.km).value
            - default_options.height.to(u.km).value
        )
    )

    # Build extended time window for time averaging
    if len(timeselect) > 0:
        # Convert max_time_diff to observation index units
        # Assuming ~30-second observations
        time_buffer_indices = int(np.ceil(max_time_diff_min * 60 / 30))

        # Expand timeselect window
        extended_timeselect = set()
        for t_idx in timeselect:
            start_idx = max(0, t_idx - time_buffer_indices)
            end_idx = min(len(phase_tec), t_idx + time_buffer_indices + 1)
            extended_timeselect.update(range(start_idx, end_idx))

        extended_timeselect = np.array(sorted(extended_timeselect))
    else:
        extended_timeselect = timeselect

    for seg in np.unique(cycle_slips):
        seg_idx = np.nonzero(cycle_slips == seg)[0]

        if seg_idx.shape[0] < MIN_OBSERVATIONS_PER_SEGMENT:
            phase_bias[seg_idx] = np.nan
            continue

        # OPTIMIZATION: Skip arcs that don't overlap with extended time window
        if np.intersect1d(seg_idx, extended_timeselect).size == 0:
            continue

        elevation = ipp_sat_stat.altaz.alt.deg[seg_idx]

        gim_tec = interpolate_ionex(
            ionex,
            ipp_sat_stat.lon[:, h_idx][seg_idx].to(u.deg).value,
            ipp_sat_stat.lat[:, h_idx][seg_idx].to(u.deg).value,
            ipp_sat_stat.times[seg_idx],
            apply_earth_rotation=default_options.apply_earth_rotation,
        )

        high_el_mask = elevation > ELEVATION_CUT_BIAS

        if np.sum(high_el_mask) > 0:
            phase_bias[seg_idx] = np.nanmean(
                gim_tec[high_el_mask]
                * ipp_sat_stat.airmass[:, h_idx][seg_idx][high_el_mask]
                - phase_tec[seg_idx][high_el_mask],
            )

            data_count = np.sum(~np.isnan(phase_tec[seg_idx][high_el_mask]))
            if data_count > 1:
                # std/sqrt(N) is the statistical uncertainty on the bias estimate.
                # Add GIM_ERROR_FLOOR to represent the systematic GIM error that
                # does not average down with more observations, and ensures
                # GIM-corrected data is always weighted lower than DCB-corrected data.
                phase_std[seg_idx] = (
                    np.nanstd(
                        gim_tec[high_el_mask]
                        * ipp_sat_stat.airmass[:, h_idx][seg_idx][high_el_mask]
                        - phase_tec[seg_idx][high_el_mask]
                    )
                    / np.sqrt(data_count)
                    + GIM_ERROR_FLOOR_TECU
                )
            else:
                phase_std[seg_idx] = np.nan

    return phase_tec + phase_bias, phase_std


# ============================================================================
# Time Averaging Functions (Vectorized)
# ============================================================================


def _build_time_mapping_vectorized(
    target_times_mjd: np.ndarray,
    obs_times_mjd: np.ndarray,
    n_slots: int = 5,
    max_diff_min: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build time mapping for ALL target times at once (vectorized).

    This is much faster than calling _select_time_window in a loop.

    Parameters
    ----------
    target_times_mjd : np.ndarray
        Target times [n_targets]
    obs_times_mjd : np.ndarray
        Observation times [n_obs]
    n_slots : int
        Number of slots to select per target
    max_diff_min : float
        Maximum time difference in minutes

    Returns
    -------
    time_mapping : np.ndarray
        Indices of selected obs times [n_targets, n_slots]
        Values may be -1 for unused slots
    time_weights : np.ndarray
        Weights for each selected time [n_targets, n_slots]
    """
    n_targets = len(target_times_mjd)

    # Calculate all pairwise time differences at once
    # Shape: (n_targets, n_obs)
    time_diffs_min = (
        np.abs(target_times_mjd[:, np.newaxis] - obs_times_mjd[np.newaxis, :]) * 24 * 60
    )

    # Initialize outputs
    time_mapping = np.full((n_targets, n_slots), -1, dtype=int)
    time_weights = np.zeros((n_targets, n_slots), dtype=float)

    # For each target, select n_slots nearest observations
    for tidx in range(n_targets):
        valid_mask = time_diffs_min[tidx] <= max_diff_min

        if not np.any(valid_mask):
            # No observations within window - use nearest
            nearest_idx = np.argmin(time_diffs_min[tidx])
            time_mapping[tidx, 0] = nearest_idx
            time_weights[tidx, 0] = 1.0
            continue

        # Get indices and distances of valid observations
        valid_indices = np.where(valid_mask)[0]
        valid_diffs = time_diffs_min[tidx, valid_mask]

        # Select n_slots nearest
        n_select = min(n_slots, len(valid_indices))

        if n_select == len(valid_indices):
            selected_local = np.arange(len(valid_indices))
        else:
            selected_local = np.argpartition(valid_diffs, n_select - 1)[:n_select]

        selected_indices = valid_indices[selected_local]
        selected_diffs = valid_diffs[selected_local]

        # Store mapping
        time_mapping[tidx, : len(selected_indices)] = selected_indices

        # Calculate weights (inverse time distance)
        weights = 1.0 / (selected_diffs + 0.1)
        weights /= np.sum(weights)  # Normalize
        time_weights[tidx, : len(selected_indices)] = weights

    return time_mapping, time_weights


def _get_distance_ipp_time_averaged(
    stec_values: np.ndarray,
    stec_errors: np.ndarray,
    ipp_sat_stat: list[IPP],
    ipp_target: IPP,
    time_mapping: np.ndarray,
    time_weights: np.ndarray,
    profiles: np.ndarray,
    use_time_weighting: bool = True,
) -> list[list[np.ndarray]]:
    """
    Calculate VTEC with time averaging (vectorized).

    Parameters
    ----------
    stec_values : np.ndarray
        STEC values [n_prns, n_obs_times]
    stec_errors : np.ndarray
        STEC errors [n_prns, n_obs_times]
    ipp_sat_stat : list[IPP]
        IPPs for each satellite
    ipp_target : IPP
        Target IPPs
    time_mapping : np.ndarray
        Pre-computed time indices [n_targets, n_slots]
    time_weights : np.ndarray
        Pre-computed time weights [n_targets, n_slots]
    profiles : np.ndarray
        Density profiles [n_targets, n_heights]
    use_time_weighting : bool
        Include time weights in output

    Returns
    -------
    list[list[np.ndarray]]
        Nested list [times][heights] of arrays with columns:
        [VTEC, error, dlon, dlat, time_weight (optional)]
    """
    Ntimes_target = ipp_target.times.shape[0]
    Nheights = ipp_target.lon[0].shape[0]
    Nprns = stec_values.shape[0]

    result = []

    # Process each target time
    for target_idx in range(Ntimes_target):
        # Get time slot indices for this target
        slot_indices = time_mapping[target_idx]
        valid_slots = slot_indices >= 0
        slot_indices = slot_indices[valid_slots]
        slot_weights = time_weights[target_idx, valid_slots]

        if len(slot_indices) == 0:
            result.append([np.array([]) for _ in range(Nheights)])
            continue

        # Pre-compute selection masks for all slots
        el_select_all = np.array(
            [
                [
                    ipp.altaz.alt.deg[slot_idx] > ELEVATION_CUT
                    for slot_idx in slot_indices
                ]
                for ipp in ipp_sat_stat
            ]
        )

        valid_stec_all = np.array(
            [
                [~np.isnan(stec_values[prn_idx, slot_idx]) for slot_idx in slot_indices]
                for prn_idx in range(Nprns)
            ]
        )

        el_select_all = np.logical_and(el_select_all, valid_stec_all)

        height_data = []

        loc1_list = [_convert_ipp_lonlatr_to_xyz(ipp) for ipp in ipp_sat_stat]
        loc2 = _convert_ipp_lonlatr_to_xyz(ipp_target)

        # Process each height
        for hidx in range(Nheights):
            # Distance selection for this height
            dist_select_all = np.array(
                [
                    [
                        _get_distance_km(loc1[slot_idx, hidx], loc2[target_idx, hidx])
                        < DISTANCE_KM_CUT
                        for slot_idx in slot_indices
                    ]
                    for loc1 in loc1_list
                ]
            )

            # Combined selection
            prn_select_all = np.logical_and(el_select_all, dist_select_all)

            # Collect measurements
            all_vtec = []
            all_vtec_errors = []
            all_dlon = []
            all_dlat = []
            all_time_weights = []

            for slot_local_idx, (slot_idx, slot_weight) in enumerate(
                zip(slot_indices, slot_weights)
            ):
                selected_prns = np.where(prn_select_all[:, slot_local_idx])[0]

                if len(selected_prns) == 0:
                    continue

                for prn_idx in selected_prns:
                    weighted_am = np.sum(
                        profiles[target_idx] * ipp_sat_stat[prn_idx].airmass[slot_idx]
                    )

                    vtec = (
                        profiles[target_idx, hidx]
                        * stec_values[prn_idx, slot_idx]
                        / weighted_am
                    )
                    vtec_error = (
                        profiles[target_idx, hidx] * stec_errors[prn_idx, slot_idx]
                    )

                    dlon = np.cos(ipp_target.lat[target_idx, hidx].to(u.rad).value) * (
                        ipp_sat_stat[prn_idx].lon[slot_idx, hidx].to(u.deg).value
                        - ipp_target.lon[target_idx, hidx].to(u.deg).value
                    )
                    dlat = (
                        ipp_sat_stat[prn_idx].lat[slot_idx, hidx].to(u.deg).value
                        - ipp_target.lat[target_idx, hidx].to(u.deg).value
                    )

                    all_vtec.append(vtec)
                    all_vtec_errors.append(vtec_error)
                    all_dlon.append(dlon)
                    all_dlat.append(dlat)
                    all_time_weights.append(slot_weight)

            # Combine measurements
            if len(all_vtec) > 0:
                if use_time_weighting:
                    height_data.append(
                        np.column_stack(
                            [
                                all_vtec,
                                all_vtec_errors,
                                all_dlon,
                                all_dlat,
                                all_time_weights,
                            ]
                        )
                    )
                else:
                    height_data.append(
                        np.column_stack([all_vtec, all_vtec_errors, all_dlon, all_dlat])
                    )
            else:
                height_data.append(np.array([]))

        result.append(height_data)

    return result


# ============================================================================
# Nearest-Neighbor Functions (Original, Fast)
# ============================================================================


def _get_distance_ipp_nearest(
    stec_values: np.ndarray,
    stec_errors: np.ndarray,
    ipp_sat_stat: list[IPP],
    ipp_target: IPP,
    timeselect: np.ndarray,
    profiles: np.ndarray,
) -> list[list[np.ndarray]]:
    """Calculate VTEC using nearest-neighbor time matching (original method)."""
    Ntimes = ipp_target.times.shape[0]
    Nheights = ipp_target.lon[0].shape[0]
    Nprns = stec_values.shape[0]

    vtecs = np.full((Nprns, Ntimes, Nheights), np.nan, dtype=float)
    vtec_errors = np.full((Nprns, Ntimes, Nheights), np.nan, dtype=float)

    el_select = np.array(
        [ipp.altaz.alt.deg[timeselect] > ELEVATION_CUT for ipp in ipp_sat_stat]
    )

    el_select = np.logical_and(~np.isnan(stec_values[:, timeselect]), el_select)
    loc1_list = [_convert_ipp_lonlatr_to_xyz(ipp) for ipp in ipp_sat_stat]
    loc2 = _convert_ipp_lonlatr_to_xyz(ipp_target)
    dist_select = np.array(
        [
            _get_distance_km(loc1[timeselect], loc2) < DISTANCE_KM_CUT
            for loc1 in loc1_list
        ]
    )

    prn_select = np.logical_and(el_select[:, :, np.newaxis], dist_select)

    weighted_am = np.array([profiles * ipp.airmass[timeselect] for ipp in ipp_sat_stat])
    weighted_am = np.sum(weighted_am, axis=-1)

    vtec_values = profiles * (stec_values[:, timeselect] / weighted_am)[..., np.newaxis]
    vtec_error_values = profiles * stec_errors[:, timeselect][..., np.newaxis]

    dlons = np.array(
        [
            np.cos(ipp_target.lat.to(u.rad).value)
            * (ipp.lon[timeselect].to(u.deg).value - ipp_target.lon.to(u.deg).value)
            for ipp in ipp_sat_stat
        ]
    )

    dlats = np.array(
        [
            ipp.lat[timeselect].to(u.deg).value - ipp_target.lat.to(u.deg).value
            for ipp in ipp_sat_stat
        ]
    )

    vtecs[prn_select] = vtec_values[prn_select]
    vtec_errors[prn_select] = vtec_error_values[prn_select]

    return [
        [
            np.concatenate(
                (
                    vtecs[:, timeidx, hidx][~np.isnan(vtecs[:, timeidx, hidx])][
                        :, np.newaxis
                    ],
                    vtec_errors[:, timeidx, hidx][~np.isnan(vtecs[:, timeidx, hidx])][
                        :, np.newaxis
                    ],
                    dlons[:, timeidx, hidx][~np.isnan(vtecs[:, timeidx, hidx])][
                        :, np.newaxis
                    ],
                    dlats[:, timeidx, hidx][~np.isnan(vtecs[:, timeidx, hidx])][
                        :, np.newaxis
                    ],
                ),
                axis=-1,
            )
            for hidx in range(Nheights)
        ]
        for timeidx in range(Ntimes)
    ]


# ============================================================================
# Interpolation
# ============================================================================


def get_interpolated_tec(
    input_data: list[list[np.ndarray]], use_time_weighting: bool = False
) -> np.ndarray:
    """
    Interpolate VTEC to target locations.

    Parameters
    ----------
    input_data : list[list[np.ndarray]]
        Nested list [times][heights] of arrays with columns:
        [VTEC, error, dlon, dlat] or [VTEC, error, dlon, dlat, time_weight]
    use_time_weighting : bool
        If True, expects 5 columns with time weights

    Returns
    -------
    np.ndarray
        Electron density [times × heights]
    """
    fitted_density = np.zeros((len(input_data), len(input_data[0])))

    for timeidx, input_time in enumerate(input_data):
        for hidx, measurements in enumerate(input_time):
            if not measurements.shape or measurements.shape[0] < 2:
                print(f"DEBUG: No measurements at time {timeidx}, height {hidx}")
                continue

            vtec = measurements[:, 0]
            errors = measurements[:, 1]
            dlon = measurements[:, 2]
            dlat = measurements[:, 3]
            if use_time_weighting and measurements.shape[1] >= 5:
                time_weights = measurements[:, 4]
            # filter nans
            if np.any(np.isnan(measurements)):
                nan_select = np.any(np.isnan(measurements), axis=1)
                vtec = vtec[~nan_select]
                errors = errors[~nan_select]
                dlon = dlon[~nan_select]
                dlat = dlat[~nan_select]
                if use_time_weighting and measurements.shape[1] >= 5:
                    time_weights = time_weights[~nan_select]
            # Select nearest measurements
            dist = np.sqrt(dlon**2 + dlat**2)
            dist_select = np.zeros(dist.shape, dtype=bool)
            nearest_indices = np.argpartition(
                dist, min(NDIST_POINTS, dist.shape[0] - 1), axis=0
            )[:NDIST_POINTS]
            dist_select[nearest_indices] = True

            # Build design matrix
            A = np.ones(
                (
                    np.sum(dist_select),
                    ((INTERPOLATION_ORDER**2 + INTERPOLATION_ORDER) // 2),
                ),
                dtype=float,
            )

            # Calculate weights
            if use_time_weighting and measurements.shape[1] >= 5:
                time_weights = time_weights[dist_select]
                variance_weights = 1.0 / errors[dist_select]
                weights = variance_weights * time_weights
            else:
                weights = 1.0 / errors[dist_select]

            # Build polynomial terms
            idx = 0
            for ilon in range(INTERPOLATION_ORDER):
                for ilat in range(INTERPOLATION_ORDER):
                    if ilon + ilat <= INTERPOLATION_ORDER - 1:
                        if idx > 0:
                            A[:, idx] = (
                                dlon[dist_select] ** ilon * dlat[dist_select] ** ilat
                            )
                        idx += 1

            # Weighted least squares
            w = weights * np.eye(A.shape[0])
            AwT = A.T @ w

            try:
                par = (
                    np.linalg.inv(AwT @ A) @ (AwT @ vtec[dist_select][:, np.newaxis])
                ).squeeze()
                fitted_density[timeidx, hidx] = par[0]
            except np.linalg.LinAlgError as e:
                print(f"⚠ Interpolation failed at time={timeidx}, height={hidx}: {e}")
                continue
            except Exception as e:
                print(
                    f"❌ Unexpected error at time={timeidx}, height={hidx}: {type(e).__name__}: {e}"
                )
                continue
    return fitted_density


# ============================================================================
# Main Processing Functions
# ============================================================================


def get_gnss_station_density(
    gnss_data: GNSSData,
    ipp_target: IPP,
    profiles: np.ndarray,
    sp3_data,
    ionex: IonexData,
    dcb_data: DCBData | None = None,
    n_time_slots: int = 1,
    max_time_diff_min: float = 2.5,
    use_time_weighting: bool = False,
    strategy: RinexStrategy = RinexStrategy.DCB_WITH_GIM_FALLBACK,
) -> list[list[np.ndarray]]:
    """
    Process one GNSS station with optional time averaging.

    Parameters
    ----------
    gnss_data : GNSSData
        Observations from one station
    ipp_target : IPP
        Target IPPs
    profiles : np.ndarray
        Density profiles
    sp3_data
        Satellite positions
    ionex : IonexData
        Global ionospheric map
    dcb_data : DCBData, optional
        DCB corrections for satellites and receivers
    n_time_slots : int, optional
        Number of time slots to average (1 = nearest neighbor)
    max_time_diff_min : float, optional
        Maximum time difference for averaging (minutes)
    use_time_weighting : bool, optional
        Weight by time distance

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
            # Choose correction method based on strategy
            satellite_dcb_ns = None
            receiver_dcb_ns = None
            # Choose correction method based on DCB availability
            # Try DCB-based correction first
            obs1 = gnss_data.c1_str  # e.g., 'C1W', 'C1P', 'C1C'
            obs2 = gnss_data.c2_str  # e.g., 'C2W', 'C2P', 'C2C'
            if dcb_data is not None and strategy in (
                RinexStrategy.DCB_ONLY,
                RinexStrategy.DCB_WITH_GIM_FALLBACK,
            ):
                satellite_dcb_ns = get_satellite_dcb(
                    dcb_data.satellite_dcb, prn, obs1, obs2
                )
                receiver_dcb_ns = get_receiver_dcb_c1c2(
                    dcb_data.receiver_dcb,
                    gnss_data.station,
                    obs1,
                    obs2,
                    constellation=gnss_data.constellation,
                )

            have_dcb = satellite_dcb_ns is not None and receiver_dcb_ns is not None
            #check if we need to process this one
            if not have_dcb and strategy==RinexStrategy.DCB_ONLY:
                print(f"No dcb for {gnss_data.station} {prn}: {e} and user requested DCB_ONLY")
                continue
            
            tec_coeff = None
            sat_data = gnss_data.gnss[prn]
            if not gnss_data.tec_coefficients is None:
                if prn in gnss_data.tec_coefficients:
                    tec_coeff = gnss_data.tec_coefficients[prn]
            transmission_time = get_transmission_time(sat_data[:, 1], gnss_data.times)
            phase_stec = getphase_tec(
                sat_data[:, 2],
                sat_data[:, 3],
                constellation=gnss_data.constellation,
                tec_coefficient=tec_coeff,
            )
            sat_pos = get_sat_pos(sp3_data, transmission_time, prn)
            ipp_sat_stat.append(
                get_stat_sat_ipp(
                    satpos=sat_pos,
                    gnsspos=gnss_pos_dict[gnss_data.station],
                    times=transmission_time,
                    height_array=ipp_target.height[0] - R_EARTH_MEAN,
                )
            )

            all_time_indices = np.arange(len(gnss_data.times))

            if strategy == RinexStrategy.DCB_ONLY:
                if not have_dcb:
                    # No DCB available and no GIM fallback — skip satellite
                    continue
                stec_value, stec_error = _get_phase_corrected_with_dcb(
                    phase_tec=phase_stec,
                    c1=sat_data[:, 0],
                    c2=sat_data[:, 1],
                    ipp_sat_stat=ipp_sat_stat[-1],
                    constellation=gnss_data.constellation,
                    tec_coefficient=tec_coeff,
                    satellite_dcb_ns=satellite_dcb_ns,
                    receiver_dcb_ns=receiver_dcb_ns,
                )

            elif strategy == RinexStrategy.GIM_ONLY:
                stec_value, stec_error = _get_gim_phase_corrected(
                    phase_stec,
                    ipp_sat_stat[-1],
                    all_time_indices,
                    ionex,
                    max_time_diff_min=max_time_diff_min,
                )

            else:  # DCB_WITH_GIM_FALLBACK
                if have_dcb:
                    stec_value, stec_error = _get_phase_corrected_with_dcb(
                        phase_tec=phase_stec,
                        c1=sat_data[:, 0],
                        c2=sat_data[:, 1],
                        ipp_sat_stat=ipp_sat_stat[-1],
                        constellation=gnss_data.constellation,
                        tec_coefficient=tec_coeff,
                        satellite_dcb_ns=satellite_dcb_ns,
                        receiver_dcb_ns=receiver_dcb_ns,
                    )
                else:
                    stec_value, stec_error = _get_gim_phase_corrected(
                        phase_stec,
                        ipp_sat_stat[-1],
                        all_time_indices,
                        ionex,
                        max_time_diff_min=max_time_diff_min,
                    )
            stec_values.append(stec_value)
            stec_errors.append(stec_error)
        except Exception as e:
            print(f"Failed for {gnss_data.station} {prn}: {e}")

    if len(stec_values) == 0:
        Ntimes = ipp_target.times.shape[0]
        Nheights = ipp_target.lon[0].shape[0]
        return [[np.array([]) for _ in range(Nheights)] for _ in range(Ntimes)]

    stec_values = np.array(stec_values)
    stec_errors = np.array(stec_errors)

    # Choose processing method based on n_time_slots
    if n_time_slots == 1:
        # Nearest-neighbor (fast, original method)
        timeselect = np.argmin(
            np.abs(ipp_target.times.utc.mjd - gnss_data.times.utc.mjd[:, np.newaxis]),
            axis=0,
        )

        result = _get_distance_ipp_nearest(
            stec_values=stec_values,
            stec_errors=stec_errors,
            ipp_sat_stat=ipp_sat_stat,
            ipp_target=ipp_target,
            timeselect=timeselect,
            profiles=profiles,
        )
    else:
        # Time averaging (better coverage, still fast with vectorization)
        time_mapping, time_weights = _build_time_mapping_vectorized(
            ipp_target.times.utc.mjd,
            gnss_data.times.utc.mjd,
            n_slots=n_time_slots,
            max_diff_min=max_time_diff_min,
        )

        result = _get_distance_ipp_time_averaged(
            stec_values=stec_values,
            stec_errors=stec_errors,
            ipp_sat_stat=ipp_sat_stat,
            ipp_target=ipp_target,
            time_mapping=time_mapping,
            time_weights=time_weights,
            profiles=profiles,
            use_time_weighting=use_time_weighting,
        )

    del stec_values, stec_errors, ipp_sat_stat
    return result


def get_ipp_density(
    ipp_target: IPP,
    gnss_data_list: list[GNSSData],
    sp3_data: SP3Data,
    ionex: IonexData,
    dcb_data: DCBData,
    n_time_slots: int = 1,
    max_time_diff_min: float = 2.5,
    use_time_weighting: bool = False,
    max_workers: int = MAX_WORKERS_DENSITY,
    strategy: RinexStrategy = RinexStrategy.DCB_WITH_GIM_FALLBACK,
) -> tec_data.ElectronDensity:
    """
    Calculate electron density with optional time averaging.

    Parameters
    ----------
    ipp_target : IPP
        Target ionospheric pierce points
    gnss_data_list : list[GNSSData]
        GNSS observations from all stations
    sp3_data
        Satellite orbit data
    ionex : IonexData
        Global ionospheric map
    n_time_slots : int, optional
        Number of time slots to average
        1 = nearest neighbor (fast, default)
        5 = ±2 slots (better coverage, recommended)
        9 = ±4 slots (maximum coverage)
    max_time_diff_min : float, optional
        Maximum time difference for averaging (minutes), default 2.5
    use_time_weighting : bool, optional
        Weight measurements by temporal distance, default False

    Returns
    -------
    tec_data.ElectronDensity
        Electron density and uncertainties

    Notes
    -----
    Time averaging increases measurement density by combining observations
    from multiple time slots. This improves spatial coverage and reduces
    interpolation uncertainty, especially in sparse regions.

    Examples
    --------
    >>> # Nearest neighbor (fast, default)
    >>> density = get_ipp_density(ipp, gnss_data, sp3_data, ionex)
    >>>
    >>> # Time averaging (better coverage)
    >>> density = get_ipp_density(
    ...     ipp, gnss_data, sp3_data, ionex,
    ...     n_time_slots=5,
    ...     max_time_diff_min=2.5,
    ...     use_time_weighting=True
    ... )
    """
    profiles = get_profile(ipp_target)

    Ntimes = ipp_target.times.shape[0]
    Nheights = ipp_target.lon.shape[1]

    all_data = [[[] for _ in range(Nheights)] for _ in range(Ntimes)]

    # Process stations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_station = {
            executor.submit(
                get_gnss_station_density,
                gnss_data,
                ipp_target,
                profiles,
                sp3_data,
                ionex,
                dcb_data,
                n_time_slots,
                max_time_diff_min,
                use_time_weighting,
                strategy,
            ): gnss_data.station
            + gnss_data.constellation
            for gnss_data in gnss_data_list
        }

        for future in as_completed(future_to_station):
            station = future_to_station[future]
            try:
                result = future.result()

                # Validate and merge
                if not isinstance(result, list) or len(result) != Ntimes:
                    print(f"Error: {station} returned invalid structure")
                    continue

                for itm in range(Ntimes):
                    if (
                        not isinstance(result[itm], list)
                        or len(result[itm]) != Nheights
                    ):
                        continue

                    for hidx in range(Nheights):
                        measurement = result[itm][hidx]

                        if (
                            isinstance(measurement, np.ndarray)
                            and measurement.shape
                            and measurement.shape[0] > 0
                        ):
                            all_data[itm][hidx].append(measurement)

            except Exception as e:
                print(f"Error processing {station}: {e}")
                import traceback

                traceback.print_exc()

    # Concatenate measurements
    for itm in range(Ntimes):
        for hidx in range(Nheights):
            if all_data[itm][hidx]:
                try:
                    all_data[itm][hidx] = np.concatenate(all_data[itm][hidx], axis=0)
                except Exception as e:
                    print(f"Error concatenating all_data[{itm}][{hidx}]: {e}")
                    all_data[itm][hidx] = np.array([])
            else:
                all_data[itm][hidx] = np.array([])

    # Interpolate
    electron_density = get_interpolated_tec(all_data, use_time_weighting)

    del all_data, profiles
    gc.collect()

    return tec_data.ElectronDensity(
        electron_density=electron_density,
        electron_density_error=np.zeros_like(electron_density),
    )
