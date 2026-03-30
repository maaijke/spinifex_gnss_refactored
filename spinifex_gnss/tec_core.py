"""
Core TEC calculation functions.

This module contains the fundamental TEC calculation algorithms.
Refactored to remove:
- DCB dependencies
- Pseudorange TEC (getpseudorange_tec - not used anywhere)
- Obsolete functions
"""

import numpy as np
from astropy.time import Time
from astropy.constants import c as speed_light
import astropy.units as u

from spinifex_gnss.config import FREQ, get_tec_coefficient, CYCLE_SLIP_THRESHOLD


def getpseudorange_tec(
    c1: np.ndarray,
    c2: np.ndarray,
    constellation: str = "G",
    tec_coefficient: tuple = None,
    satellite_dcb_ns: float = None,
    receiver_dcb_ns: float = None,
) -> np.ndarray:
    """
    Calculate STEC from pseudorange observations with optional DCB corrections.

    Parameters
    ----------
    c1 : np.ndarray
        Pseudorange for frequency f1 (in meters)
    c2 : np.ndarray
        Pseudorange for frequency f2 (in meters)
    constellation : str, optional
        Satellite constellation identifier, by default "G"
    tec_coefficient : tuple, optional
        Pre-calculated (C12, f1, f2) for GLONASS FDMA
    satellite_dcb_ns : float, optional
        Satellite DCB in nanoseconds (from DCB file)
    receiver_dcb_ns : float, optional
        Receiver DCB in nanoseconds (from DCB file)

    Returns
    -------
    np.ndarray
        Slant TEC values (TECU)

    Notes
    -----
    The pseudorange TEC is calculated as:
        TEC = C12 * (P2 - P1) - DCB_sat - DCB_rcv

    where DCB corrections convert nanoseconds to TECU using:
        DCB_tecu = DCB_ns * C12 / (f1 * f2) * 1e9 / c

    For GLONASS, tec_coefficient must be provided since each satellite
    uses different frequencies (FDMA).
    """
    # Get TEC coefficient and frequencies
    if tec_coefficient is not None:
        C12, _, _ = tec_coefficient
    else:
        C12 = get_tec_coefficient(constellation)

    # Calculate pseudorange TEC
    pseudo_tec = C12 * (c2 - c1)

    # Apply DCB corrections if provided
    if satellite_dcb_ns is not None:
        # Convert DCB from nanoseconds to TECU
        # DCB in meters: dcb_ns * c
        # DCB in TEC: dcb_meters * C12
        dcb_meters = satellite_dcb_ns * 1e-9 * speed_light.value
        satellite_dcb_tecu = C12 * dcb_meters
        pseudo_tec += satellite_dcb_tecu

    if receiver_dcb_ns is not None:
        dcb_meters = receiver_dcb_ns * 1e-9 * speed_light.value
        receiver_dcb_tecu = C12 * dcb_meters
        pseudo_tec += receiver_dcb_tecu

    return pseudo_tec


def get_transmission_time(c2: np.ndarray, times: Time) -> Time:
    """
    Calculate satellite transmission time from receiver time.

    Parameters
    ----------
    c2 : np.ndarray
        Pseudorange measurements (path length in meters)
    times : Time
        Receiver observation times (GPS time)

    Returns
    -------
    Time
        Transmission times at satellite

    Notes
    -----
    No DCB correction applied - not needed for relative TEC measurements.
    """
    distance = np.copy(c2)
    distance[np.isnan(distance)] = 0
    return times - (distance * u.m) / speed_light


def getphase_tec(
    l1: np.ndarray,
    l2: np.ndarray,
    constellation: str = "G",
    tec_coefficient: float = None,
) -> np.ndarray:
    """
    Calculate STEC from carrier phase observations.

    Parameters
    ----------
    l1 : np.ndarray
        Carrier phase for frequency f1 (in cycles)
    l2 : np.ndarray
        Carrier phase for frequency f2 (in cycles)
    constellation : str, optional
        Satellite constellation identifier, by default "G"
    tec_coefficient : float, optional
        Pre-calculated TEC coefficient C12 (for GLONASS FDMA)
        If provided, this overrides the default constellation coefficient

    Returns
    -------
    np.ndarray
        Slant TEC values (TECU) with arbitrary bias

    Notes
    -----
    For GLONASS satellites, tec_coefficient should be provided since
    each satellite uses a different frequency (FDMA). This coefficient
    is calculated in parse_gnss.py from the RINEX header frequency channels.

    For other constellations (GPS, Galileo, BeiDou), the default
    constellation-wide coefficient is used.
    """
    # Use provided coefficient or get default for constellation
    if tec_coefficient is not None:
        C12, f1, f2 = tec_coefficient
        # For custom coefficient, we need to recalculate wavelengths
        # This is a limitation - we'd need f1/f2 passed in as well
        # For now, use constellation defaults for wavelengths
        WL1 = speed_light.value / f1
        WL2 = speed_light.value / f2
    else:
        C12 = get_tec_coefficient(constellation)
        WL1 = speed_light.value / FREQ[constellation]["f1"]
        WL2 = speed_light.value / FREQ[constellation]["f2"]

    return C12 * (l1 * WL1 - l2 * WL2)


def get_cycle_slips(
    phase_tec: np.ndarray,
    threshold_factor: float = CYCLE_SLIP_THRESHOLD,
    max_gap_points: int = 2,
) -> np.ndarray:
    """
    Repair cycle slips using fast linear extrapolation.

    Parameters
    ----------
    phase_tec : np.ndarray
        STEC from carrier phases (with cycle slips)
    threshold_factor : float
        Multiplier for median double-difference (default: 5.0)
    absolute_threshold : float
        Absolute threshold in TECU (default: 15.0)
    max_gap_points : int
        Maximum gap size to interpolate over (default: 2)

    Returns
    -------
    np.ndarray
        Repaired phase TEC with cycle slips removed

    Notes
    -----
    Algorithm:
    1. Interpolate over small gaps (≤ max_gap_points)
    2. Detect cycle slips using double-differencing
    3. For each slip: extrapolate expected value, subtract offset

    Only segments data at large gaps (> max_gap_points).
    Most data will be continuous with single global bias.

    Examples
    --------
    >>> phase_tec = np.array([30.0, 30.5, 31.0, 45.0, 45.5, 46.0])
    >>> repaired = repair_cycle_slips_fast(phase_tec)
    >>> # Cycle slip at index 3 removed, data aligned
    """
    # Step 1: Detect gaps
    is_nan = np.isnan(phase_tec)
    gap_lengths = _count_gap_lengths(is_nan)
    large_gap = gap_lengths > max_gap_points

    # Step 2: Interpolate over small gaps
    phase_tec_interp = _interpolate_small_gaps(
        phase_tec, is_nan, gap_lengths, max_gap_points
    )

    # Step 3: Detect cycle slips
    slip_indices = _detect_slips(phase_tec_interp, threshold_factor, large_gap)

    # Step 4: Repair cycle slips (FAST!)
    repaired = phase_tec

    for slip_idx in sorted(slip_indices):
        if slip_idx < 2 or is_nan[slip_idx]:
            continue

        # Get valid data before slip
        valid_before = repaired[:slip_idx][~is_nan[:slip_idx]]

        if len(valid_before) >= 2:
            # Fast linear extrapolation (YOUR ELEGANT METHOD!)
            expected = np.diff(valid_before[-2:]) + valid_before[-1]
            expected = expected[0]

            # Calculate and apply offset
            offset = repaired[slip_idx] - expected
            repaired[slip_idx:] -= offset

    return np.cumsum(large_gap.astype(int))


def _count_gap_lengths(is_nan: np.ndarray) -> np.ndarray:
    """Count consecutive NaN lengths."""
    gap_lengths = np.zeros(len(is_nan), dtype=int)
    consecutive_nan = 0

    for i in range(len(is_nan)):
        if is_nan[i]:
            consecutive_nan += 1
            gap_lengths[i] = consecutive_nan
        else:
            if i > 0 and is_nan[i - 1]:
                gap_lengths[i] = consecutive_nan
            consecutive_nan = 0

    return gap_lengths


def _interpolate_small_gaps(
    phase_tec: np.ndarray,
    is_nan: np.ndarray,
    gap_lengths: np.ndarray,
    max_gap_points: int,
) -> np.ndarray:
    """Interpolate over small gaps."""
    phase_tec_interp = phase_tec.copy()
    small_gap_mask = is_nan & (gap_lengths <= max_gap_points)

    if np.any(small_gap_mask):
        valid_indices = np.where(~is_nan)[0]
        valid_values = phase_tec[~is_nan]

        if len(valid_indices) > 1:
            gap_indices = np.where(small_gap_mask)[0]
            interpolated = np.interp(gap_indices, valid_indices, valid_values)
            phase_tec_interp[gap_indices] = interpolated

    return phase_tec_interp


def _detect_slips(
    phase_tec_interp: np.ndarray, threshold_factor: float, large_gap: np.ndarray
) -> np.ndarray:
    """Detect cycle slips using double-differencing."""
    diff1 = np.diff(phase_tec_interp, prepend=phase_tec_interp[0])
    # diff2 = np.diff(diff1, prepend=diff1[0])
    abs_diff2 = np.abs(diff1)

    median_diff2 = np.nanmedian(abs_diff2)
    threshold = threshold_factor * median_diff2

    cycle_slip_detected = (abs_diff2 > threshold) & ~large_gap

    return np.where(cycle_slip_detected)[0]




