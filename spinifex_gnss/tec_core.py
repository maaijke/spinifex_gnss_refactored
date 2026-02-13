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
    tec_coefficient: float = None
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
        C12, f1,f2 = tec_coefficient
        # For custom coefficient, we need to recalculate wavelengths
        # This is a limitation - we'd need f1/f2 passed in as well
        # For now, use constellation defaults for wavelengths
        WL1 = speed_light.value / f1
        WL2 = speed_light.value / f2
    else:
        C12 = get_tec_coefficient(constellation)
        WL1 = speed_light.value / FREQ[constellation]["f1"]
        WL2 = speed_light.value / FREQ[constellation]["f2"]
    
    return -C12 * (l1 * WL1 - l2 * WL2)


def _get_cycle_slips(
    phase_tec: np.ndarray,
    threshold_factor: float = CYCLE_SLIP_THRESHOLD,
    max_gap_points: int = 2
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
    slip_indices = _detect_slips(
        phase_tec_interp, threshold_factor, large_gap
    )
    
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
            if i > 0 and is_nan[i-1]:
                gap_lengths[i] = consecutive_nan
            consecutive_nan = 0
    
    return gap_lengths


def _interpolate_small_gaps(
    phase_tec: np.ndarray,
    is_nan: np.ndarray,
    gap_lengths: np.ndarray,
    max_gap_points: int
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
    phase_tec_interp: np.ndarray,
    threshold_factor: float,
    large_gap: np.ndarray
) -> np.ndarray:
    """Detect cycle slips using double-differencing."""
    diff1 = np.diff(phase_tec_interp, prepend=phase_tec_interp[0])
    #diff2 = np.diff(diff1, prepend=diff1[0])
    abs_diff2 = np.abs(diff1)
    
    median_diff2 = np.nanmedian(abs_diff2)
    threshold = threshold_factor * median_diff2
    
    cycle_slip_detected = (abs_diff2 > threshold) & ~large_gap
    
    return np.where(cycle_slip_detected)[0]



def _get_cycle_slips_old(
    phase_tec: np.ndarray, threshold_factor: float = CYCLE_SLIP_THRESHOLD, max_gap_points:int=2
) -> np.ndarray:
    """
    Detect cycle slips in carrier phase TEC.

    Parameters
    ----------
    phase_tec : np.ndarray
        STEC calculated from carrier phases
    threshold_factor : float, optional
        Threshold as multiple of median difference, by default 15.0

    Returns
    -------
    np.ndarray
        Segment IDs - continuous segments have the same ID
    """
    # Find valid (non-NaN) indices
    valid_mask = ~np.isnan(phase_tec)
    valid_indices = np.where(valid_mask)[0]
    valid_values = phase_tec[valid_mask]
    
    # Calculate differences only between valid points
    if len(valid_indices) < 3:
        # Not enough valid points for double-differencing
        return np.zeros(len(phase_tec), dtype=int)
    
    # First differences between consecutive valid points
    diff1_valid = np.diff(valid_values)
    
    # Second differences
    diff2_valid = np.abs(np.diff(diff1_valid))
    threshold = threshold_factor * np.median(diff2_valid)
    slips_in_valid = diff2_valid > threshold
    # Map back to full array
    # A cycle slip in diff2_valid[i] corresponds to valid_indices[i+2]
    cycle_slip_detected = np.zeros(len(phase_tec), dtype=bool)
    for i, slip in enumerate(slips_in_valid):
        if slip and i + 2 < len(valid_indices):
            cycle_slip_detected[valid_indices[i + 2]] = True    


    # Detect data gaps
    is_nan = np.isnan(phase_tec)   
    # Count consecutive NaNs
    gap_lengths = np.zeros(len(phase_tec), dtype=int)
    consecutive_nan = 0
    
    for i in range(len(phase_tec)):
        if is_nan[i]:
            consecutive_nan += 1
            gap_lengths[i] = consecutive_nan
        else:
            # Check if we just ended a gap
            if i > 0 and is_nan[i-1]:
                # Mark the position after gap with the gap length
                gap_lengths[i] = consecutive_nan
            consecutive_nan = 0
    
    # Start new segment only if gap is LARGER than max_gap_points
    large_gap = gap_lengths > max_gap_points
    
    # Combine: new segment starts on cycle slip OR large gap
    segment_breaks = cycle_slip_detected | large_gap
    
    # Create segment IDs
    return np.cumsum(segment_breaks.astype(int))
    



def _get_phase_corrected(
    phase_tec: np.ndarray, pseudo_tec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove phase bias using pseudorange TEC.

    Parameters
    ----------
    phase_tec : np.ndarray
        STEC from carrier phases (has bias)
    pseudo_tec : np.ndarray
        STEC from pseudoranges (no bias but noisy)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - Bias-corrected phase TEC
        - Standard deviation of bias estimate per segment
    """
    cycle_slips = _get_cycle_slips(phase_tec)
    phase_bias = np.zeros_like(phase_tec)
    phase_std = np.zeros_like(phase_tec)

    for seg in np.unique(cycle_slips):
        seg_idx = np.nonzero(cycle_slips == seg)[0]
        bias = np.nanmean(pseudo_tec[seg_idx] - phase_tec[seg_idx])
        std = np.nanstd(pseudo_tec[seg_idx] - phase_tec[seg_idx])
        phase_bias[seg_idx] = bias
        phase_std[seg_idx] = std

    return phase_tec + phase_bias, phase_std


# Note: getpseudorange_tec has been REMOVED - it's not used anywhere in the codebase
# We only use carrier phase TEC corrected with pseudorange (via _get_phase_corrected)
