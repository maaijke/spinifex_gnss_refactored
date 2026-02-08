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

from spinifex_gnss.config import FREQ, get_tec_coefficient


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


def getphase_tec(l1: np.ndarray, l2: np.ndarray, constellation: str = "G") -> np.ndarray:
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
        
    Returns
    -------
    np.ndarray
        Slant TEC values (TECU) with arbitrary bias
    """
    C12 = get_tec_coefficient(constellation)
    WL1 = speed_light.value / FREQ[constellation]["f1"]
    WL2 = speed_light.value / FREQ[constellation]["f2"]
    return -C12 * (l1 * WL1 - l2 * WL2)


def _get_cycle_slips(phase_tec: np.ndarray, threshold_factor: float = 5.0) -> np.ndarray:
    """
    Detect cycle slips in carrier phase TEC.
    
    Parameters
    ----------
    phase_tec : np.ndarray
        STEC calculated from carrier phases
    threshold_factor : float, optional
        Threshold as multiple of median difference, by default 5.0
        
    Returns
    -------
    np.ndarray
        Segment IDs - continuous segments have the same ID
    """
    diff = np.abs(np.diff(phase_tec, prepend=phase_tec[0]))
    threshold = threshold_factor * np.nanmedian(diff)
    slips = diff > threshold
    gaps = np.isnan(phase_tec)
    gap_starts = np.diff(gaps, prepend=0) > 0
    segment_breaks = np.logical_or(slips, gap_starts)
    return np.cumsum(segment_breaks.astype(int))


def _get_phase_corrected(
    phase_tec: np.ndarray,
    pseudo_tec: np.ndarray
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
