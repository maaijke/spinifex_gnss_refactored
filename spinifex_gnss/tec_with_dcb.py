"""
TEC calculation with pseudorange and DCB corrections.

This module extends tec_core.py to include:
1. Pseudorange TEC calculation with DCB corrections
2. Phase wrap estimation from pseudorange-phase difference
3. Fallback to GIM when DCBs unavailable
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from typing import Optional, Tuple, Dict

from spinifex_gnss.config import FREQ
from spinifex_gnss.tec_core import _get_cycle_slips


def calculate_pseudorange_tec(
    c1: np.ndarray,
    c2: np.ndarray,
    f1_hz: float,
    f2_hz: float,
    satellite_dcb_ns: Optional[float] = None,
    receiver_dcb_ns: Optional[float] = None
) -> Tuple[np.ndarray, bool]:
    """
    Calculate TEC from pseudorange with DCB corrections.
    
    Formula:
    TEC_code = (f1² × f2²) / (40.3 × (f2² - f1²)) × (P2 - P1 - DCB_sat - DCB_rec)
    
    Parameters
    ----------
    c1, c2 : np.ndarray
        Pseudorange observations in meters
    f1_hz, f2_hz : float
        Frequencies in Hz
    satellite_dcb_ns : float, optional
        Satellite DCB in nanoseconds
    receiver_dcb_ns : float, optional
        Receiver DCB in nanoseconds
        
    Returns
    -------
    tec_tecu : np.ndarray
        TEC in TECU
    has_dcb : bool
        Whether DCB corrections were applied
    """
    # Speed of light
    c_light = 299792458.0  # m/s
    
    # Range difference
    delta_p = c2 - c1  # meters
    
    # Apply DCB corrections if available
    has_dcb = False
    if satellite_dcb_ns is not None:
        delta_p -= satellite_dcb_ns * 1e-9 * c_light
        has_dcb = True
    
    if receiver_dcb_ns is not None:
        delta_p -= receiver_dcb_ns * 1e-9 * c_light
        has_dcb = True
    
    # TEC coefficient
    coeff = (f1_hz**2 * f2_hz**2) / (40.3 * (f2_hz**2 - f1_hz**2))
    
    # TEC in electrons/m²
    tec_m2 = coeff * delta_p
    
    # Convert to TECU
    tec_tecu = tec_m2 / 1e16
    
    return tec_tecu, has_dcb


def estimate_phase_wraps(
    phase_tec: np.ndarray,
    pseudo_tec: np.ndarray,
    cycle_slips: Optional[np.ndarray] = None,
    min_points: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate phase wraps from pseudorange-phase difference.
    
    For each continuous segment (no cycle slips):
    - Calculate mean difference: bias = <pseudo_tec - phase_tec>
    - Add bias to phase_tec to align with pseudorange
    
    Parameters
    ----------
    phase_tec : np.ndarray
        Phase TEC (has arbitrary bias per segment)
    pseudo_tec : np.ndarray
        Pseudorange TEC (absolute, but noisy)
    cycle_slips : np.ndarray, optional
        Segment IDs from _get_cycle_slips
        If None, automatically detected
    min_points : int
        Minimum points in segment to estimate bias
        
    Returns
    -------
    corrected_tec : np.ndarray
        Phase TEC with wraps corrected
    bias_error : np.ndarray
        Error estimate for each segment (std of difference)
    """
    # Detect cycle slips if not provided
    if cycle_slips is None:
        cycle_slips = _get_cycle_slips(phase_tec)
    
    corrected_tec = phase_tec.copy()
    bias_error = np.zeros_like(phase_tec)
    
    # Process each segment
    for seg_id in np.unique(cycle_slips):
        seg_mask = cycle_slips == seg_id
        valid_mask = ~np.isnan(phase_tec) & ~np.isnan(pseudo_tec) & seg_mask
        
        if np.sum(valid_mask) < min_points:
            # Not enough points, mark as invalid
            corrected_tec[seg_mask] = np.nan
            bias_error[seg_mask] = np.nan
            continue
        
        # Calculate bias for this segment
        diff = pseudo_tec[valid_mask] - phase_tec[valid_mask]
        bias = np.nanmean(diff)
        std = np.nanstd(diff)
        
        # Apply correction
        corrected_tec[seg_mask] = phase_tec[seg_mask] + bias
        bias_error[seg_mask] = std
    
    return corrected_tec, bias_error


def calculate_stec_with_dcb(
    c1: np.ndarray,
    c2: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    constellation: str,
    satellite_id: str,
    satellite_dcb: Dict[str, float],
    receiver_dcb: Dict[str, float],
    station: str,
    obs_codes: Tuple[str, str, str, str],  # (C1, C2, L1, L2)
    tec_coefficient: Optional[Tuple[float, float, float]] = None,
    gim_stec: Optional[np.ndarray] = None,
    elevation: Optional[np.ndarray] = None,
    elevation_threshold: float = 60.0
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Calculate STEC with optimal method based on available data.
    
    Priority:
    1. If DCB available: Use pseudorange TEC + phase wraps
    2. If no DCB but GIM available: Use GIM bias correction
    3. Otherwise: Flag as unavailable
    
    Parameters
    ----------
    c1, c2 : np.ndarray
        Pseudorange observations
    l1, l2 : np.ndarray
        Phase observations
    constellation : str
        Constellation ID
    satellite_id : str
        Satellite PRN (e.g., 'G01')
    satellite_dcb : Dict[str, float]
        Satellite DCBs from parse_dcb_file
    receiver_dcb : Dict[str, float]
        Receiver DCBs from parse_dcb_file
    station : str
        Station name
    obs_codes : tuple
        (C1_code, C2_code, L1_code, L2_code) e.g., ('C1W', 'C2W', 'L1W', 'L2W')
    tec_coefficient : tuple, optional
        (C12, f1, f2) for GLONASS
    gim_stec : np.ndarray, optional
        GIM STEC for fallback bias estimation
    elevation : np.ndarray, optional
        Elevation angles for GIM method
    elevation_threshold : float
        Elevation threshold for GIM bias
        
    Returns
    -------
    stec : np.ndarray
        Corrected STEC in TECU
    stec_error : np.ndarray
        Error estimate in TECU
    method : str
        'dcb', 'gim', or 'flagged'
    """
    from spinifex_gnss.tec_core import getphase_tec
    from spinifex_gnss.parse_dcb import get_receiver_dcb_c1c2, convert_dcb_to_tec
    
    # Get frequencies
    if tec_coefficient is not None:
        C12, f1, f2 = tec_coefficient
    else:
        f1 = FREQ[constellation]['f1']
        f2 = FREQ[constellation]['f2']
    
    # Calculate phase TEC
    phase_tec = getphase_tec(l1, l2, constellation, tec_coefficient)
    
    # Try DCB method first
    sat_dcb_ns = satellite_dcb.get(satellite_id)
    c1_code, c2_code, l1_code, l2_code = obs_codes
    rec_dcb_ns = get_receiver_dcb_c1c2(receiver_dcb, station, c1_code, c2_code)
    
    if sat_dcb_ns is not None or rec_dcb_ns is not None:
        # Calculate pseudorange TEC with DCB
        pseudo_tec, has_dcb = calculate_pseudorange_tec(
            c1, c2, f1, f2, sat_dcb_ns, rec_dcb_ns
        )
        
        # Estimate phase wraps
        corrected_tec, bias_error = estimate_phase_wraps(phase_tec, pseudo_tec)
        
        return corrected_tec, bias_error, 'dcb'
    
    # Fall back to GIM method
    elif gim_stec is not None and elevation is not None:
        # Estimate bias from GIM at high elevation
        high_el = elevation > elevation_threshold
        valid = ~np.isnan(phase_tec) & ~np.isnan(gim_stec) & high_el
        
        if np.sum(valid) >= 5:
            bias = np.nanmean(gim_stec[valid] - phase_tec[valid])
            bias_error = np.nanstd(gim_stec[valid] - phase_tec[valid])
            
            corrected_tec = phase_tec + bias
            error = np.full_like(corrected_tec, bias_error)
            
            return corrected_tec, error, 'gim'
    
    # No method available
    return (
        np.full_like(phase_tec, np.nan),
        np.full_like(phase_tec, np.nan),
        'flagged'
    )

