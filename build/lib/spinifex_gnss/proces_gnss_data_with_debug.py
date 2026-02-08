"""
GNSS data processing with optional debug mode.

Add DEBUG_MODE parameter to save intermediate data for algorithm testing.
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import Optional
import pickle
from datetime import datetime

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

# Global debug settings
DEBUG_MODE = False
DEBUG_DIR = Path('./debug_gnss')


def enable_debug_mode(debug_dir: Optional[Path] = None):
    """
    Enable debug mode to save intermediate data.
    
    Parameters
    ----------
    debug_dir : Path, optional
        Directory for debug files, default './debug_gnss'
        
    Examples
    --------
    >>> from spinifex_gnss.proces_gnss_data import enable_debug_mode
    >>> enable_debug_mode()
    >>> # Now all processing saves debug data
    """
    global DEBUG_MODE, DEBUG_DIR
    DEBUG_MODE = True
    if debug_dir:
        DEBUG_DIR = debug_dir
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Debug mode enabled! Files will be saved to: {DEBUG_DIR}")


def disable_debug_mode():
    """Disable debug mode."""
    global DEBUG_MODE
    DEBUG_MODE = False
    print("✓ Debug mode disabled")


def _save_debug_data(data: dict, filename: str):
    """Save debug data to pickle file."""
    if not DEBUG_MODE:
        return
    
    filepath = DEBUG_DIR / filename
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"  [DEBUG] Saved: {filepath}")


def _save_debug_summary(text: str, filename: str):
    """Save debug summary to text file."""
    if not DEBUG_MODE:
        return
    
    filepath = DEBUG_DIR / filename
    with open(filepath, 'w') as f:
        f.write(text)
    print(f"  [DEBUG] Saved: {filepath}")


# [Rest of the functions remain the same as before - just add debug calls]

def _get_distance_km(loc1: EarthLocation, loc2: EarthLocation) -> np.ndarray:
    """Calculate distance between two sets of locations."""
    dx = loc1.x.value - loc2.x.value
    dy = loc1.y.value - loc2.y.value
    dz = loc1.z.value - loc2.z.value
    return np.sqrt(dx**2 + dy**2 + dz**2) / 1000.0


def _get_gim_phase_corrected(
    phase_tec: np.ndarray,
    ipp_sat_stat: IPP,
    timeselect: np.ndarray,
    ionex: IonexData
) -> tuple[np.ndarray, np.ndarray]:
    """Correct carrier phase TEC using GIM."""
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
    
    for seg in np.unique(cycle_slips):
        seg_idx = np.nonzero(cycle_slips == seg)[0]
        
        if seg_idx.shape[0] < 2:
            phase_bias[seg_idx] = np.nan
            continue
            
        if np.intersect1d(seg_idx, timeselect).size == 0:
            continue
            
        ipp = ipp_sat_stat.loc[:, h_idx][seg_idx]
        elevation = ipp_sat_stat.altaz.alt.deg[seg_idx]
        
        gim_tec = interpolate_ionex(
            ionex, ipp.lon.deg, ipp.lat.deg,
            ipp_sat_stat.times[seg_idx],
            apply_earth_rotation=default_options.apply_earth_rotation,
        )
        
        high_el_mask = elevation > ELEVATION_CUT
        
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


def _get_distance_ipp(
    stec_values: np.ndarray,
    stec_errors: np.ndarray,
    ipp_sat_stat: list[IPP],
    ipp_target: IPP,
    timeselect: np.ndarray,
    profiles: np.ndarray,
) -> list[list[np.ndarray]]:
    """Calculate VTEC and distance to target IPPs."""
    Ntimes = ipp_target.times.shape[0]
    Nheights = ipp_target.loc[0].shape[0]
    Nprns = stec_values.shape[0]
    
    vtecs = np.full((Nprns, Ntimes, Nheights), np.nan, dtype=float)
    vtec_errors = np.full((Nprns, Ntimes, Nheights), np.nan, dtype=float)
    
    el_select = np.array([
        ipp.altaz.alt.deg[timeselect] > ELEVATION_CUT
        for ipp in ipp_sat_stat
    ])
    
    el_select = np.logical_and(~np.isnan(stec_values[:, timeselect]), el_select)
    
    dist_select = np.array([
        _get_distance_km(ipp.loc[timeselect], ipp_target.loc) < DISTANCE_KM_CUT
        for ipp in ipp_sat_stat
    ])
    
    prn_select = np.logical_and(el_select[:, :, np.newaxis], dist_select)
    
    weighted_am = np.array([
        profiles * ipp.airmass[timeselect]
        for ipp in ipp_sat_stat
    ])
    weighted_am = np.sum(weighted_am, axis=-1)
    
    vtec_values = profiles * (stec_values[:, timeselect] / weighted_am)[..., np.newaxis]
    vtec_error_values = profiles * stec_errors[:, timeselect][..., np.newaxis]
    
    dlons = np.array([
        np.cos(ipp_target.loc.lat.rad) * (ipp.loc.lon.deg[timeselect] - ipp_target.loc.lon.deg)
        for ipp in ipp_sat_stat
    ])
    
    dlats = np.array([
        ipp.loc.lat.deg[timeselect] - ipp_target.loc.lat.deg
        for ipp in ipp_sat_stat
    ])
    
    vtecs[prn_select] = vtec_values[prn_select]
    vtec_errors[prn_select] = vtec_error_values[prn_select]
    
    return [
        [
            np.concatenate((
                vtecs[:, timeidx, hidx][~np.isnan(vtecs[:, timeidx, hidx])][:, np.newaxis],
                vtec_errors[:, timeidx, hidx][~np.isnan(vtecs[:, timeidx, hidx])][:, np.newaxis],
                dlons[:, timeidx, hidx][~np.isnan(vtecs[:, timeidx, hidx])][:, np.newaxis],
                dlats[:, timeidx, hidx][~np.isnan(vtecs[:, timeidx, hidx])][:, np.newaxis],
            ), axis=-1)
            for hidx in range(Nheights)
        ]
        for timeidx in range(Ntimes)
    ]


def get_interpolated_tec(input_data: list[list[np.ndarray]]) -> np.ndarray:
    """Interpolate VTEC to target location."""
    fitted_density = np.zeros((len(input_data), len(input_data[0])))
    
    for timeidx, input_time in enumerate(input_data):
        for hidx, vtec_dlong_dlat in enumerate(input_time):
            if not vtec_dlong_dlat.shape or vtec_dlong_dlat.shape[0] < 2:
                continue
            
            dist = np.linalg.norm(vtec_dlong_dlat[:, 2:], axis=1)
            dist_select = np.zeros(dist.shape, dtype=bool)
            nearest_indices = np.argpartition(
                dist, min(NDIST_POINTS, dist.shape[0] - 1), axis=0
            )[:NDIST_POINTS]
            dist_select[nearest_indices] = True
            
            A = np.ones(
                vtec_dlong_dlat[dist_select].shape[:1] +
                (((INTERPOLATION_ORDER) ** 2 + INTERPOLATION_ORDER) // 2,),
                dtype=float,
            )
            
            weight = 1.0 / vtec_dlong_dlat[dist_select][:, 1]
            
            idx = 0
            for ilon in range(INTERPOLATION_ORDER):
                for ilat in range(INTERPOLATION_ORDER):
                    if ilon + ilat <= INTERPOLATION_ORDER - 1:
                        if idx > 0:
                            A[:, idx] = (
                                vtec_dlong_dlat[dist_select][:, 2] ** ilon *
                                vtec_dlong_dlat[dist_select][:, 3] ** ilat
                            )
                        idx += 1
            
            w = weight * np.eye(A.shape[0])
            AwT = A.T @ w
            
            try:
                par = (np.linalg.inv(AwT @ A) @ (AwT @ vtec_dlong_dlat[dist_select][:, :1])).squeeze()
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
) -> list[list[np.ndarray]]:
    """Process one GNSS station."""
    prns = sorted(gnss_data.gnss.keys())
    stec_values = []
    stec_errors = []
    ipp_sat_stat = []
    
    gpstime_correction = GPS_TO_UTC_CORRECTION_DAYS
    
    timeselect = np.argmin(
        np.abs(
            ipp_target.times.mjd - gnss_data.times.mjd[:, np.newaxis] + gpstime_correction
        ),
        axis=0,
    )
    
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
            stec_value, stec_error = _get_gim_phase_corrected(
                phase_stec, ipp_sat_stat[-1], timeselect, ionex
            )
            stec_values.append(stec_value)
            stec_errors.append(stec_error)
        except Exception as e:
            print(f"Failed for {gnss_data.station} {prn}: {e}")
    
    result = _get_distance_ipp(
        stec_values=np.array(stec_values),
        stec_errors=np.array(stec_errors),
        ipp_sat_stat=ipp_sat_stat,
        ipp_target=ipp_target,
        timeselect=timeselect,
        profiles=profiles,
    )
    
    # Save debug data if enabled
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_debug_data({
            'station': gnss_data.station,
            'constellation': gnss_data.constellation,
            'satellites': prns,
            'stec_values': np.array(stec_values),
            'stec_errors': np.array(stec_errors),
            'vtec_data': result,
            'obs_codes': {
                'C1': gnss_data.c1_str,
                'C2': gnss_data.c2_str,
                'L1': gnss_data.l1_str,
                'L2': gnss_data.l2_str,
            }
        }, f"station_{gnss_data.station}_{gnss_data.constellation}_{timestamp}.pkl")
    
    del gnss_data, stec_values, ipp_sat_stat
    return result


def get_ipp_density(
    ipp_target: IPP,
    gnss_data_list: list[GNSSData],
    sp3_data,
    ionex: IonexData,
) -> tec_data.ElectronDensity:
    """Calculate electron density at target IPPs."""
    profiles = get_profile(ipp_target)
    
    Ntimes = ipp_target.times.shape[0]
    Nheights = ipp_target.loc.shape[1]
    
    all_data = [[[] for _ in range(Nheights)] for _ in range(Ntimes)]
    stec_gnss_data = {}
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_DENSITY) as executor:
        future_to_station_constellation = {
            executor.submit(
                get_gnss_station_density,
                gnss_data, ipp_target, profiles, sp3_data, ionex,
            ): gnss_data.station + gnss_data.constellation
            for gnss_data in gnss_data_list
        }
        
        for future in as_completed(future_to_station_constellation):
            station = future_to_station_constellation[future]
            try:
                result = future.result()
                stec_gnss_data[station] = result
            except Exception as e:
                print(f"Error processing {station}: {e}")
                stec_gnss_data[station] = f"Error: {e}"
    
    for station, station_data in stec_gnss_data.items():
        if isinstance(station_data, str):
            print(f"Skipping {station}: {station_data}")
            continue
            
        for itm in range(Ntimes):
            for hidx in range(Nheights):
                all_data[itm][hidx].append(station_data[itm][hidx])
    
    for itm in range(Ntimes):
        for hidx in range(Nheights):
            if all_data[itm][hidx]:
                all_data[itm][hidx] = np.concatenate(all_data[itm][hidx], axis=0)
            else:
                all_data[itm][hidx] = np.array([])
    
    # Save debug data BEFORE interpolation
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save interpolation input
        _save_debug_data({
            'input_data': all_data,
            'ipp_times': ipp_target.times.iso,
            'ipp_location': {
                'lat': ipp_target.loc.lat.deg,
                'lon': ipp_target.loc.lon.deg,
                'height_km': ipp_target.loc.height.to('km').value,
            },
            'station_list': list(stec_gnss_data.keys()),
        }, f"interpolation_input_{timestamp}.pkl")
        
        # Save human-readable summary
        summary = _generate_debug_summary(all_data, ipp_target, stec_gnss_data)
        _save_debug_summary(summary, f"interpolation_summary_{timestamp}.txt")
    
    electron_density = get_interpolated_tec(all_data)
    
    del stec_gnss_data, all_data
    
    return tec_data.ElectronDensity(
        electron_density=electron_density,
        electron_density_error=np.zeros_like(electron_density),
    )


def _generate_debug_summary(
    all_data: list[list[np.ndarray]],
    ipp_target: IPP,
    station_data: dict
) -> str:
    """Generate human-readable debug summary."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("INTERPOLATION INPUT DATA SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Group by constellation
    by_constellation = {}
    for station_key in station_data.keys():
        if not isinstance(station_data[station_key], str):  # Skip errors
            const = station_key[-1]  # Last character is constellation
            if const not in by_constellation:
                by_constellation[const] = []
            by_constellation[const].append(station_key)
    
    lines.append(f"Constellations present: {', '.join(sorted(by_constellation.keys()))}")
    for const in sorted(by_constellation.keys()):
        lines.append(f"  {const}: {len(by_constellation[const])} stations")
    lines.append("")
    
    lines.append(f"Number of times: {len(all_data)}")
    lines.append(f"Number of heights: {len(all_data[0]) if all_data else 0}")
    lines.append("")
    
    lines.append("Target IPP Times (first 5):")
    for i in range(min(5, len(ipp_target.times))):
        lines.append(f"  {i}: {ipp_target.times[i].iso}")
    if len(ipp_target.times) > 5:
        lines.append(f"  ... and {len(ipp_target.times) - 5} more")
    lines.append("")
    
    lines.append("Target IPP Heights (km):")
    heights = ipp_target.loc[0].height.to('km').value
    for i in range(min(5, len(heights))):
        lines.append(f"  {i}: {heights[i]:.1f}")
    if len(heights) > 5:
        lines.append(f"  ... and {len(heights) - 5} more")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("DATA AVAILABILITY (first 3 times)")
    lines.append("-" * 80)
    
    for tidx in range(min(3, len(all_data))):
        lines.append(f"\nTime {tidx} ({ipp_target.times[tidx].iso}):")
        for hidx in range(len(all_data[tidx])):
            data = all_data[tidx][hidx]
            n_meas = data.shape[0] if data.shape else 0
            lines.append(f"  Height {hidx} ({heights[hidx]:.1f} km): {n_meas} measurements")
            
            if n_meas > 0:
                vtec = data[:, 0]
                lines.append(f"    VTEC: [{np.min(vtec):.2f}, {np.max(vtec):.2f}] TECU (mean={np.mean(vtec):.2f})")
    
    if len(all_data) > 3:
        lines.append(f"\n... and {len(all_data) - 3} more time steps")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("CONSTELLATION BREAKDOWN")
    lines.append("=" * 80)
    
    for const in sorted(by_constellation.keys()):
        lines.append(f"\n{const} - GPS" if const == 'G' else f"\n{const}")
        for station_key in sorted(by_constellation[const]):
            lines.append(f"  {station_key}")
    
    return "\n".join(lines)
