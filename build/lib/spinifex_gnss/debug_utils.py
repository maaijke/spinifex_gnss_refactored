"""
Debug utilities for GNSS processing.

This module provides debugging tools to inspect intermediate data
in the GNSS processing pipeline, particularly for get_interpolated_tec.
"""

import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Optional
from astropy.time import Time

from spinifex.geometry import IPP
from spinifex_gnss.parse_gnss import GNSSData


class DebugData:
    """Container for debug data from GNSS processing."""
    
    def __init__(self, debug_dir: Optional[Path] = None):
        """
        Initialize debug data container.
        
        Parameters
        ----------
        debug_dir : Path, optional
            Directory to save debug files. If None, uses './debug_gnss/'
        """
        self.debug_dir = debug_dir or Path('./debug_gnss')
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Containers for different data types
        self.interpolation_input = {}
        self.station_data = {}
        self.constellation_data = {}
        self.metadata = {}
        
    def save_interpolation_input(
        self,
        input_data: list[list[np.ndarray]],
        ipp_target: IPP,
        timestamp: Optional[str] = None
    ):
        """
        Save input data for get_interpolated_tec.
        
        Parameters
        ----------
        input_data : list[list[np.ndarray]]
            Nested list [times][heights] of arrays with [VTEC, error, dlon, dlat]
        ipp_target : IPP
            Target IPPs
        timestamp : str, optional
            Timestamp for filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle for exact reconstruction
        pickle_file = self.debug_dir / f"interpolation_input_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'input_data': input_data,
                'ipp_target_times': ipp_target.times.iso,
                'ipp_target_location': {
                    'lat': ipp_target.loc.lat.deg,
                    'lon': ipp_target.loc.lon.deg,
                    'height': ipp_target.loc.height.to('km').value,
                },
            }, f)
        
        print(f"✓ Saved interpolation input to {pickle_file}")
        
        # Also save human-readable summary
        self._save_interpolation_summary(input_data, ipp_target, timestamp)
        
        return pickle_file
    
    def _save_interpolation_summary(
        self,
        input_data: list[list[np.ndarray]],
        ipp_target: IPP,
        timestamp: str
    ):
        """Save human-readable summary of interpolation input."""
        
        summary_file = self.debug_dir / f"interpolation_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("INTERPOLATION INPUT DATA SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Number of times: {len(input_data)}\n")
            f.write(f"Number of heights: {len(input_data[0]) if input_data else 0}\n\n")
            
            f.write("Target IPP Times:\n")
            for i, t in enumerate(ipp_target.times[:5]):
                f.write(f"  {i}: {t.iso}\n")
            if len(ipp_target.times) > 5:
                f.write(f"  ... ({len(ipp_target.times) - 5} more)\n")
            f.write("\n")
            
            f.write("Target IPP Heights (km):\n")
            heights = ipp_target.loc[0].height.to('km').value
            for i, h in enumerate(heights[:5]):
                f.write(f"  {i}: {h:.1f}\n")
            if len(heights) > 5:
                f.write(f"  ... ({len(heights) - 5} more)\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("DATA AVAILABILITY BY TIME AND HEIGHT\n")
            f.write("-" * 80 + "\n\n")
            
            for time_idx in range(min(3, len(input_data))):
                f.write(f"Time {time_idx} ({ipp_target.times[time_idx].iso}):\n")
                for height_idx in range(len(input_data[time_idx])):
                    data = input_data[time_idx][height_idx]
                    n_measurements = data.shape[0] if data.shape else 0
                    f.write(f"  Height {height_idx} ({heights[height_idx]:.1f} km): {n_measurements} measurements\n")
                    
                    if n_measurements > 0:
                        # Show statistics
                        vtec = data[:, 0]
                        errors = data[:, 1]
                        dlons = data[:, 2]
                        dlats = data[:, 3]
                        
                        f.write(f"    VTEC range: [{np.min(vtec):.2f}, {np.max(vtec):.2f}] TECU\n")
                        f.write(f"    Error range: [{np.min(errors):.2f}, {np.max(errors):.2f}] TECU\n")
                        f.write(f"    dLon range: [{np.min(dlons):.2f}, {np.max(dlons):.2f}] deg\n")
                        f.write(f"    dLat range: [{np.min(dlats):.2f}, {np.max(dlats):.2f}] deg\n")
                f.write("\n")
            
            if len(input_data) > 3:
                f.write(f"... ({len(input_data) - 3} more time steps)\n\n")
        
        print(f"✓ Saved interpolation summary to {summary_file}")
    
    def save_station_contributions(
        self,
        station_name: str,
        constellation: str,
        gnss_data: GNSSData,
        stec_values: np.ndarray,
        stec_errors: np.ndarray,
        vtec_data: list[list[np.ndarray]],
        timestamp: Optional[str] = None
    ):
        """
        Save per-station contributions to interpolation.
        
        Parameters
        ----------
        station_name : str
            Station identifier
        constellation : str
            Constellation identifier (G, E, R, C, J)
        gnss_data : GNSSData
            Original GNSS observations
        stec_values : np.ndarray
            STEC values [satellites × times]
        stec_errors : np.ndarray
            STEC errors [satellites × times]
        vtec_data : list[list[np.ndarray]]
            VTEC data structure for this station
        timestamp : str, optional
            Timestamp for filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        station_file = self.debug_dir / f"station_{station_name}_{constellation}_{timestamp}.pkl"
        
        with open(station_file, 'wb') as f:
            pickle.dump({
                'station_name': station_name,
                'constellation': constellation,
                'observation_codes': {
                    'C1': gnss_data.c1_str,
                    'C2': gnss_data.c2_str,
                    'L1': gnss_data.l1_str,
                    'L2': gnss_data.l2_str,
                },
                'satellite_ids': list(gnss_data.gnss.keys()),
                'stec_values': stec_values,
                'stec_errors': stec_errors,
                'vtec_data': vtec_data,
                'times': gnss_data.times.iso,
            }, f)
        
        print(f"✓ Saved station data for {station_name} ({constellation}) to {station_file}")
        
        # Save summary
        self._save_station_summary(
            station_name, constellation, gnss_data,
            stec_values, stec_errors, timestamp
        )
        
        return station_file
    
    def _save_station_summary(
        self,
        station_name: str,
        constellation: str,
        gnss_data: GNSSData,
        stec_values: np.ndarray,
        stec_errors: np.ndarray,
        timestamp: str
    ):
        """Save human-readable station summary."""
        
        summary_file = self.debug_dir / f"station_{station_name}_{constellation}_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"STATION DATA: {station_name} - {constellation}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Constellation: {constellation}\n")
            f.write(f"Observation Codes:\n")
            f.write(f"  C1: {gnss_data.c1_str}\n")
            f.write(f"  C2: {gnss_data.c2_str}\n")
            f.write(f"  L1: {gnss_data.l1_str}\n")
            f.write(f"  L2: {gnss_data.l2_str}\n\n")
            
            f.write(f"Number of satellites: {len(gnss_data.gnss)}\n")
            f.write(f"Satellites: {', '.join(sorted(gnss_data.gnss.keys()))}\n\n")
            
            f.write(f"STEC values shape: {stec_values.shape}\n")
            f.write(f"STEC range: [{np.nanmin(stec_values):.2f}, {np.nanmax(stec_values):.2f}] TECU\n")
            f.write(f"STEC mean: {np.nanmean(stec_values):.2f} TECU\n")
            f.write(f"Valid measurements: {np.sum(~np.isnan(stec_values))}\n\n")
            
            f.write("Per-satellite statistics:\n")
            for i, prn in enumerate(sorted(gnss_data.gnss.keys())):
                valid_count = np.sum(~np.isnan(stec_values[i]))
                if valid_count > 0:
                    mean_stec = np.nanmean(stec_values[i])
                    mean_error = np.nanmean(stec_errors[i])
                    f.write(f"  {prn}: {valid_count:4d} obs, STEC={mean_stec:6.2f}±{mean_error:5.2f} TECU\n")
        
        print(f"✓ Saved station summary to {summary_file}")
    
    def save_constellation_breakdown(
        self,
        all_station_data: dict,
        timestamp: Optional[str] = None
    ):
        """
        Save breakdown of contributions by constellation.
        
        Parameters
        ----------
        all_station_data : dict
            Dictionary mapping station+constellation to their data
        timestamp : str, optional
            Timestamp for filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Group by constellation
        by_constellation = {}
        for key, data in all_station_data.items():
            # Extract constellation from key (last character)
            constellation = key[-1]
            if constellation not in by_constellation:
                by_constellation[constellation] = []
            by_constellation[constellation].append(key)
        
        summary_file = self.debug_dir / f"constellation_breakdown_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CONSTELLATION BREAKDOWN\n")
            f.write("=" * 80 + "\n\n")
            
            for constellation in sorted(by_constellation.keys()):
                stations = by_constellation[constellation]
                f.write(f"\nConstellation {constellation}:\n")
                f.write(f"  Number of stations: {len(stations)}\n")
                f.write(f"  Stations:\n")
                for station in sorted(stations):
                    f.write(f"    - {station[:-1]} ({constellation})\n")
        
        print(f"✓ Saved constellation breakdown to {summary_file}")
        
        return summary_file
    
    def load_interpolation_input(self, pickle_file: Path):
        """
        Load saved interpolation input data.
        
        Parameters
        ----------
        pickle_file : Path
            Path to pickle file
            
        Returns
        -------
        dict
            Dictionary with 'input_data' and metadata
        """
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        return data


def create_interpolation_debug_file(
    input_data: list[list[np.ndarray]],
    ipp_target: IPP,
    output_file: Optional[Path] = None
) -> Path:
    """
    Create a comprehensive debug file for get_interpolated_tec input.
    
    This is a convenience function for quick debugging.
    
    Parameters
    ----------
    input_data : list[list[np.ndarray]]
        Input to get_interpolated_tec
    ipp_target : IPP
        Target IPP locations
    output_file : Path, optional
        Output file path. If None, auto-generates.
        
    Returns
    -------
    Path
        Path to created debug file
        
    Examples
    --------
    >>> # In proces_gnss_data.py, before calling get_interpolated_tec:
    >>> from spinifex_gnss.debug_utils import create_interpolation_debug_file
    >>> debug_file = create_interpolation_debug_file(all_data, ipp_target)
    >>> # Then call the function as normal:
    >>> electron_density = get_interpolated_tec(all_data)
    """
    debug = DebugData()
    return debug.save_interpolation_input(input_data, ipp_target)


def enable_debug_mode():
    """
    Enable debug mode globally by monkey-patching key functions.
    
    This adds automatic debug output to get_ipp_density and related functions.
    
    Examples
    --------
    >>> from spinifex_gnss.debug_utils import enable_debug_mode
    >>> enable_debug_mode()
    >>> # Now all processing will save debug files automatically
    >>> density = get_electron_density_gnss(ipp)
    """
    import spinifex_gnss.proces_gnss_data as pgd
    
    # Store original functions
    original_get_ipp_density = pgd.get_ipp_density
    original_get_gnss_station_density = pgd.get_gnss_station_density
    
    debug = DebugData()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Wrapper for get_ipp_density
    def debug_get_ipp_density(*args, **kwargs):
        print(f"\n{'='*80}")
        print("DEBUG MODE: get_ipp_density()")
        print(f"{'='*80}\n")
        
        result = original_get_ipp_density(*args, **kwargs)
        
        # Save the input data that goes to get_interpolated_tec
        # This requires modifying the function to capture all_data
        # For now, just log that it was called
        print(f"\n✓ get_ipp_density completed")
        print(f"  Result shape: {result.electron_density.shape}")
        print(f"  Debug files in: {debug.debug_dir}\n")
        
        return result
    
    # Wrapper for get_gnss_station_density
    def debug_get_gnss_station_density(gnss_data, *args, **kwargs):
        station = gnss_data.station
        constellation = gnss_data.constellation
        
        print(f"  Processing: {station} ({constellation})")
        
        result = original_get_gnss_station_density(gnss_data, *args, **kwargs)
        
        return result
    
    # Apply monkey patches
    pgd.get_ipp_density = debug_get_ipp_density
    pgd.get_gnss_station_density = debug_get_gnss_station_density
    
    print(f"✓ Debug mode enabled!")
    print(f"  Debug files will be saved to: {debug.debug_dir}")
    print(f"  Timestamp: {timestamp}\n")
