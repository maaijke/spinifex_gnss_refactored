"""
GNSS station position data loading.

This module loads and provides access to GNSS station positions worldwide.
"""

from astropy.coordinates import EarthLocation
import astropy.units as u
from pathlib import Path
from importlib import resources
import sys


def load_gnss_station_positions() -> dict[str, EarthLocation]:
    """
    Load GNSS station positions from data file.
    
    Tries multiple locations in order:
    1. Package data directory (spinifex_gnss.data)
    2. Package directory (spinifex_gnss/data/)
    3. Parent data directory (../data/)
    4. Uploaded data location
    
    Returns
    -------
    dict[str, EarthLocation]
        Dictionary mapping station name to position
        
    Examples
    --------
    >>> gnss_pos = load_gnss_station_positions()
    >>> wsrt_pos = gnss_pos['WSRT00NLD']
    >>> print(wsrt_pos.lat, wsrt_pos.lon)
    """
    gnss_pos_dict = {}
    loaded = False
    
    # Try 1: Load from package data (installed package)
    try:
        if sys.version_info >= (3, 9):
            # Python 3.9+
            from importlib.resources import files
            station_file = files("spinifex_gnss.data").joinpath("data_gnss_pos.txt")
        else:
            # Python 3.8 and earlier
            import pkg_resources
            station_file = pkg_resources.resource_filename(
                "spinifex_gnss.data", "data_gnss_pos.txt"
            )
        
        with open(station_file) as f:
            for line in f:
                pos = [float(i) for i in line.strip().split()[1:]]
                gnss_pos_dict[line[:9]] = EarthLocation.from_geocentric(*pos, unit=u.m)
        loaded = True
        print(f"✓ Loaded {len(gnss_pos_dict)} GNSS stations from package data")
        
    except Exception as e:
        # Expected to fail if not installed as package
        pass
    
    # Try 2: Load from package directory (development mode)
    if not loaded:
        package_dir = Path(__file__).parent
        data_file = package_dir / "data" / "data_gnss_pos.txt"
        
        if data_file.exists():
            try:
                with open(data_file) as f:
                    for line in f:
                        pos = [float(i) for i in line.strip().split()[1:]]
                        gnss_pos_dict[line[:9]] = EarthLocation.from_geocentric(*pos, unit=u.m)
                loaded = True
                print(f"✓ Loaded {len(gnss_pos_dict)} GNSS stations from {data_file}")
            except Exception as e:
                print(f"Warning: Could not load from {data_file}: {e}")
    
    # Try 3: Load from parent data directory
    if not loaded:
        package_root = Path(__file__).parent.parent
        data_file = package_root / "data" / "data_gnss_pos.txt"
        
        if data_file.exists():
            try:
                with open(data_file) as f:
                    for line in f:
                        pos = [float(i) for i in line.strip().split()[1:]]
                        gnss_pos_dict[line[:9]] = EarthLocation.from_geocentric(*pos, unit=u.m)
                loaded = True
                print(f"✓ Loaded {len(gnss_pos_dict)} GNSS stations from {data_file}")
            except Exception as e:
                print(f"Warning: Could not load from {data_file}: {e}")
    
    # Try 4: Load from uploaded location (for testing)
    if not loaded:
        test_file = Path("/mnt/user-data/uploads/data_gnss_pos.txt")
        if test_file.exists():
            try:
                with open(test_file) as f:
                    for line in f:
                        pos = [float(i) for i in line.strip().split()[1:]]
                        gnss_pos_dict[line[:9]] = EarthLocation.from_geocentric(*pos, unit=u.m)
                loaded = True
                print(f"✓ Loaded {len(gnss_pos_dict)} GNSS stations from test location")
            except Exception as e:
                print(f"Warning: Could not load from test location: {e}")
    
    if not loaded or not gnss_pos_dict:
        print("=" * 70)
        print("WARNING: No GNSS station positions loaded!")
        print("=" * 70)
        print("Tried locations:")
        print("  1. Package data: spinifex_gnss.data")
        print("  2. Package dir: spinifex_gnss/data/data_gnss_pos.txt")
        print("  3. Parent dir: ../data/data_gnss_pos.txt")
        print("  4. Test location: /mnt/user-data/uploads/data_gnss_pos.txt")
        print()
        print("To fix:")
        print("  pip install -e . --force-reinstall")
        print("or manually place data_gnss_pos.txt in one of the above locations")
        print("=" * 70)
    
    return gnss_pos_dict


# Load stations at module import
gnss_pos_dict = load_gnss_station_positions()
