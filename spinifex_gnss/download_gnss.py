"""
GNSS data file downloading.

This module handles downloading of GNSS data files from various servers.

Refactored to remove:
- DCB file downloading (no longer needed)
"""

import asyncio
import gps_time
from datetime import datetime, timedelta
from spinifex.download import download_or_copy_url
from spinifex.asyncio_wrapper import sync_wrapper
from pathlib import Path
from bs4 import BeautifulSoup
import requests


def get_gps_week(date: datetime) -> tuple[int, int]:
    """
    Get GPS week number and day of week from date.
    
    Parameters
    ----------
    date : datetime
        Date to convert
        
    Returns
    -------
    tuple[int, int]
        GPS week number and day of week
    """
    gpstime = gps_time.GPSTime.from_datetime(date)
    return gpstime.week_number, int(gpstime.time_of_week / (24 * 3600))


async def _download_satpos_files_coro(
    date: datetime,
    url: str = "ftp://ftp.gfz-potsdam.de/GNSS/products/mgex/",
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """
    Download SP3 satellite orbit files for date ± 1 day.
    
    Downloads 3 consecutive days of SP3 files for interpolation purposes.
    CLK files are no longer downloaded (not used).
    
    Parameters
    ----------
    date : datetime
        Target date
    url : str, optional
        Base URL for MGEX products
    datapath : Path, optional
        Output directory
        
    Returns
    -------
    list[Path]
        Paths to downloaded SP3 files (3 files for consecutive days)
    """
    sp3_names = []
    
    # Get dates for yesterday, today, tomorrow
    yesterday = date - timedelta(days=1)
    tomorrow = date + timedelta(days=1)
    
    # Get GPS weeks
    yesterweek, _ = get_gps_week(yesterday)
    gpsweek, _ = get_gps_week(date)
    tomorrowweek, _ = get_gps_week(tomorrow)
    if yesterweek >= 2238:
        use_path_label = "_IGS20/"
    else:
        use_path_label = ""
    # Build SP3 file URLs
    sp3_names.append(
        f"{url}{yesterweek}{use_path_label}/GBM0MGXRAP_{yesterday.year}{yesterday.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3.gz"
    )
    sp3_names.append(
        f"{url}{gpsweek}{use_path_label}/GBM0MGXRAP_{date.year}{date.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3.gz"
    )
    sp3_names.append(
        f"{url}{tomorrowweek}{use_path_label}/GBM0MGXRAP_{tomorrow.year}{tomorrow.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3.gz"
    )
    
    # Note: CLK files removed - not used in refactored version
    # Old code: clk_name = f"{url}{gpsweek}_IGS20/GBM0MGXRAP_{date.year}{date.timetuple().tm_yday:03d}0000_01D_30S_CLK.CLK.gz"
    
    # Download all files
    coros = [download_or_copy_url(url, output_directory=datapath) for url in sp3_names]
    return await asyncio.gather(*coros)


def download_satpos_files(
    date: datetime,
    url: str = "ftp://ftp.gfz-potsdam.de/GNSS/products/mgex/",
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """
    Download SP3 satellite position files for date ± 1 day.
    
    Gets 3 consecutive days of orbit files for interpolation.
    
    Parameters
    ----------
    date : datetime
        Target date
    url : str, optional
        Server URL for GNSS products
    datapath : Path, optional
        Output directory
        
    Returns
    -------
    list[Path]
        Paths to 3 SP3 files (sorted: yesterday, today, tomorrow)
        
    Notes
    -----
    No longer downloads CLK files - satellite clock corrections
    from SP3 files are sufficient for TEC calculations.
    
    Examples
    --------
    >>> from datetime import datetime
    >>> date = datetime(2024, 6, 18)
    >>> sp3_files = download_satpos_files(date)
    >>> print(len(sp3_files))
    3
    """
    return _download_satpos_files(date, url, datapath)


def check_url(url_list: list[str]) -> set[str]:
    """
    Check URLs and parse directory listings for available files.
    
    Parameters
    ----------
    url_list : list[str]
        List of directory URLs to check
        
    Returns
    -------
    set[str]
        Set of available file URLs
        
    Notes
    -----
    This function is relatively slow as it checks each server serially.
    Could be optimized with async requests in future.
    """
    files = set()
    for url in url_list:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML directory listing
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links in the directory
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and not href.startswith(('?', '/', 'http')):
                    files.add(f"{url}{href}")
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            continue
    
    return files


async def download_rinex_coro(
    date: datetime,
    stations: list[str],
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """
    Download RINEX observation files for multiple stations.
    
    Tries multiple servers to find data for each station.
    
    Parameters
    ----------
    date : datetime
        Observation date
    stations : list[str]
        List of station identifiers (9 characters, e.g., 'WSRT00NLD')
    datapath : Path, optional
        Output directory
        
    Returns
    -------
    list[Path]
        Paths to successfully downloaded RINEX files
        
    Notes
    -----
    Searches multiple GNSS data servers:
    - CDDIS (NASA)
    - EUREF Permanent Network
    - Italian RING Network
    - Geoscience Australia
    
    Files are in Hatanaka compressed format (.crx.gz).
    """
    urls = []
    year = date.year
    yy = date.year - 2000
    doy = date.timetuple().tm_yday
    
    # List of RINEX data servers
    server_list = [
        "https://cddis.nasa.gov/archive/gnss/data/daily/",
        "https://www.epncb.oma.be/pub/obs/",
        "https://webring.gm.ingv.it:44324/rinex/RING/",
        "https://ga-gnss-data-rinex-v1.s3.amazonaws.com/index.html#public/daily/",
    ]
    
    url_list = [
        f"{server_list[0]}/{year}/{doy:03d}/{yy}d/",
        f"{server_list[1]}/{year}/{doy:03d}/",
        f"{server_list[2]}/{year}/{doy:03d}/",
        f"{server_list[3]}/{year}/{doy:03d}/"
    ]
    
    # Get directory listings from all servers
    print(f"Checking servers: {url_list}")
    files_per_url = check_url(url_list)
    
    # Find RINEX file for each station
    for station in stations:
        fname = f"{station}_R_{year}{doy:03d}0000_01D_30S_MO.crx.gz"
        found = False
        
        # Check each server for this file
        for url in url_list:
            if f"{url}{fname}" in files_per_url:
                urls.append(f"{url}{fname}")
                found = True
                break
        
        if not found:
            print(f"  Warning: {fname} not found on any server")
    
    # Download all found files
    print(f"Downloading {len(urls)} RINEX files...")
    coros = [download_or_copy_url(url, output_directory=datapath) for url in urls]
    return await asyncio.gather(*coros)


def download_rinex(
    date: datetime,
    stations: list[str],
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """
    Download RINEX observation files for multiple stations (sync wrapper).
    
    Parameters
    ----------
    date : datetime
        Observation date
    stations : list[str]
        List of station identifiers
    datapath : Path, optional
        Output directory
        
    Returns
    -------
    list[Path]
        Paths to downloaded RINEX files
        
    Examples
    --------
    >>> from datetime import datetime
    >>> stations = ['WSRT00NLD', 'IJMU00NLD']
    >>> date = datetime(2024, 6, 18)
    >>> rinex_files = download_rinex(date, stations)
    >>> print(f"Downloaded {len(rinex_files)} files")
    """
    return _download_rinex(date, stations, datapath)


# Create sync wrappers for async functions
_download_satpos_files = sync_wrapper(_download_satpos_files_coro)
_download_rinex = sync_wrapper(download_rinex_coro)


# Note: download_dcb functions removed - DCB files no longer needed!
# If DCB is needed in future, it can be restored from version control.
