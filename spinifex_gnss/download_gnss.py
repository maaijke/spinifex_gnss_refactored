"""
GNSS data file downloading with RINEX2/RINEX3 fallback.

This module handles downloading of GNSS data files with automatic
fallback from RINEX3 to RINEX2 format when needed.

Key features:
- Tries RINEX3 first (modern format)
- Falls back to RINEX2 if RINEX3 not available
- Searches multiple servers automatically
- Async downloads for speed
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
    
    Handles both modern (with _IGS20 suffix) and older (without suffix) paths.
    IGS20 reference frame was adopted around week 2238 (late 2022).
    
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
        
    Notes
    -----
    Directory structure changed around GPS week 2238 (Nov 2022):
    - New: {gpsweek}_IGS20/ (IGS20 reference frame)
    - Old: {gpsweek}/ (no suffix)
    
    The function tries the modern path first, then falls back to legacy path.
    """
    sp3_names = []
    
    yesterday = date - timedelta(days=1)
    tomorrow = date + timedelta(days=1)
    
    yesterweek, _ = get_gps_week(yesterday)
    gpsweek, _ = get_gps_week(date)
    tomorrowweek, _ = get_gps_week(tomorrow)
    
    # IGS20 reference frame introduced around GPS week 2238 (late 2022)
    # For older data, use path without _IGS20 suffix
    igs20_week_threshold = 2238
    
    # Helper function to build SP3 URL with fallback
    def build_sp3_url(week: int, date_obj: datetime) -> str:
        """Build SP3 URL with _IGS20 suffix if recent, otherwise without."""
        doy = date_obj.timetuple().tm_yday
        year = date_obj.year
        filename = f"GBM0MGXRAP_{year}{doy:03d}0000_01D_05M_ORB.SP3.gz"
        
        if week >= igs20_week_threshold:
            # Modern path with _IGS20
            return f"{url}{week}_IGS20/{filename}"
        else:
            # Legacy path without suffix
            return f"{url}{week}/{filename}"
    
    sp3_names.append(build_sp3_url(yesterweek, yesterday))
    sp3_names.append(build_sp3_url(gpsweek, date))
    sp3_names.append(build_sp3_url(tomorrowweek, tomorrow))
    
    # Try to download with fallback
    downloaded_files = []
    
    for sp3_url in sp3_names:
        try:
            # Try primary URL
            file = await download_or_copy_url(sp3_url, output_directory=datapath)
            downloaded_files.append(file)
        except Exception as e:
            # If modern path failed, try legacy path (or vice versa)
            print(f"Failed to download {sp3_url}: {e}")
            
            # Extract week and filename from URL
            if "_IGS20/" in sp3_url:
                # Try without _IGS20
                fallback_url = sp3_url.replace("_IGS20/", "/")
                print(f"  Trying fallback: {fallback_url}")
            else:
                # Try with _IGS20
                # Extract week number from URL
                week_match = sp3_url.split("/")[-2]  # Get directory part
                fallback_url = sp3_url.replace(f"/{week_match}/", f"/{week_match}_IGS20/")
                print(f"  Trying fallback: {fallback_url}")
            
            try:
                file = await download_or_copy_url(fallback_url, output_directory=datapath)
                downloaded_files.append(file)
                print(f"  ✓ Downloaded from fallback URL")
            except Exception as e2:
                print(f"  ✗ Fallback also failed: {e2}")
                raise Exception(f"Failed to download SP3 file from both primary and fallback URLs")
    
    return downloaded_files


def download_satpos_files(
    date: datetime,
    url: str = "ftp://ftp.gfz-potsdam.de/GNSS/products/mgex/",
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """
    Download SP3 satellite position files for date ± 1 day.
    
    Parameters
    ----------
    date : datetime
        Target date
    url : str, optional
        Server URL
    datapath : Path, optional
        Output directory
        
    Returns
    -------
    list[Path]
        Paths to 3 SP3 files
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
    """
    files = set()
    for url in url_list:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and not href.startswith(('?', '/', 'http')):
                    files.add(f"{url}{href}")
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            continue
    
    return files


def _build_rinex3_filename(station: str, year: int, doy: int) -> str:
    """
    Build RINEX3 filename.
    
    Format: STATIONID_R_YYYYDDD0000_01D_30S_MO.crx.gz
    
    Parameters
    ----------
    station : str
        9-character station ID (e.g., 'WSRT00NLD')
    year : int
        Year
    doy : int
        Day of year
        
    Returns
    -------
    str
        RINEX3 filename
    """
    return f"{station}_R_{year}{doy:03d}0000_01D_30S_MO.crx.gz"


def _build_rinex2_filenames(station: str, year: int, doy: int) -> list[str]:
    """
    Build possible RINEX2 filenames.
    
    RINEX2 format variations:
    - ssssdddf.yyo.gz (standard)
    - ssssdddf.yyd.gz (daily)
    - ssssdddf.yyO.gz (capital O)
    
    Where:
    - ssss = 4-char station ID (first 4 chars of 9-char ID)
    - ddd = day of year
    - f = file sequence (usually '0')
    - yy = 2-digit year
    
    Parameters
    ----------
    station : str
        9-character station ID
    year : int
        Year
    doy : int
        Day of year
        
    Returns
    -------
    list[str]
        List of possible RINEX2 filenames
    """
    # Take first 4 characters of station ID and lowercase
    station_4char = station[:4].lower()
    yy = year % 100
    
    # Common variations
    filenames = [
        f"{station_4char}{doy:03d}0.{yy}o.gz",  # Standard
        f"{station_4char}{doy:03d}0.{yy}d.gz",  # Daily
        f"{station_4char}{doy:03d}0.{yy}O.gz",  # Capital O
        f"{station_4char}{doy:03d}0.{yy}o.Z",   # Compress format
        f"{station_4char}{doy:03d}0.{yy}o.Z".upper(),   # Compress format uppercase
        f"{station_4char}{doy:03d}0.{yy}d.Z",   # Compress daily,
        f"{station_4char}{doy:03d}0.{yy}d.Z".upper(),   # Compress daily uppercase
    ]
    
    return filenames


async def download_rinex_coro(
    date: datetime,
    stations: list[str],
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """
    Download RINEX files with automatic RINEX2/RINEX3 fallback.
    
    Strategy:
    1. Try RINEX3 first 
    2. Fall back to RINEX2 if RINEX3 not found
    3. Search multiple servers for each file
    
    Parameters
    ----------
    date : datetime
        Observation date
    stations : list[str]
        List of station identifiers (9 characters)
    datapath : Path, optional
        Output directory
        
    Returns
    -------
    list[Path]
        Paths to successfully downloaded files
        
    Notes
    -----
    RINEX3 advantages:
    - Modern format
    - Better metadata
    - Multi-GNSS support
    - Longer filenames (more info)
    
    RINEX2 advantages:
    - Available for older data
    - More servers have it
    - Smaller files sometimes
    """
    urls = []
    year = date.year
    yy = year - 2000
    doy = date.timetuple().tm_yday
    
    # RINEX data servers
    server_list = [
        "https://cddis.nasa.gov/archive/gnss/data/daily/",
        "https://www.epncb.oma.be/pub/obs/",
        "https://webring.gm.ingv.it:44324/rinex/RING/",
        "https://ga-gnss-data-rinex-v1.s3.amazonaws.com/index.html#public/daily/",
    ]
    
    # Build server URLs for both RINEX3 and RINEX2
    rinex3_urls = [
        f"{server_list[0]}/{year}/{doy:03d}/{yy}d/",
        f"{server_list[1]}/{year}/{doy:03d}/",
        f"{server_list[2]}/{year}/{doy:03d}/",
        f"{server_list[3]}/{year}/{doy:03d}/",
    ]
    
    # RINEX2 may be in different directories
    rinex2_urls = [
        f"{server_list[0]}/{year}/{doy:03d}/{yy}o/",  # Often in separate dir
        f"{server_list[0]}/{year}/{doy:03d}/{yy}d/",  # Sometimes same as RINEX3
        f"{server_list[1]}/{year}/{doy:03d}/",
        f"{server_list[2]}/{year}/{doy:03d}/",
        f"{server_list[3]}/{year}/{doy:03d}/",
    ]
    
    # Get directory listings
    print(f"Checking RINEX3 servers for {date.date()}...")
    rinex3_files = check_url(rinex3_urls) 
    print(f"Checking RINEX2 directories for {date.date()}...")
    rinex2_files = check_url(rinex2_urls)    
    # Find file for each station
    for station in stations:
        found = False
        
        # Try RINEX3 first
        rinex3_filename = _build_rinex3_filename(station, year, doy)
        
        for url in rinex3_urls:
            full_url = f"{url}{rinex3_filename}"
            if full_url in rinex3_files:
                urls.append((full_url, 'RINEX3'))
                found = True
                print(f"  Found RINEX3: {station}")
                break
        
        # Fall back to RINEX2
        if not found:
            rinex2_filenames = _build_rinex2_filenames(station, year, doy)            
            for rinex2_filename in rinex2_filenames:
                for url in rinex2_urls:
                    full_url = f"{url}{rinex2_filename}"
                    if full_url in rinex2_files:
                        urls.append((full_url, 'RINEX2'))
                        found = True
                        print(f"  Found RINEX2: {station} ({rinex2_filename})")
                        break
                
                if found:
                    break
        
        if not found:
            print(f"  ⚠️  Warning: No RINEX file found for {station}")
    
    # Download all found files
    if urls:
        print(f"\nDownloading {len(urls)} RINEX files...")
        print(f"  RINEX3: {sum(1 for _, fmt in urls if fmt == 'RINEX3')}")
        print(f"  RINEX2: {sum(1 for _, fmt in urls if fmt == 'RINEX2')}")
        
        coros = [download_or_copy_url(url, output_directory=datapath) for url, _ in urls]
        return await asyncio.gather(*coros)
    else:
        print("No files to download!")
        return []


def download_rinex(
    date: datetime,
    stations: list[str],
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """
    Download RINEX files with automatic format fallback (sync wrapper).
    
    Parameters
    ----------
    date : datetime
        Observation date
    stations : list[str]
        List of station identifiers (9 characters)
    datapath : Path, optional
        Output directory
        
    Returns
    -------
    list[Path]
        Paths to downloaded files
        
    Examples
    --------
    >>> from datetime import datetime
    >>> stations = ['WSRT00NLD', 'IJMU00NLD']
    >>> date = datetime(2024, 6, 18)
    >>> files = download_rinex(date, stations)
    >>> print(f"Downloaded {len(files)} files")
    
    """
    return _download_rinex(date, stations, datapath)


# Create sync wrappers
_download_satpos_files = sync_wrapper(_download_satpos_files_coro)
_download_rinex = sync_wrapper(download_rinex_coro)
