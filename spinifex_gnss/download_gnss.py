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

def _build_sp3_filenames(date: datetime) -> list[str]:
    """
    Build list of possible SP3 filenames for a given date.
    
    SP3 filename evolution:
    
    1. Modern (2020+): GBM0MGXRAP_YYYYDDD0000_01D_05M_ORB.SP3.gz
       Example: GBM0MGXRAP_20241700000_01D_05M_ORB.SP3.gz
       
    2. Intermediate (2015-2020): gbmYYYYD.sp3.Z or gbmWWWWD.sp3.Z
       Example: gbm19010.sp3.Z (2019, DOY 010)
       Example: gbm20010.sp3.Z (GPS week 2001, DOY 0)
       
    3. Legacy (<2015): gbmWWWWD.sp3.Z (week + DOY)
       Example: gbm17650.sp3.Z (GPS week 1765, DOY 0)
    
    Parameters
    ----------
    date : datetime
        Date for which to build filenames
        
    Returns
    -------
    list[str]
        List of possible filenames, ordered by likelihood
    """
    year = date.year
    yy = year % 100  # 2-digit year
    doy = date.timetuple().tm_yday
    gps_week, gps_dow = get_gps_week(date)
    
    filenames = []
    
    # Format 1: Modern RINEX3-style (2020+)
    filenames.append(f"GBM0MGXRAP_{year}{doy:03d}0000_01D_05M_ORB.SP3.gz")
    
    # Format 2: Intermediate with 4-digit year (2015-2020)
    # Format: gbmYYDDD.sp3.Z where YY is last 2 digits of year
    filenames.append(f"gbm{yy}{doy:03d}0.sp3.Z")
    filenames.append(f"gbm{yy:02d}{doy:03d}0.sp3.gz")
    filenames.append(f"gbm{yy:02d}{doy:03d}0.sp3")
    
    # Format 3: GPS week-based (older, can appear in any era)
    # Format: gbmWWWWD.sp3.Z where WWWW is GPS week, D is day of week
    filenames.append(f"gbm{gps_week:04d}{gps_dow}.sp3.Z")
    filenames.append(f"gbm{gps_week:04d}{gps_dow}.sp3.gz")
    filenames.append(f"gbm{gps_week:04d}{gps_dow}.sp3")
    
    # Format 4: Alternative week-based formats
    filenames.append(f"GBM{gps_week:04d}{gps_dow}.sp3.Z")
    filenames.append(f"GBM{gps_week:04d}{gps_dow}.SP3.Z")
    
    # Format 5: Year-DOY without leading zero on day
    filenames.append(f"gbm{yy:02d}{doy:03d}.sp3.Z")
    
    return filenames


def _build_sp3_directory_paths(date: datetime, base_url: str) -> list[str]:
    """
    Build list of possible directory paths for SP3 files.
    
    Directory structure has also evolved:
    
    1. Modern (2022+): {gpsweek}_IGS20/
       Example: 2318_IGS20/
       
    2. Intermediate (2015-2022): {gpsweek}/
       Example: 2095/
       
    3. Legacy (<2015): Multiple possible structures
       - {gpsweek}/
       - {year}/{doy}/
       - repro2/{gpsweek}/ (reprocessed products)
    
    Parameters
    ----------
    date : datetime
        Target date
    base_url : str
        Base URL for MGEX products
        
    Returns
    -------
    list[str]
        List of directory URLs to try
    """
    gps_week, _ = get_gps_week(date)
    year = date.year
    doy = date.timetuple().tm_yday
    
    paths = []
    
    # IGS20 reference frame (2022+)
    if gps_week >= 2238:
        paths.append(f"{base_url}{gps_week}_IGS20/")
    
    # Standard week-based directory
    paths.append(f"{base_url}{gps_week}/")
    
    # Reprocessed products (often in separate directory)
    paths.append(f"{base_url}repro2/{gps_week}/")
    paths.append(f"{base_url}repro3/{gps_week}/")
    
    # Year/DOY structure (some servers use this)
    paths.append(f"{base_url}{year}/{doy:03d}/")
    
    return paths


async def _download_sp3_file_with_fallback(
    date: datetime,
    base_url: str,
    datapath: Path
) -> Path:
    """
    Download SP3 file for a specific date with comprehensive fallback.
    
    Tries multiple combinations of:
    - Directory paths (modern, intermediate, legacy)
    - Filename formats (modern, intermediate, legacy)
    - Compression formats (.gz, .Z, uncompressed)
    
    Parameters
    ----------
    date : datetime
        Target date
    base_url : str
        Base URL for GNSS products
    datapath : Path
        Output directory
        
    Returns
    -------
    Path
        Path to downloaded file
        
    Raises
    ------
    Exception
        If file cannot be found in any format
    """
    # Build all possible directory paths
    directory_paths = _build_sp3_directory_paths(date, base_url)
    
    # Build all possible filenames
    filenames = _build_sp3_filenames(date)
    
    # Try all combinations
    errors = []
    
    for dir_path in directory_paths:
        for filename in filenames:
            url = f"{dir_path}{filename}"
            
            try:
                print(f"  Trying: {url}")
                file = await download_or_copy_url(url, output_directory=datapath)
                print(f"  ✓ Downloaded: {filename}")
                return file
            except Exception as e:
                errors.append(f"{url}: {e}")
                continue
    
    # If we get here, nothing worked
    raise Exception(
        f"Failed to download SP3 for {date.date()} after trying "
        f"{len(directory_paths) * len(filenames)} combinations.\n"
        f"Tried directories: {directory_paths[:3]}...\n"
        f"Tried filenames: {filenames[:3]}..."
    )


async def _download_satpos_files_coro(
    date: datetime,
    url: str = "ftp://ftp.gfz-potsdam.de/GNSS/products/mgex/",
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """
    Download SP3 satellite orbit files for date ± 1 day.
    
    Handles multiple SP3 filename and directory formats with comprehensive
    fallback logic for historical data support.
    
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
    SP3 format evolution:
    
    Modern (2020+):
    - Filename: GBM0MGXRAP_YYYYDDD0000_01D_05M_ORB.SP3.gz
    - Directory: {gpsweek}_IGS20/ (if week >= 2238) or {gpsweek}/
    - Compression: .gz
    
    Intermediate (2015-2020):
    - Filename: gbmYYDDD0.sp3.Z or gbmWWWWD.sp3.Z
    - Directory: {gpsweek}/
    - Compression: .Z (Unix compress)
    
    Legacy (<2015):
    - Filename: gbmWWWWD.sp3.Z
    - Directory: {gpsweek}/ or repro2/{gpsweek}/
    - Compression: .Z
    
    The function tries all possible combinations automatically.
    """
    yesterday = date - timedelta(days=1)
    tomorrow = date + timedelta(days=1)
    
    print(f"\nDownloading SP3 files for {date.date()} ± 1 day...")
    
    # Download files for all three days
    files = []
    
    for day_date, day_name in [(yesterday, "yesterday"), (date, "today"), (tomorrow, "tomorrow")]:
        print(f"\n{day_name.capitalize()} ({day_date.date()}):")
        try:
            file = await _download_sp3_file_with_fallback(day_date, url, datapath)
            files.append(file)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            raise
    
    return files


def download_satpos_files(
    date: datetime,
    url: str = "ftp://ftp.gfz-potsdam.de/GNSS/products/mgex/",
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """
    Download SP3 satellite position files for date ± 1 day (sync wrapper).
    
    Automatically handles multiple SP3 file formats across different eras.
    
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
        
    Examples
    --------
    >>> from datetime import datetime
    >>> 
    >>> # Modern data (2024)
    >>> date = datetime(2024, 6, 18)
    >>> sp3_files = download_satpos_files(date)
    >>> # Downloads: GBM0MGXRAP_2024170*.SP3.gz
    >>> 
    >>> # Intermediate data (2018)
    >>> date = datetime(2018, 3, 15)
    >>> sp3_files = download_satpos_files(date)
    >>> # Downloads: gbm18074*.sp3.Z or gbm1987*.sp3.Z
    >>> 
    >>> # Legacy data (2012)
    >>> date = datetime(2012, 1, 10)
    >>> sp3_files = download_satpos_files(date)
    >>> # Downloads: gbm1668*.sp3.Z
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

def check_local_rinex_files(
    stations: list[str],
    date: datetime,
    datapath: Path,
) -> tuple[list[str], list[str],list[str]]:
    """
    Check which stations already have local RINEX files.
    
    This optimization avoids unnecessary server queries.
    
    Parameters
    ----------
    stations : list[str]
        List of station IDs (9-char format)
    date : datetime
        Target date
    datapath : Path
        Data directory to check
        
    Returns
    -------
    tuple[list[str], list[str], list[Path]]
        (stations_found, stations_missing, existing_file_paths)
        
    Examples
    --------
    >>> found, missing, paths = check_local_rinex_files(
    ...     ['WSRT00NLD', 'IJMU00NLD'],
    ...     datetime(2024, 6, 18),
    ...     Path('./data/')
    ... )
    >>> print(f"Already have: {found}")
    >>> print(f"Need to download: {missing}")
    """
    year = date.year
    doy = date.timetuple().tm_yday
    yy = year % 100
    
    datapath = Path(datapath)
    stations_missing = []
    existing_paths = []
    
    for station in stations:
        found_path = None
        
        # RINEX3 patterns (preferred)
        rinex3_patterns = [
                f"{station}_R_{year}{doy:03d}0000_01D_30S_MO.crx.gz",
                f"{station}_R_{year}{doy:03d}0000_01D_30S_MO.crx",
                f"{station}_R_{year}{doy:03d}0000_01D_30S_MO.rnx.gz",
                f"{station}_R_{year}{doy:03d}0000_01D_30S_MO.rnx",
            ]
            
        for pattern in rinex3_patterns:
            path = datapath / pattern
            if path.exists():
                found_path = (path.name, 'RINEX3')
                break
        
        # RINEX2 patterns (fallback)
        if found_path is None:
            station_4char = station[:4].lower()
            rinex2_patterns = [
                f"{station_4char}{doy:03d}0.{yy}o.gz",
                f"{station_4char}{doy:03d}0.{yy}d.gz",
                f"{station_4char}{doy:03d}0.{yy}O.gz",
                f"{station_4char}{doy:03d}0.{yy}d.Z",
                f"{station_4char}{doy:03d}0.{yy}o",
                f"{station_4char}{doy:03d}0.{yy}d",
                f"{station_4char}{doy:03d}0.{yy}d.Z".upper(),
                f"{station_4char}{doy:03d}0.{yy}o".upper(),
                f"{station_4char}{doy:03d}0.{yy}d".upper(),
            ]
            
            for pattern in rinex2_patterns:
                path = datapath / pattern
                if path.exists():
                    found_path = (path.name, 'RINEX2')
                    break
        
        if found_path:
            existing_paths.append(found_path)
        else:
            stations_missing.append(station)
    
    return stations_missing, existing_paths


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
    print("checking local files for {date.date()}")
    stations_missing, urls = check_local_rinex_files(stations, date, datapath)
    if len(stations_missing):
        print(f"Checking RINEX3 servers for {date.date()}...")
        rinex3_files = check_url(rinex3_urls) 
        print(f"Checking RINEX2 directories for {date.date()}...")
        rinex2_files = check_url(rinex2_urls)    
    # Find file for each station
    for station in stations_missing:
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
