import hatanaka
from pathlib import Path
import numpy as np
from typing import NamedTuple, Any
from astropy.time import Time


class RinexHeader(NamedTuple):
    version: str
    datatypes: dict[str, list[str]]


class RinexData(NamedTuple):
    data: dict[str, dict[Any]]
    times: Time
    header: RinexHeader


def _read_rinex_header(raw_rinex_lines: list[str])-> RinexHeader:
    """Read header information from rinex 3 file

    Parameters
    ----------
    raw_rinex_lines : list[str]
        all lines in the file

    Returns
    -------
    RinexHeader
        object with rinex version and datatypes 
    """    
    version = ""
    obs_map = {}
    sys = ""
    for line_number, line in enumerate(raw_rinex_lines):
        type = line[60:81]
        if "END OF HEADER" in type:
            return RinexHeader(version=version, datatypes=obs_map), line_number
        else:
            if "RINEX VERSION / TYPE" in type:
                version = line[:61].strip()
        if "SYS / # / OBS TYPES" in type:
            toks = line[6:60].split()
            if not sys:
                sys = line.strip()[0]  # satelite system (G,E,S,R,C,J,I)
                nobs = int(line[4:6])
                types = toks
            else:
                types += toks
            if len(types) < nobs:
                # continuation lines
                continue
            obs_map[sys] = types
            sys = ""
    return None, None


def get_rinex_data(fname: Path)->RinexData:
    """parse rinex3 file

    Parameters
    ----------
    fname : Path
        path to the file, assumed hatanaka (+optional gzip) compressed 

    Returns
    -------
    RinexData
        object with data, times (gpstime) and header 
    """    
    rinex_lines = hatanaka.decompress(fname).decode().split("\n")
    header, end_of_header = _read_rinex_header(rinex_lines)
    cur_time = None
    all_times = []
    data = {}
    width = 16
    no_cur_time=True
    for line in rinex_lines[end_of_header:]:
        if line.startswith(">"):  # epoch record (RINEX 3)
            try:
                parts = line.split()
                yr, mo, dy, hr, mi = map(int, parts[1:6])
                sec = float(parts[6])
                cur_time = Time(
                    f"{yr}-{mo}-{dy}T{hr}:{mi}:{sec}"
                )  # note: this is gps time (leap seconds!!)
                all_times.append(cur_time.mjd)
                no_cur_time = False
                continue
            except:
                no_cur_time = True
                continue
        if no_cur_time:
            continue #continue reading until we find a correct time 
        sat_id = line[:3].strip()
        if not sat_id:
            continue
        n_types = len(header.datatypes[sat_id[:1]])
        obs_vals = [
            float(i.split()[0]) if i.split() else np.nan
            for i in [
                line[i : i + width].strip()
                for i in range(3, 3 + n_types * width, width)
            ]
        ]
        if sat_id not in data:
            data[sat_id] = {"time": [cur_time], "data": [obs_vals]}
        else:
            data[sat_id]["time"].append(cur_time)
            data[sat_id]["data"].append(obs_vals)
    all_times = np.array(all_times)
    newdata = {}
    for prn, prndata in data.items():
        alldata = np.empty((len(all_times), len(prndata["data"][0])))
        alldata.fill(np.nan)
        for tm, dt in zip(prndata["time"], prndata["data"]):
            tm_idx = np.argmin(np.abs(all_times - tm.mjd))
            alldata[tm_idx] = dt
        newdata[prn] = alldata
    return RinexData(header=header, times=Time(all_times, format="mjd"), data=newdata)


#TODO: add RNX2 parser