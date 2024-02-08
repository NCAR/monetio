"""
Load NOAA Global Monitoring Laboratory (GML) ozonesondes
from https://gml.noaa.gov/aftp/data/ozwv/Ozonesonde/

More info: https://gml.noaa.gov/ozwv/ozsondes/
"""
import re
import warnings
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests


def retry(func):
    import time
    from functools import wraps
    from random import random as rand

    n = 3

    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(n):
            try:
                res = func(*args, **kwargs)
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ):
                time.sleep(0.5 * i + rand() * 0.1)
            else:
                break
        else:
            raise RuntimeError(f"{func.__name__} failed after {n} tries.")

        return res

    return wrapper


PLACES = [
    "Boulder, Colorado",
    "Hilo, Hawaii",
    "Huntsville, Alabama",
    "Narragansett, Rhode Island",
    "Pago Pago, American Samoa",
    "San Cristobal, Galapagos",
    "South Pole, Antarctica",
    "Summit, Greenland",
    "Suva, Fiji",
    "Trinidad Head, California",
]


_FILES_L100_CACHE = {place: None for place in PLACES}


def discover_files(place=None, *, n_threads=3, cache=True):
    import itertools
    from multiprocessing.pool import ThreadPool

    base = "https://gml.noaa.gov/aftp/data/ozwv/Ozonesonde"

    if place is None:
        places = PLACES
    elif isinstance(place, str):
        places = [place]
    else:
        places = place

    invalid = set(places) - set(PLACES)
    if invalid:
        raise ValueError(f"Invalid place(s): {invalid}. Valid options: {PLACES}.")

    @retry
    def get_files(place):
        cached = _FILES_L100_CACHE[place]
        if cached is not None:
            return cached

        if place == "South Pole, Antarctica":
            url_place = "South Pole, Antartica"  # note sp
        else:
            url_place = place
        url = f"{base}/{url_place}/100 Meter Average Files/".replace(" ", "%20")
        print(url)

        r = requests.get(url, timeout=10)
        r.raise_for_status()

        data = []
        for m in re.finditer(r'href="([a-z0-9_]+\.l100)"', r.text):
            fn = m.group(1)
            if fn.startswith("san_cristobal_"):
                a, b = 3, -1
            else:
                a, b = 1, -1
            t_str = "".join(re.split(r"[_\.]", fn)[a:b])
            try:
                t = pd.to_datetime(t_str, format=r"%Y%m%d%H")
            except ValueError:
                warnings.warn(f"Failed to parse file name {fn!r} for time.")
                t = np.nan
            data.append((place, t, fn, f"{url}{fn}"))

        if not data:
            warnings.warn(f"No files detected for place {place!r}.")

        return data

    with ThreadPool(processes=min(n_threads, len(places))) as pool:
        data = list(itertools.chain.from_iterable(pool.imap_unordered(get_files, places)))

    df = pd.DataFrame(data, columns=["place", "time", "fn", "url"])

    if cache:
        for place in places:
            _FILES_L100_CACHE[place] = list(
                df[df["place"] == place].itertuples(index=False, name=None)
            )

    return df


def add_data(dates, *, place=None, n_procs=1, errors="raise"):
    """Retrieve and load GML ozonesonde data as a DataFrame.

    Parameters
    ----------
    dates : sequence of datetime-like
        The period between the min and max (both inclusive)
        will be used to select the files to load.
    place : str or sequence of str, optional
        For example 'Boulder, Colorado'.
        If not provided, all places will be used.
    n_procs : int
        For Dask.
    errors : {'raise', 'warn', 'skip'}
        What to do when there is an error reading a file.
    """
    import dask
    import dask.dataframe as dd

    dates = pd.DatetimeIndex(dates)
    dates_min, dates_max = dates.min(), dates.max()

    if errors not in {"raise", "warn", "ignore"}:
        raise ValueError(f"Invalid errors setting: {errors!r}.")

    print("Discovering files...")
    df_urls = discover_files(place=place)
    print(f"Discovered {len(df_urls)} 100-m files.")

    urls = df_urls[df_urls["time"].between(dates_min, dates_max, inclusive="both")]["url"].tolist()

    if not urls:
        raise RuntimeError(f"No files found for dates {dates_min} to {dates_max}, place={place}.")

    def func(fp_or_url):
        try:
            return read_100m(fp_or_url)
        except Exception as e:
            msg = f"Failed to read {fp_or_url}: {e}"
            if errors == "raise":
                raise RuntimeError(msg) from e
            else:
                if errors == "warn":
                    warnings.warn(msg)
                return pd.DataFrame()

    print(f"Aggregating {len(urls)} files...")
    dfs = [dask.delayed(func)(url) for url in urls]
    dff = dd.from_delayed(dfs, verify_meta=errors == "raise")
    df = dff.compute(num_workers=n_procs).reset_index()

    # Time subset again in case of times in files extending
    df = df[df["time"].between(dates_min, dates_max, inclusive="both")]

    # Normalize station
    # All values, as of 2024-02-08:
    # > df.station.value_counts().sort_index()
    # Boulder, CO                          650757
    # Hilo, Hawaii                         627325
    # Hilo,Hawaii                             192
    # Huntsville                            10982
    # Huntsville, AL                       314375
    # Mauna Loa Observatory, Hawaii           605 (different site than Hilo)
    # Pago Pago, American Samoa            370141
    # San Cristobal, Galapagos, Ecuador    150244
    # South Pole                           661422
    # Summit, Greenland                    164620
    # Suva, Fiji                           164065
    # Trinidad Head, CA                    426409
    # University of Rhode Island           105878
    # helikite test                           326
    # hsv                                     340
    repl = {
        "Boulder, CO": "Boulder, Colorado",
        "Hilo,Hawaii": "Hilo, Hawaii",
        "Huntsville": "Huntsville, Alabama",
        "Huntsville, AL": "Huntsville, Alabama",
        "San Cristobal, Galapagos, Ecuador": "San Cristobal, Galapagos",
        "South Pole": "South Pole, Antarctica",
        "Trinidad Head, CA": "Trinidad Head, California",
    }
    assert set(repl.values()) <= set(PLACES)
    df["station"] = df["station"].replace(repl)

    # Add metadata
    if hasattr(df, "attrs"):
        df.attrs["ds_attrs"] = {"urls": urls}
        df.attrs["var_attrs"] = {
            c.name: {
                "long_name": c.long_name,
                "units": c.units,
            }
            for c in COL_INFO_L100
        }

    return df.drop(columns=["index"], errors="ignore").reset_index(drop=True)


class ColInfo(NamedTuple):
    name: str
    long_name: str
    units: str
    na_val: Optional[Union[str, Tuple[str, ...]]]


COL_INFO_L100 = [
    # name, long name, units, na val
    #
    # "Level" (just a counter, should never be nan)
    ColInfo("lev", "level", "", None),
    #
    # "Press"
    ColInfo("press", "radiosonde corrected pressure", "hPa", "9999.9"),
    #
    # "Alt"
    # TODO: not sure about this na val
    ColInfo("altitude", "altitude", "km", "999.999"),
    #
    # "Pottp"
    ColInfo("theta", "potential temperature", "K", "9999.9"),
    #
    # "Temp"
    ColInfo("temp", "radiosonde corrected temperature", "degC", "999.9"),
    #
    # "FtempV"
    ColInfo("ftempv", "frost point temperature (radiosonde)", "degC", "999.9"),
    #
    # "Hum"
    ColInfo("rh", "radiosonde corrected relative humidity", "%", "999"),
    #
    # "Ozone"
    ColInfo("o3_press", "ozone partial pressure", "mPa", "99.90"),
    #
    # "Ozone"
    ColInfo("o3", "ozone mixing ratio", "ppmv", "99.999"),
    #
    # "Ozone"
    # Note 1 DU = 0.001 atm-cm
    # TODO: goes up with height so could be ozone below?
    ColInfo("o3_cm", "total ozone", "atm-cm", "99.9990"),
    #
    # "Ptemp"
    ColInfo("ptemp", "pump temperature", "degC", "999.9"),
    #
    # "O3 # DN"
    ColInfo("o3_nd", "ozone number density", "10^11 cm-3", "999.999"),
    #
    # "O3 Res"
    # From Owen Cooper (NOAA CSL):
    #   This is the amount of ozone in Dobson units above a given altitude.
    #   The values above the maximum balloon altitude are from a climatology.
    #   This is mainly for UV absorption research.
    ColInfo("o3_col", "total column ozone above", "DU", ("9999", "99999", "99.999")),
    #
    # "O3 Uncert"
    # Estimated uncertainty in the ozone measurement at a given altitude.
    ColInfo("o3_uncert", "uncertainty in ozone", "%", ("99999.000", "99.999")),
]


_DATA_BLOCK_START_L100 = """\
Level   Press    Alt   Pottp   Temp   FtempV   Hum  Ozone  Ozone   Ozone  Ptemp  O3 # DN O3 Res  O3 Uncert
 Num     hPa      km     K      C       C       %    mPa    ppmv   atmcm    C   10^11/cc   DU          %
"""

_DATA_BLOCK_START_L100_NO_UNCERT = """\
Level   Press    Alt   Pottp   Temp   FtempV   Hum  Ozone  Ozone   Ozone  Ptemp  O3 # DN O3 Res
 Num     hPa      km     K      C       C       %    mPa    ppmv   atmcm    C   10^11/cc   DU
"""


def read_100m(fp_or_url):
    """Read a GML ozonesonde 100-m file (``.l100``).

    Notes
    -----
    Close to ICARTT format, but not quite conformant enough to use the ICARTT reader.
    """
    from io import StringIO

    if isinstance(fp_or_url, str) and fp_or_url.startswith(("http://", "https://")):

        @retry
        def get_text():
            r = requests.get(fp_or_url, timeout=10)
            r.raise_for_status()
            return r.text

    else:

        def get_text():
            with open(fp_or_url) as f:
                text = f.read()
            return text

    blocks = get_text().replace("\r", "").split("\n\n")
    nblocks = len(blocks)
    if nblocks == 5:  # normal
        meta_block = blocks[3]
        data_block = blocks[4]
    elif nblocks == 2:
        block_lines = blocks[0].splitlines()
        for i, line in enumerate(block_lines):
            if line.startswith(("Station:", "Station: ", "Station  ")):
                break
        else:
            raise ValueError(f"Expected to find metadata to start with Station, got:\n{blocks[0]}")
        meta_block = "\n".join(block_lines[i:])
        data_block = blocks[1]
    else:
        heads = "\n".join("\n".join(b.splitlines()[:2] + ["..."]) for b in blocks)
        raise ValueError(f"Expected 2 or 5 blocks, got {nblocks}:\n{heads}")

    # Metadata
    meta = {}
    todo = meta_block.splitlines()[::-1]
    on_val_side = ["Background: ", "Flowrate: ", "RH Corr: ", "Sonde Total O3 (SBUV): "]
    while todo:
        line = todo.pop()
        key, val = line.split(":", 1)
        for key_ish in on_val_side:
            if key_ish in val:
                i = val.index(key_ish)
                meta[key.strip()] = val[:i].strip()
                todo.append(val[i:])
                break
        else:
            meta[key.strip()] = val.strip()

    for k, v in meta.items():
        meta[k] = re.sub(r"\s{2,}", " ", v)

    meta_keys_expected = [
        "Station",
        "Station Height",
        "Latitude",
        "Longitude",
        "Flight Number",
        "Launch Date",
        "Launch Time",
        # May see 'Vaisala number' and 'Vaisala humicap' instead of these two:
        # "Radiosonde Type",
        # "Radiosonde Num",
        "O3 Sonde ID",
        "Background",
        "Flowrate",
        "RH Corr",
        "Sonde Total O3",
        "Sonde Total O3 (SBUV)",
    ]
    if not set(meta) >= set(meta_keys_expected):
        raise ValueError(f"Expected metadata keys {meta_keys_expected}, got {list(meta)}.")

    if data_block.startswith(_DATA_BLOCK_START_L100):
        have_uncert = True
    elif data_block.startswith(_DATA_BLOCK_START_L100_NO_UNCERT):
        have_uncert = False
    else:
        head = "\n".join(data_block.splitlines()[:2] + ["..."])
        raise ValueError(
            "Data block does not start with expected header line(s) "
            "(O3 Uncert allowed to be missing):\n"
            f"{_DATA_BLOCK_START_L100}\n"
            f"got\n{head}"
        )

    col_info = COL_INFO_L100[:]
    if not have_uncert:
        _ = col_info.pop()

    ncol_expected = len(col_info)
    data_block_ncol = len(data_block[:400].splitlines()[2].split())
    if not data_block_ncol == ncol_expected:
        head = "\n".join(data_block.splitlines()[:4] + ["..."])
        raise ValueError(
            f"Expected {ncol_expected} columns in data block, "
            f"got {data_block_ncol} in first data line:\n{head}"
        )
        # TODO: allow pandas to skip bad lines with `on_bad_lines='skip'`?

    names = [c.name for c in col_info]
    dtype = {c.name: float for c in col_info}
    dtype["lev"] = int
    na_values = {c.name: c.na_val for c in col_info if c.na_val is not None}

    df = pd.read_csv(
        StringIO(data_block),
        skiprows=2,
        header=None,
        delimiter=r"\s+",
        names=names,
        dtype=dtype,
        na_values=na_values,
    )

    # Note: This is close to "Pottp" but not exactly the same
    # theta_calc = (df.temp + 273.15) * (df.press / 1000) ** (-0.286)

    # Add some variables from header (these don't change in the profile)
    time = pd.Timestamp(f"{meta['Launch Date']} {meta['Launch Time']}")
    df["time"] = time.tz_localize(None)
    df["latitude"] = float(meta["Latitude"])
    df["longitude"] = float(meta["Longitude"])
    df["station"] = meta["Station"]
    df["station_height_str"] = meta["Station Height"]  # e.g. '1743 meters'
    df["o3_tot_cmr_str"] = meta["Sonde Total O3"]
    df["o3_tot_sbuv_str"] = meta["Sonde Total O3 (SBUV)"]  # e.g. '325 (62) DU'
    # TODO: '99999 (99999) DU' if NA, could put empty string or None or NaN instead?

    # Add metadata
    if hasattr(df, "attrs"):
        df.attrs["ds_attrs"] = meta
        df.attrs["var_attrs"] = {
            c.name: {
                "long_name": c.long_name,
                "units": c.units,
            }
            for c in col_info
        }

    return df
