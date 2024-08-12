"""TEMPO L2 NO2 data file reader.

History:


"""

import logging
import os
import sys
from collections import OrderedDict
from glob import glob
import copy
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from datetime import datetime
from cftime import num2date
from scipy.interpolate import griddata
import pdb


def _open_one_dataset(fname, variable_dict):
    """Read locally stored INTELSAT TEMPO NO2 level 2
    Parameters
    ----------
    fname: string
        fname is local path to netcdf4 (HDF5) file

    variable_dict: dict
    Returns
    -------
    ds: xarray dataset
    """
    print("reading " + fname)

    ds = xr.Dataset()

    dso = Dataset(fname, "r")
    lon_var = dso.groups["geolocation"]["longitude"]
    lat_var = dso.groups["geolocation"]["latitude"]
    time_var = dso.groups["geolocation"]["time"]

    ds["lon"] = (
        ("x", "y"),
        lon_var[:].squeeze(),
        {"long_name": lon_var.long_name, "units": lon_var.units},
    )
    ds["lat"] = (
        ("x", "y"),
        lat_var[:].squeeze(),
        {"long_name": lat_var.long_name, "units": lat_var.units},
    )
    ds["time"] = (
        ("time",),
        time_var[:].squeeze(),
        {"long_name": time_var.long_name, "units": time_var.units},
    )
    ds["lon"] = ds["lon"].fillna(-999.99)
    ds["lat"] = ds["lat"].fillna(-999.99)
    ds = ds.set_coords(["time", "lon", "lat"])

    ds.attrs["reference_time_string"] = dso.time_coverage_start
    ds.attrs["granule_number"] = dso.granule_num
    ds.attrs["scan_num"] = dso.scan_num

    for varname in variable_dict:
        values_var = dso.groups["product"][varname]
        values = values_var[:].squeeze()
        fv = values_var.getncattr("_FillValue")
        if not np.isfinite(fv):
            values[:][values[:] == fv] = np.nan

        if "scale" in variable_dict[varname]:
            values[:] = variable_dict[varname]["scale"] * values[:]

        if "minimum" in variable_dict[varname]:
            minimum = variable_dict[varname]["minimum"]
            values[:][values[:] < minimum] = np.nan

        if "maximum" in variable_dict[varname]:
            maximum = variable_dict[varname]["maximum"]
            values[:][values[:] > maximum] = np.nan

        ds[varname] = (("x", "y"), values, values_var.__dict__)

        if "quality_flag_max" in variable_dict[varname]:
            ds.attrs["quality_flag"] = varname
            ds.attrs["quality_thresh_max"] = variable_dict[varname]["quality_flag_max"]

    dso.close()

    return ds

    # time_var = dso.groups["geolocation"]["time"]


def apply_quality_flag(ds):
    """Mask variables in place based on quality flag
    Parameters
    ----------
    ds: xr.Dataset
    """
    if "quality_flag" in ds.attrs:
        quality_flag = ds[ds.attrs["quality_flag"]]
        quality_thresh_max = ds.attrs["quality_thresh_max"]

        # Apply the quality thersh maximum to all variables in ds
        for varname in ds:
            if varname != ds.attrs["quality_flag"]:
                logging.debug(varname)
                values = ds[varname].values
                values[quality_flag > quality_thresh_max] = np.nan


def open_dataset(fnames, variable_dict, debug=False):
    if debug:
        logging_level = logging.DEBUG
        logging.basicConfig(strea=sys.stdout, level=logging_level)

    if isinstance(fnames, Path):
        fnames = fnames.as_posix()
    if isinstance(variable_dict, str):
        variable_dict = {variable_dict: {}}
    elif isinstance(variable_dict, dict):
        pass
    else:  # Assume sequence
        variable_dict = {varname: {} for varname in variable_dict}

    for subpath in fnames.split("/"):
        if "$" in subpath:
            envvar = subpath.replace("$", "")
            envval = os.getenv(envvar)
            if envval is None:
                raise Exception("Environment variable not defined: " + subpath)
            else:
                fnames = fnames.replace(subpath, envval)

    print(fnames)
    files = sorted(glob(fnames))

    granules = OrderedDict()

    for file in files:
        granule = _open_one_dataset(file, variable_dict)
        apply_quality_flag(granule)
        granules[granule.attrs["reference_time_string"]] = granule

    return granules
