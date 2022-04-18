#from . import grids, obs,profile,sat,hdf,models
import monetio
__version__ = "0.1"

# point observations
airnow = monetio.obs.airnow
aeronet = monetio.obs.aeronet
aqs = monetio.obs.aqs
cems = monetio.obs.cems
crn = monetio.obs.crn
improve = monetio.obs.improve
ish = monetio.obs.ish
ish_lite = monetio.obs.ish_lite
nadp = monetio.obs.nadp
openaq = monetio.obs.openaq
pams = monetio.obs.pams

# models
fv3chem = monetio.models.fv3chem
cmaq = monetio.models.cmaq
camx = monetio.models.camx
prepchem = monetio.models.prepchem
ncep_grib = monetio.models.ncep_grib
# emitimes = models.emitimes
# cdump2netcdf = models.cdump2netcdf
hysplit = monetio.models.hysplit
hytraj = monetio.models.hytraj
pardump = monetio.models.pardump
raqms = monetio.models.raqms

# profiles
icartt = monetio.profile.icartt
tolnet = monetio.profile.tolnet

# sat
goes = monetio.sat.goes
modis_l2 = monetio.sat.modis_l2
omps_limb = monetio.sat.omps_limb
omps_nadir = monetio.sat.omps_nadir

# hdf
hdfio = monetio.hdf.hdfio


__all__ = ["models", "obs", "sat", "hdf", "util", "grids", "profile", "__version__"]


def rename_latlon(ds):
    """Short summary.

    Parameters
    ----------
    ds : type
        Description of parameter `ds`.

    Returns
    -------
    type
        Description of returned object.

    """
    if "latitude" in ds.coords:
        return ds.rename({"latitude": "lat", "longitude": "lon"})
    elif "Latitude" in ds.coords:
        return ds.rename({"Latitude": "lat", "Longitude": "lon"})
    elif "Lat" in ds.coords:
        return ds.rename({"Lat": "lat", "Lon": "lon"})
    else:
        return ds


def rename_to_monet_latlon(ds):
    """Short summary.

    Parameters
    ----------
    ds : type
        Description of parameter `ds`.

    Returns
    -------
    type
        Description of returned object.

    """
    if "lat" in ds.coords:
        return ds.rename({"lat": "latitude", "lon": "longitude"})
    elif "Latitude" in ds.coords:
        return ds.rename({"Latitude": "latitude", "Longitude": "longitude"})
    elif "Lat" in ds.coords:
        return ds.rename({"Lat": "latitude", "Lon": "longitude"})
    elif "grid_lat" in ds.coords:
        return ds.rename({"grid_lat": "latitude", "grid_lon": "longitude"})
    else:
        return ds


def dataset_to_monet(dset, lat_name="lat", lon_name="lon", latlon2d=False):
    if len(dset[lat_name].shape) != 2:
        latlon2d = False
    if latlon2d is False:
        dset = coards_to_netcdf(dset, lat_name=lat_name, lon_name=lon_name)
    return dset


def coards_to_netcdf(dset, lat_name="lat", lon_name="lon"):
    from numpy import arange, meshgrid

    lon = dset[lon_name]
    lat = dset[lat_name]
    lons, lats = meshgrid(lon, lat)
    x = arange(len(lon))
    y = arange(len(lat))
    dset = dset.rename({lon_name: "x", lat_name: "y"})
    dset.coords["longitude"] = (("y", "x"), lons)
    dset.coords["latitude"] = (("y", "x"), lats)
    dset["x"] = x
    dset["y"] = y
    dset = dset.set_coords(["latitude", "longitude"])
    return dset
