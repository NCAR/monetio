"""Read GEMS L2 NO2 data."""

# ############
# example yaml to call the code
# sel_var=['ColumnAmountNO2Trop','FinalAlgorithmFlags','CloudFraction','SolarZenithAngle','ViewingZenithAngle']
# files=['/glade/campaign/acom/acom-da/sma/GEMS_NO2_V2/202401/30/GK2_GEMS_L2_20240130_0045_NO2_FC-ETC_DPRO_ORI.nc',
#  '/glade/campaign/acom/acom-da/sma/GEMS_NO2_V2/202401/31/GK2_GEMS_L2_20240131_0645_NO2_FW_DPRO_ORI.nc',
#  '/glade/campaign/acom/acom-da/sma/GEMS_NO2_V2/202401/31/GK2_GEMS_L2_20240131_0545_NO2_FW_DPRO_ORI.nc',
#  '/glade/campaign/acom/acom-da/sma/GEMS_NO2_V2/202401/31/GK2_GEMS_L2_20240131_0445_NO2_FW_DPRO_ORI.nc',
#  '/glade/campaign/acom/acom-da/sma/GEMS_NO2_V2/202401/31/GK2_GEMS_L2_20240131_0345_NO2_FC_DPRO_ORI.nc',
#  '/glade/campaign/acom/acom-da/sma/GEMS_NO2_V2/202401/31/GK2_GEMS_L2_20240131_0245_NO2_FC_DPRO_ORI.nc',
#  '/glade/campaign/acom/acom-da/sma/GEMS_NO2_V2/202401/31/GK2_GEMS_L2_20240131_0145_NO2_FC_DPRO_ORI.nc',
#  '/glade/campaign/acom/acom-da/sma/GEMS_NO2_V2/202401/31/GK2_GEMS_L2_20240131_0045_NO2_FC-ETC_DPRO_ORI.nc']
# open_dataset(files, sel_var)


import logging
import os
import sys
from collections import OrderedDict
from glob import glob
import copy

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from datetime import datetime
from scipy.interpolate import griddata


def expand_coord_domain(dataset):
    # creating an extended domain that includes all hours and changes in lon to be able to merge all data into one xarray
    lon_full_1=50
    lon_full_2=150
    diff_lon=.0410367 #looking at original grid spacings

    lat_full_2=50
    lat_full_1=-7
    diff_lat=.085 #looking at original grid spacings
    
    extended_lon = np.arange(lon_full_1, lon_full_2, diff_lon)
    extended_lat = np.arange(lat_full_1, lat_full_2, diff_lat)
     # as through hours the lon domain changes we reindex all data to extended domain 
    dataset = dataset.reindex(lon=extended_lon,lat=extended_lat, method='nearest')
    dataset = dataset.reindex(lon=extended_lon,lat=extended_lat, method='nearest')
    return(dataset)

def filter_dict(dictionary,str):
    filtered_dict = {key: value for key, value in dictionary.items() if str not in key}
    return filtered_dict

def flatten_data(values, lons, lats):
    """
    Check if data is already flattened. If not, flatten it.
    Parameters:
    - values, lons, lats: numpy arrays or array-like objects
    Returns:
    - Tuple of flattened numpy arrays: (values, lons, lats)
    """
   
    values = values.flatten()
    lats = lats.flatten()
    lons = lons.flatten()
    
    return (values, lons, lats)

def create_regirded_data2d(values, lons, lats, new_lats, new_lons,var_name):
    if  values.ndim > 1 or lons.ndim>1:
        values, lons, lats=flatten_data(values,lons,lats)
    gridded_ds = xr.apply_ufunc(interp_to_grid2d,
                 values, lons, lats, new_lats, new_lons,var_name,
                 input_core_dims=[['l'], ['l'],['l'],['lat'],['lon'],[]],
                 output_core_dims=[['lat', 'lon']],
                 dask='parallelized',
                 vectorize=True,
                 output_dtypes=[float],
                 dask_gufunc_kwargs={'output_sizes': {'lat': len(new_lats), 'lon': len(new_lons)}}
                 )  
    return(gridded_ds)

def interp_to_grid2d(u, xc, yc, new_lats, new_lons,var_name):
    new_points = np.stack(np.meshgrid(new_lats, new_lons), axis=2).reshape((new_lats.size * new_lons.size, 2))
    grid_data = griddata((xc, yc), u, (new_points[:, 1], new_points[:, 0]), method='linear', fill_value=np.nan)
    out = grid_data.reshape((new_lats.size, new_lons.size), order="F")
    #display(out)
    del grid_data
    return (out )

def _open_one_dataset(fname, variable_dict):
    """
    Parameters
    ----------
    fname : str
        Input file path.
    variable_dict : dict

    Returns
    -------
    xarray.Dataset
    """
    print("reading " + fname)

    FILE_NAME=fname.strip()
    ds = Dataset(FILE_NAME, 'r')
    grp_keys=list(ds.groups.keys())
    grp=grp_keys[1]     
    grp2=grp_keys[0]     
    lat= ds.groups[grp].variables['Latitude'][:][:]
    lon= ds.groups[grp].variables['Longitude'][:][:]
    _new_lats = np.linspace(lat.min(), lat.max(), np.shape(lat)[1])
    _new_lons = np.linspace(lon.min(), lon.max(), np.shape(lon)[0])
    mask_coord=lat.mask
    #if there is any missing lat and lon that should be excluded
    lat=lat[~mask_coord]
    lon=lon[~mask_coord]
    #creating reqular spacing grids for lat and lon
    new_lons = xr.DataArray(_new_lons, dims="lon", coords={"lon": _new_lons})
    new_lats = xr.DataArray(_new_lats, dims="lat", coords={"lat": _new_lats})
    
    #time date
    date_time_str=('_').join(FILE_NAME.split('/')[-1].split('_')[3:5])
    date_time_obj = datetime.strptime(date_time_str, '%Y%m%d_%H%M')
    
    # Create an empty xarray Dataset
    dataset = xr.Dataset()
    
     # Remove variables from the dictionary of grp ('Geolocation Fields')
    var_geo_f=ds.groups[grp].variables
    filtered_geo_dict = {key: value for key, value in var_geo_f.items() if key in sel_var}
    del var_geo_f
    # Loop through variables in the dictionary to add them to the xarray also include thier meta data
    for var_name, var_data in filtered_geo_dict.items():
        metadata = {attr: getattr(var_data, attr) for attr in var_data.ncattrs()}
        mask_var=var_data[:][:].mask
        var_data=var_data[:][:]
        # replacing all fill vlaue (the mask associated with each data array)
        var_data[mask_var]=np.nan
        # if any lat lon coordinate is missing the data associated with that should be also removed
        var_data=var_data[~mask_coord]
        #var_data=var_data[~mask_var]
        #regriding
        var=create_regirded_data2d(var_data.data, lon, lat, new_lats, new_lons,var_name)
        var.attrs.update(metadata)
        dataset[var_name] = var
        del var, metadata, mask_var
    # sel var which are in grp2 ('Data Fields')
    var_list=list(ds.groups[grp2].variables.keys())
    var_list_sel=[x for x in var_list if x in sel_var]
    for var_name in var_list_sel:
        var_data=ds.groups[grp2].variables[var_name]
        metadata = {attr: getattr(var_data, attr) for attr in var_data.ncattrs()}
        if len(np.shape(var_data[:][:]))==2:
            mask_var=var_data[:][:].mask
            var_data=var_data[:][:]
            if var_name!='FinalAlgorithmFlags':
                var_data[mask_var]=np.nan
               
                
            var_data=var_data[~mask_coord]
            #regriding
            var=create_regirded_data2d(var_data.data, lon, lat, new_lats, new_lons,var_name)
            var.attrs.update(metadata)
            dataset[var_name] = var
            del var, metadata, mask_var
    # #adding global attributes
    dataset['time']=date_time_obj
    dataset = dataset.set_coords(['time'])
    
    # using the global metadata in xarray format
    meta_data=ds.groups['METADATA'].groups['ALGORITHM_SETTINGS']
    ds.close
    # Create an empty dictionary to store attributes
    dict_meta_data = {}
    # Iterate over group attributes and add them to the dictionary
    for attr_name in meta_data.ncattrs():
        dict_meta_data[attr_name] = getattr(meta_data, attr_name)
    constant_meta_data=filter_dict(dict_meta_data,'file_name')
    constant_meta_data
    dataset.attrs.update(constant_meta_data)
    return(dataset)


def open_dataset(fnames, variable_dict):
    for file in fnames:
        ds=_open_one_dataset(file, variable_dict)
        ds=expand_coord_domain(ds)
        if file==fnames[0]:
            merged_dataset=copy.copy(ds)
            day_files=file.split('/')[-1].split('_')[3]
        else:
            merged_dataset = xr.concat([merged_dataset, ds], dim='time')
    merged_dataset.sortby('time')
    return(merged_dataset)

           
