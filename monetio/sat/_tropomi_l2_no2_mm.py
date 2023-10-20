# Reading TROPOMI L2 NO2 data

import os
import sys
import logging
from glob import glob
from collections import OrderedDict

import numpy as np
import xarray as xr

from netCDF4 import Dataset
from datetime import datetime

#from monetio.hdf.hdfio import hdf_open, hdf_close, hdf_list, hdf_read

def read_dataset(fname, variable_dict):
    """
    Parameters
    __________
    fname : str
        Input file path.

    Returns
    _______
    xarray.Dataset
    """
    print('reading ' + fname)

    ds = xr.Dataset()
    dso = Dataset(fname, "r") 
    
    longitude = dso.groups['PRODUCT']['longitude']
    latitude =  dso.groups['PRODUCT']['latitude']
    
    # squeeze 1-dimension
    longitude = np.squeeze(longitude)
    latitude = np.squeeze(latitude)
    
    ds['lon'] = xr.DataArray(longitude)
    ds['lat'] = xr.DataArray(latitude)
        
    for varname in variable_dict:
        print('Reading tropomi l2 no2 data: ', varname)

        # for calculating the vertical pressure
        if varname == 'preslev':
            logging.debug(varname)

            # tropopause index in tm5
            groupname = variable_dict[varname]['tm5_tropopause_layer_index']['group'][0]
            tpindex = dso[groupname]['tm5_tropopause_layer_index']
            tpindex = np.squeeze(tpindex, axis=0)

            # get pleva
            groupname = variable_dict[varname]['tm5_constant_a']['group'][0]
            pleva = dso[groupname]['tm5_constant_a']
            if 'fillvalue' in variable_dict[varname]['tm5_constant_a']:
                fillvalue = variable_dict[varname]['tm5_constant_a']['fillvalue']
                pleva[:][pleva[:] == fillvalue] = np.nan
            if 'maximum' in variable_dict[varname]['tm5_constant_a']:
                maximum = variable_dict[varname]['tm5_constant_a']['maximum']
                pleva[:][pleva[:] > maximum] = np.nan


            # get plevb
            groupname = variable_dict[varname]['tm5_constant_b']['group'][0]
            plevb = dso[groupname]['tm5_constant_b']
            if 'fillvalue' in variable_dict[varname]['tm5_constant_b']:
                fillvalue = variable_dict[varname]['tm5_constant_b']['fillvalue']
                plevb[:][plevb[:] == fillvalue] = np.nan
            if 'maximum' in variable_dict[varname]['tm5_constant_b']:
                maximum = variable_dict[varname]['tm5_constant_b']['maximum']
                plevb[:][plevb[:] > maximum] = np.nan            

            # surface pressure
            groupname = variable_dict[varname]['surface_pressure']['group'][0]
            spre = dso[groupname]['surface_pressure']
            spre = np.squeeze(spre, axis=0)

            if 'fillvalue' in variable_dict[varname]['surface_pressure']:
                fillvalue = variable_dict[varname]['surface_pressure']['fillvalue']
                spre[:][spre[:] == fillvalue] = np.nan
            if 'maximum' in variable_dict[varname]['surface_pressure']:
                maximum = variable_dict[varname]['surface_pressure']['maximum']
                spre[:][spre[:] > maximum] = np.nan 

            # the pressure in the center of vertical layer
            print('Working on TROPOMI NO2 pressure')

            dim0 =  np.shape(spre)[0]
            dim1 =  np.shape(spre)[1]
            dim2 =  np.shape(pleva)[0]
            tpvalue = np.zeros([dim0, dim1], dtype=np.float32)

            preslev = np.zeros([dim0, dim1, dim2], dtype=np.float32)
            preslev[:,:,:] = np.nan
            trpres  = np.zeros([dim0, dim1], dtype=np.float32)
            trpres[:,:] = np.nan

            for ll in range(dim2):
                tpvalue[:,:] = ((pleva[ll,0]+spre[:,:]*plevb[ll,0]) + (pleva[ll,1]+spre[:,:]*plevb[ll,1]))/ 2.0
                preslev[:,:,ll] = tpvalue[:,:]
                ind = np.where(tpindex[:,:] == ll)
                #print('check pres', ll, tpvalue[1642, 10], preslev[1642, 10, ll])
                if (ind[0].size >= 1) & (ll > 0): # to avoid tropopause in the surface layer
                    trpres[ind] = tpvalue[ind]
                
            ds['preslev'] = xr.DataArray(preslev)
            ds['troppres'] = xr.DataArray(trpres)
            

        elif (varname == 'latitude_bounds') | (varname == 'longitude_bounds'):
            groupname = variable_dict[varname]['group'][0]
            lat_bnds = dso[groupname][varname]   # 1x3245x450x4
            lat_bnds = np.squeeze(lat_bnds)      # 3245x450x4
            dims = np.shape(lat_bnds)
            lat_bnds_corners = np.zeros([dims[0], dims[1]], dtype=np.float32)

            for nc in range(4):
                lat_bnds_corners[:,:] = lat_bnds[:,:,nc]
                ds[ varname + '_' + str(nc)] = xr.DataArray(lat_bnds_corners)

        else:
            if 'group' in variable_dict[varname]:
                groupname = variable_dict[varname]['group'][0]
                values = dso[groupname][varname]
            else:
                values = dso.groups['PRODUCT'][varname]

            if min(values.shape) == 1: 
                values = np.squeeze(values)
 
            if 'fillvalue' in variable_dict[varname]:
                fillvalue = variable_dict[varname]['fillvalue']
                values[:][values[:] == fillvalue] = np.nan
        
            if 'scale' in variable_dict[varname]:
                values[:] = variable_dict[varname]['scale'] * values[:]

            if 'minimum' in variable_dict[varname]:
                minimum = variable_dict[varname]['minimum']
                values[:][values[:] < minimum] = np.nan

            if 'maximum' in variable_dict[varname]:
                maximum = variable_dict[varname]['maximum']
                values[:][values[:] > maximum] = np.nan

            ds[varname] = xr.DataArray(values)

            if 'quality_flag_min' in variable_dict[varname]: 
                ds.attrs['quality_flag'] = varname
                ds.attrs['quality_thresh_min'] = variable_dict[varname]['quality_flag_min']
                ds.attrs['var_applied'] = variable_dict[varname]['var_applied']
    
    dso.close()

    return ds


def apply_quality_flag(ds):
    """
    Parameters
    __________
    ds : xarray.Dataset
    """
    if 'quality_flag' in ds.attrs:
        quality_flag = ds[ds.attrs['quality_flag']]
        quality_thresh_min = ds.attrs['quality_thresh_min']
        variable_qf = ds.attrs['var_applied'] # variable applied for the quality flag
        
        # apply the quality thresh minimum to the applied variables in ds
        
        for varname in variable_qf:
            print(varname)
            if varname != ds.attrs['quality_flag']:
                logging.debug(varname)
                values = ds[varname].values
                
                values[quality_flag <= quality_thresh_min] = np.nan
                ds[varname].values = values


def read_trpdataset(fnames, variable_dict, debug=False):
    """
    Parameters
    __________
    fnames : str
        Regular expression for input file paths.

    Returns
    _______
    xarray.Dataset
    """
    from datetime import datetime

    if debug:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(stream=sys.stdout, level=logging_level)

    for subpath in fnames.split('/'):
        if '$' in subpath:
            envvar = subpath.replace('$', '')
            envval = os.getenv(envvar)
            if envval is None:
                print('environment variable not defined: ' + subpath)
                exit(1)
            else:
                fnames = fnames.replace(subpath, envval)

    print(fnames)
    
    files = sorted(glob(fnames))
    granules = OrderedDict()

    for file in files:
        granule = read_dataset(file, variable_dict)
        granule = granule.rename_dims({"dim_0": "x", "dim_1": "y", "dim_2":"z"})
        apply_quality_flag(granule)
        granule_str = file.split('/')[-1]
        granule_info = granule_str.split('____')
        datetime_str = granule_info[1][0:4] + '-' + granule_info[1][4:6] + '-' + granule_info[1][6:8]

        if datetime_str in granules.keys():
            granules[datetime_str].append(granule)
        else:
            granules[datetime_str] = [granule]

    return granules
