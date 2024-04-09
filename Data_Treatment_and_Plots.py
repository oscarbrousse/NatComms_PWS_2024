# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 12:45:00 2022

@author: Oscar Brousse
"""
#%%
import pandas as pd
import fiona
import geopandas as gpd
import numpy as np
from pathlib import Path
import glob
from shapely.geometry import Polygon
from rasterstats import zonal_stats
import rioxarray as rio
from rioxarray import merge
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from matplotlib.colors import Normalize
from matplotlib import cm

from num2words import num2words

#%%
rootdir = r'[YOUR DIRECTORY]'
### The directory where the data provided in the GitHub is provided
### and where the List of Netatmo stations should be saved
datadir = rootdir + 'data/' 
date_range = pd.date_range(start='01-05-2022 00:30:00+00:00', end='01-09-2022 00:30:00+00:00', 
                           freq='1H', inclusive='left')
savedir = rootdir + 'figures/'


# %% 
### Load IMD LSOA to be on the same scales as the others
### Check if England and Wales merged file exists if not build it
if os.path.exists(datadir + 'IMD_2019/IMD_2019.shp')==False:
    en_lsoa = gpd.read_file(datadir + 'raw_data/england_lsoa_imd/Indices_of_Multiple_Deprivation_(IMD)_2019.shp')
    wl_lsoa = gpd.read_file(datadir + 'raw_data/wales_lsoa_imd/wimd2019_overall.shp')
    ### Keep only common attributed
    en_lsoa = en_lsoa[["IMDDec0", "lsoa11cd", "geometry"]].rename(
                        columns={"IMDDec0":"IMDdecile", "lsoa11cd":"lsoa_code"}).to_crs(wl_lsoa.crs)
    wl_lsoa = wl_lsoa[["decile", "lsoa_code", "geometry"]].rename(columns={"decile":"IMDdecile"})
    enwl_lsoa = pd.concat([en_lsoa, wl_lsoa], axis=0, ignore_index=True)
    enwl_lsoa.to_file(datadir + 'IMD_2019/IMD_2019.shp', driver='ESRI Shapefile')
    del wl_lsoa, en_lsoa, enwl_lsoa


imd_lsoa = gpd.read_file(datadir + 'IMD_2019/IMD_2019.shp')

# %%
### lat/lon outline for the UK
### Get boundaries from LSOA shapefile in WGS 84 coordinates for environmental data

enwl = imd_lsoa.to_crs("epsg:4326").dissolve()
## Define boundaries of the box in which data will be gathered
llon, llat, ulon, ulat = enwl.geometry.bounds.values.squeeze()

# %%
### Open list of netatmo locations
cws = pd.read_csv(datadir + 'List_Netatmo_stations_London_ENWL.csv')
cws_gdf = gpd.GeoDataFrame(
    cws,
    geometry=gpd.points_from_xy(cws.Lon, cws.Lat),
    crs="EPSG:4326",
)
cws_loc = [(x,y) for x,y in zip(cws_gdf.geometry.x , cws_gdf.geometry.y)]
# %%
### Open list of MIDAS locations to compare official AWS locations to CWS
### One needs to download the CPAS.DATA file from the CEDA archive: https://data.ceda.ac.uk/badc/ukmo-midas/metadata/CPAS
### See https://help.ceda.ac.uk/article/280-ftp for more information on how to access the data via FTP
### Another option would be to link the SRCC.DATA file to the SRCE.DATA

midas_dir = datadir + 'MIDAS/'
### Valid ID Type (see CEDA doc)
id_type = ['WMO ', 'DCNN']
### Valid met domain (see CEDA doc)
mdom = ['SYNOP   ', 'AWSHRLY ']

col_n = ["SRC_ID","SRC_NAME","ID_TYPE","ID","MET_DOMAIN_NAME",
"SRC_CAP_BGN_DATE","SRC_CAP_END_DATE","PRIME_CAPABILITY_FLAG","RCPT_METHOD_NAME",
"DB_SEGMENT_NAME","DATA_RETENTION_PERIOD","LOC_GEOG_AREA_ID","POST_CODE",
"WMO_REGION_CODE","HIGH_PRCN_LAT","HIGH_PRCN_LON","GRID_REF_TYPE","EAST_GRID_REF",
"NORTH_GRID_REF","ELEVATION","HYDR_AREA_ID","DRAINAGE_STREAM_ID","ZONE_TIME",
"SRC_BGN_DATE","SRC_END_DATE"]
aws_metad = pd.read_csv(midas_dir + 'CPAS.DATA', sep=',', on_bad_lines='skip',
                        names = col_n)
aws_metad = aws_metad[(aws_metad.ID_TYPE.isin(id_type)) & 
                      (aws_metad.MET_DOMAIN_NAME.isin(mdom)) & 
                      (aws_metad.SRC_CAP_END_DATE == "3999-12-31")]
aws_metad = aws_metad.drop_duplicates("SRC_NAME")

aws_gdf = gpd.GeoDataFrame(
    aws_metad,
    geometry=gpd.points_from_xy(aws_metad.HIGH_PRCN_LON, aws_metad.HIGH_PRCN_LAT),
    crs="EPSG:4326",
)

#%%
### Gather the net building heights from GHSL (don't use the gross in that case)
### Open, merge and crop the GHSL data
for file in glob.glob(datadir + 'GHS_BUILT_H_ANBH*.tif'):
    if file==glob.glob(datadir + 'GHS_BUILT_H_ANBH*.tif')[0]:
        ### Store first GHSL file
        bh = rio.open_rasterio(file)
        bh = bh.rio.reproject(cws_gdf.crs)
    ### Open temporaly the other ones and merge them to the first
    else:
        tmp_f = rio.open_rasterio(file)
        tmp_f = tmp_f.rio.reproject(cws_gdf.crs)
        bh = merge.merge_arrays([bh, tmp_f])
        del tmp_f
bh = bh.squeeze(dim='band', drop = True)
### Clip to upper and lower bounds of England and Wales
bh = bh.rio.clip_box(
                            minx=llon,
                            miny=llat,
                            maxx=ulon,
                            maxy=ulat,
)
### Make all values outside ENWL boundaries NaN
bh = bh.rio.clip(enwl.geometry.values, 
                      enwl.crs,
                      drop=False, invert=False)
bh = bh.where(bh!=bh._FillValue)

#%%
### Same as above for built-up fraction
### Open, merge and crop the GHSL data
for file in glob.glob(datadir + 'GHS_BUILT_S*2020*.tif'):
    if file==glob.glob(datadir + 'GHS_BUILT_S*2020*.tif')[0]:
        ### Store first GHSL file
        bf = rio.open_rasterio(file)
        bf = bf.rio.reproject(cws_gdf.crs)
    ### Open temporaly the other ones and merge them to the first
    else:
        tmp_f = rio.open_rasterio(file)
        tmp_f = tmp_f.rio.reproject(cws_gdf.crs)
        bf = merge.merge_arrays([bf, tmp_f])
        del tmp_f
bf = bf.squeeze(dim='band', drop = True)
### Clip to upper and lower bounds of England and Wales
bf = bf.rio.clip_box(
                            minx=llon,
                            miny=llat,
                            maxx=ulon,
                            maxy=ulat,
)
### Make all values outside ENWL boundaries NaN
bf = bf.rio.clip(enwl.geometry.values, 
                      enwl.crs,
                      drop=False, invert=False)
bf = bf.where(bf!=bf._FillValue)

#%%
### Gather Enhanced Vegetation Index (EVI) from MODIS median (2018-2021, included)
evi = rio.open_rasterio(datadir + "MODIS_evi_median.tif")
evi = evi.rio.reproject(cws_gdf.crs)

evi = evi.squeeze(dim='band', drop = True)
### For an obscure reason, one needs to reproject the data to treat it adquatly
### Probably some issues in the GEE export. To be checked 
evi = evi.rio.reproject(bounds = evi.rio.bounds(recalc=True),
                        transform = evi.rio.transform(recalc=True),
                        dst_crs = evi.rio.crs)

### Clip to upper and lower bounds of England and Wales
evi = evi.rio.clip_box(
                            minx=llon,
                            miny=llat,
                            maxx=ulon,
                            maxy=ulat,
)
### Make all values outside ENWL boundaries NaN
evi = evi.rio.clip(enwl.geometry.values, 
                   enwl.crs,
                   drop=False, invert=False)
evi = evi.where(evi!=evi._FillValue)

#%%
### Gather black-sky shortwave albedo from MODIS median (2018-2021, included)
alb = rio.open_rasterio(datadir + "MODIS_MCD43A3_blksky_SW_albedo_median.tif")
alb = alb.rio.reproject(cws_gdf.crs)

alb = alb.squeeze(dim='band', drop = True)
### Clip to upper and lower bounds of England and Wales
alb = alb.rio.reproject(bounds = alb.rio.bounds(recalc=True),
                        transform = alb.rio.transform(recalc=True),
                        dst_crs = alb.rio.crs)
### Make all values outside ENWL boundaries NaN
alb = alb.rio.clip(enwl.geometry.values, 
                      enwl.crs,
                      drop=False, invert=False)
alb = alb.where(alb!=alb._FillValue)

# %%
### Gather the LCZ based on the European LCZ map by Demuzere et al.(2019)
lcz_dir = r"C:/Users/oscar/Documents/Work/WUDAPT/Europe_LCZ_Map_Demuzere2019/"
lcz_eu = rio.open_rasterio(lcz_dir + "EU_LCZ_map_orig.tif")
lcz_eu = lcz_eu.squeeze(dim='band', drop = True)

### Crop the LCZ european map to EN and WL to avoid memory overload
lcz_eu = lcz_eu.rio.clip_box(
        minx=enwl.geometry.to_crs(lcz_eu.rio.crs).bounds.values.squeeze()[0],
        miny=enwl.geometry.to_crs(lcz_eu.rio.crs).bounds.values.squeeze()[1],
        maxx=enwl.geometry.to_crs(lcz_eu.rio.crs).bounds.values.squeeze()[2],
        maxy=enwl.geometry.to_crs(lcz_eu.rio.crs).bounds.values.squeeze()[3],
)
lcz_eu = lcz_eu.where(lcz_eu!=lcz_eu._FillValue)
lcz_eu = lcz_eu.rio.reproject(cws_gdf.crs)

#%%
### Need a common dimension for selecting without duplicating
cws_lon = xr.DataArray(cws_gdf.Lon.values, dims=['loc'])
cws_lat = xr.DataArray(cws_gdf.Lat.values, dims=['loc'])

# %%
### Load Ethnicity data
etn = pd.read_csv(datadir + 'QS201EW_Ethnic_group.csv',
                  skiprows = 8, skipfooter=5, engine = 'python', skip_blank_lines = True)
etn["lsoa_code"] = etn["2011 super output area - lower layer"].str[:9]
etn["EthMinProp"] = 1 - (etn[etn.columns[(etn.columns.str.contains("White:")) & 
                                         (~etn.columns.str.contains("Gypsy"))]
                            ].sum(1).astype(float) / etn['All categories: Ethnic group'].astype(float))
### Keep only the ratio of ethnic minorities against white british
etn = etn[["lsoa_code","EthMinProp"]]
etn.head()

# %% 
### Load population data
pop = pd.read_csv(datadir + 'population.csv')
pop = pop[["LSOA Code","All Ages","65+"]].rename(columns={
                                "LSOA Code":"lsoa_code", "All Ages":"TotPop", "65+":"Pop65+"})

# %%
### Merge ethnicity and population data with IMD and convert to WGS84 projection
imd_lsoa = imd_lsoa.merge(etn, on='lsoa_code')
imd_lsoa = imd_lsoa.merge(pop, on='lsoa_code')
imd_lsoa = imd_lsoa.to_crs(cws_gdf.crs)
imd_lsoa = imd_lsoa.assign(IMDdecile=imd_lsoa.IMDdecile.astype("Int32"))
imd_lsoa = imd_lsoa.assign(
    area_km2=imd_lsoa.to_crs("EPSG:27700").area / 1e6
) 
imd_lsoa["Pop65Prop"] = imd_lsoa["Pop65+"] / imd_lsoa["TotPop"]
imd_lsoa["PopDen"] = imd_lsoa.TotPop / imd_lsoa.area_km2

# %%
### /!\ Careful, this can be MEMORY INTENSIVE
### We focus only on average for now

# lst_stats = ["mean", "std", "median", "max", "min"]
lst_stats = ["mean"]

### Create list of dictionaries of GHSL raster stats per LSOA

# %%

### Building Height LSOA stats
lsoa_bh = zonal_stats(
    vectors=imd_lsoa.geometry,
    raster=bh.values,
    affine=bh.rio.transform(),
    stats=lst_stats,
    nodata=np.nan
)

# %%
### Building surface fraction LSOA stats
lsoa_bf = zonal_stats(
    vectors=imd_lsoa.geometry,
    raster=bf.values/(100**2),
    affine=bf.rio.transform(),
    stats=lst_stats,
    nodata=np.nan
)

# %%
### MODIS enhanced vegetation index (EVI) LSOA stats
lsoa_evi = zonal_stats(
    vectors=imd_lsoa.geometry,
    raster=evi.values,
    affine=evi.rio.transform(),
    stats=lst_stats,
    nodata=np.nan,
    all_touched=True
)

# %%
### MODIS shortwave albedo LSOA stats
lsoa_alb = zonal_stats(
    vectors=imd_lsoa.geometry,
    raster=alb.values,
    affine=alb.rio.transform(),
    stats=lst_stats,
    nodata=np.nan,
    all_touched=True
)

# %%
### Modal LCZ LSOA stats
lsoa_lcz = zonal_stats(
    vectors=imd_lsoa.geometry,
    raster=lcz_eu.values,
    affine=lcz_eu.rio.transform(),
    stats=["majority"],
    nodata=np.nan,
    all_touched=True
)

# %%
### Count of LCZ pixels in each LSOA
lsoa_lcz_c = zonal_stats(
    vectors=imd_lsoa.geometry,
    raster=np.ones_like(lcz_eu.values),
    affine=lcz_eu.rio.transform(),
    stats=["sum"],
    nodata=np.nan,
    all_touched=True
)

#%%
for stat in lst_stats:
    imd_lsoa["bf_" + stat] = [poly.get(stat) for poly in lsoa_bf]
    imd_lsoa["bh_" + stat] = [poly.get(stat) for poly in lsoa_bh]
    imd_lsoa["evi_" + stat] = [poly.get(stat) for poly in lsoa_evi]
    imd_lsoa["alb_" + stat] = [poly.get(stat) for poly in lsoa_alb]
    # imd_lsoa["bh_urb_" + stat] = [poly.get(stat) for poly in lsoa_bh_urb]
imd_lsoa["lcz"] = [poly.get("majority") for poly in lsoa_lcz]
imd_lsoa["lcz_c"] = [poly.get("sum") for poly in lsoa_lcz_c]
imd_lsoa = imd_lsoa[~imd_lsoa.lcz.isna()]

#%%
### Count the amount of pixels of each LCZ and store the fraction per LSOA
### We keep only positive LCZ (remove NaN and locations masked as 0)
for lcz in np.unique(lcz_eu)[1:-1]:
    lsoa_lczx_c = zonal_stats(
        vectors=imd_lsoa.geometry,
        raster=np.where(lcz_eu.values == lcz, 1.0, 0.0),
        affine=lcz_eu.rio.transform(),
        stats=["sum"],
        nodata=np.nan,
        all_touched=True
    )
    imd_lsoa["lcz" + str(int(lcz)) + "_c"] = [poly.get("sum") for poly in lsoa_lczx_c]
    imd_lsoa["lcz" + str(int(lcz)) + "_p"] = imd_lsoa["lcz" + str(int(lcz)) + "_c"] / imd_lsoa.lcz_c


# %%
### Define LCZ colormaps and tick names for plotting
lcz_colors_dict =  {1:'#910613', 2:'#D9081C', 3:'#FF0A22', 4:'#C54F1E', 5:'#FF6628', 6:'#FF985E', 
                    7:'#FDED3F', 8:'#BBBBBB', 9:'#FFCBAB',10:'#565656', 11:'#006A18', 12:'#00A926', 
                    13:'#628432', 14:'#B5DA7F', 15:'#000000', 16:'#FCF7B1', 17:'#656BFA'}
cmap_lcz = mpl.colors.ListedColormap(list(lcz_colors_dict.values()))
lcz_classes = list(lcz_colors_dict.keys()); lcz_classes.append(18)
norm_lcz = mpl.colors.BoundaryNorm(lcz_classes, cmap_lcz.N)
lcz_map = cm.ScalarMappable(norm=norm_lcz, cmap=cmap_lcz)

lcz_labels = ['Compact High Rise: LCZ 1', 'Compact Mid Rise: LCZ 2', 'Compact Low Rise: LCZ 3', 
              'Open High Rise: LCZ 4', 'Open Mid Rise: LCZ 5', 'Open Low Rise: LCZ 6',
              'Lighweight Lowrise: LCZ 7', 'Large Lowrise: LCZ 8',
              'Sparsely Built: LCZ 9', 'Heavy Industry: LCZ 10',
              'Dense Trees: LCZ A', 'Sparse Trees: LCZ B', 'Bush - Scrubs: LCZ C',
              'Low Plants: LCZ D', 'Bare Rock - Paved: LCZ E', 'Bare Soil - Sand: LCZ F',
              'Water: LCZ G']
lcz_labels_dict = dict(zip(list(lcz_colors_dict.keys()),lcz_labels))

# %%
### Join CWS data with LSOA and GHSL data
cws_lsoa = imd_lsoa.sjoin(cws_gdf)
imd_lsoa = (
    imd_lsoa.set_index("lsoa_code")
    .assign(cws=cws_lsoa.lsoa_code.value_counts())
    .reset_index()
)
### Join AWS data with LSOA
aws_lsoa = imd_lsoa.sjoin(aws_gdf.to_crs(imd_lsoa.crs))
imd_lsoa = (
    imd_lsoa.set_index("lsoa_code")
    .assign(aws=aws_lsoa.lsoa_code.value_counts())
    .reset_index()
)

### Estimate the CWS per inhabitants and per km², and presence/absence of CWS and AWS
imd_lsoa = imd_lsoa.assign(
    cws_percap=imd_lsoa.cws / imd_lsoa.TotPop, 
    cws_perkm2=imd_lsoa.cws / imd_lsoa.area_km2,
    cws_perden=imd_lsoa.cws / imd_lsoa.PopDen,
    cws_pres=imd_lsoa.cws.isna(),
    aws_pres=imd_lsoa.aws.isna()
)

imd_lsoa["cws_pres"].replace(False, 'Presence', inplace=True)
imd_lsoa["cws_pres"].replace(True, 'Absence', inplace=True)
imd_lsoa["aws_pres"].replace(False, 'Presence', inplace=True)
imd_lsoa["aws_pres"].replace(True, 'Absence', inplace=True)

### Add a column showing when only CWS is present, only AWS, both or none
imd_lsoa["cws_aws_pres"] = np.nan
imd_lsoa.loc[(imd_lsoa.cws_pres == 'Presence') & 
         (imd_lsoa.aws_pres == 'Presence'), ["cws_aws_pres"]] = "Both"
imd_lsoa.loc[(imd_lsoa.cws_pres == 'Presence') & 
         (imd_lsoa.aws_pres == 'Absence'), ["cws_aws_pres"]] = "PWS"
imd_lsoa.loc[(imd_lsoa.cws_pres == 'Absence') & 
         (imd_lsoa.aws_pres == 'Presence'), ["cws_aws_pres"]] = "AWS"
imd_lsoa.loc[(imd_lsoa.cws_pres == 'Absence') & 
         (imd_lsoa.aws_pres == 'Absence'), ["cws_aws_pres"]] = "Absence"


# %%
### Turning heights and fraction into discrete deciles

imd_lsoa["bh_dec"] = np.nan
imd_lsoa["bf_dec"] = np.nan
imd_lsoa["evi_dec"] = np.nan
imd_lsoa["alb_dec"] = np.nan
imd_lsoa["pop65_dec"] = np.nan
imd_lsoa["etnMin_dec"] = np.nan

imd_lsoa["bh_dec"], bh_dec_bin = pd.qcut(imd_lsoa["bh_mean"], 10,
                               labels = False, retbins=True)
imd_lsoa["bf_dec"], bf_dec_bin = pd.qcut(imd_lsoa["bf_mean"], 10,
                               labels = False, retbins=True)
imd_lsoa["evi_dec"], evi_dec_bin = pd.qcut(imd_lsoa["evi_mean"], 10,
                               labels = False, retbins=True)
imd_lsoa["alb_dec"], alb_dec_bin = pd.qcut(imd_lsoa["alb_mean"], 10,
                               labels = False, retbins=True)
imd_lsoa["pop65_dec"], pop65_dec_bin = pd.qcut(imd_lsoa["Pop65Prop"], 10,
                               labels = False, retbins=True)
imd_lsoa["ethMin_dec"], eth_dec_bin = pd.qcut(imd_lsoa["EthMinProp"], 10,
                               labels = False, retbins=True)

var_dec_bins = [bh_dec_bin, bf_dec_bin, evi_dec_bin, alb_dec_bin, pop65_dec_bin, eth_dec_bin]

# %%
### Relative coverage per IMD Decile and N-counts of people
perc_pop = [imd_lsoa[(~imd_lsoa.cws.isna()) & (imd_lsoa.IMDdecile == imd)].TotPop.sum() / 
            imd_lsoa[(imd_lsoa.IMDdecile == imd)].TotPop.sum() * 100 
            for imd in np.sort(imd_lsoa.IMDdecile.unique())]
counts = [imd_lsoa[(imd_lsoa.IMDdecile == imd)].TotPop.sum()
            for imd in np.sort(imd_lsoa.IMDdecile.unique())]

fig, ax = plt.subplots(figsize = (12,8))
fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.87)

ax.scatter(np.sort(imd_lsoa.IMDdecile.unique()),
                perc_pop,
                color = 'orange',
                marker = 'o',
                s = 200
                )
ax.tick_params('both', labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 12)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('dimgrey')
ax.spines[['left']].set_position(('outward', 70))
ax.spines[['bottom']].set_position(('outward', 10))

imd_label=[num2words(imd, to='ordinal_num') for imd in np.sort(imd_lsoa.IMDdecile.unique())]
imd_label[0] = imd_label[0] + " Decile " "\n" "(Most deprived)"
imd_label[-1] = imd_label[-1] + " Decile " "\n" "(Least deprived)"
ax.set_xticks(np.array(np.sort(imd_lsoa.IMDdecile.unique().astype(int))))
ax.set_xticklabels(imd_label)
# ax.set_xlabel('Index of Multiple Deprivation Deciles', fontsize = 16, color = 'dimgrey',
#               rotation = 0, y = -0.6, x=0.5)

ax.set_ylim(0,25)
ax.set_ylabel('Population covered [%]', fontsize = 16, color = 'dimgrey',
              rotation = 90, y = 0.5, x=-0.5)

### Plot the total population
for imd in np.sort(imd_lsoa.IMDdecile.unique()):
    ax.text(s="{:.2f}".format(imd_lsoa[(imd_lsoa.IMDdecile == imd)].TotPop.sum()/1e6),
            x=imd, y=1,
            horizontalalignment='center',
            color='dimgrey',
    )
ax.text(s="Total" "\n" "population " "\n" "(millions):",
        x=-0.3, y=1,
        color='dimgrey',
    )

fig.suptitle("Personal weather station coverage per deciles of Indice of Multiple Deprivation", 
             x=0.5, y = 0.98,
             fontsize = 20, color = "dimgrey")
fig.savefig(savedir + 'LSOA_PWS_per_IMD.png', dpi = 300)
fig.savefig(savedir + 'LSOA_PWS_per_IMD.pdf')

#%%
### LCZ coverage per total surface (km²) and per population
lcz_n = np.sort(imd_lsoa["lcz"].unique())
lcz_lab = [lcz_labels_dict.get(lcz) for lcz in lcz_n]

per_km2 = [imd_lsoa[(~imd_lsoa.cws.isna()) & (imd_lsoa.lcz == lcz)].cws.count() / 
            imd_lsoa[(imd_lsoa.lcz == lcz)].area_km2.sum()
            for lcz in lcz_n]
totSurf = [imd_lsoa[(imd_lsoa.lcz == lcz)].area_km2.sum()
            for lcz in lcz_n]
per_pop = [imd_lsoa[(~imd_lsoa.cws.isna()) & (imd_lsoa.lcz == lcz)].TotPop.sum() / 
           imd_lsoa[(imd_lsoa.lcz == lcz)].TotPop.sum()
           for lcz in lcz_n]
totPop = [imd_lsoa[(imd_lsoa.lcz == lcz)].TotPop.sum()
            for lcz in lcz_n]

### Create named axes of different sizes
fig, ax = plt.subplot_mosaic([['a'], 
                              ['b']],figsize=(12,18))
fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.92)

for label, ax_i in ax.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax_i.text(-0.17, 1.015, label, transform=ax_i.transAxes + trans,
            fontsize=20, verticalalignment='top', weight='bold', fontfamily='Arial',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    ax_i.tick_params('both', labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 12)
    ax_i.spines[['top', 'right']].set_visible(False)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_color('dimgrey')
    ax_i.spines[['left']].set_position(('outward', 70))
    ax_i.set_xticks(np.arange(len(lcz_n)))
    ax_i.set_xticklabels(lcz_lab, 
                   rotation = 45, rotation_mode = 'anchor', ha='right')


### Stations per km² per LCZ and total surface
ax['a'].scatter(np.arange(len(lcz_n)),
           per_km2,
           color = [lcz_colors_dict.get(lcz) for lcz in lcz_n],
           marker = 'o',
           s = 200)
ax['a'].tick_params('both', labelbottom = False, bottom = False)
ax['a'].set_ylim(-0.05,0.75)
ax['a'].set_ylabel('PWS per km²', fontsize = 16, color = 'dimgrey',
              rotation = 90, y = 0.5, x=-0.5)
ax['a'].spines[['bottom']].set_visible(False)

### Population covered in each LCZ
ax['b'].scatter(np.arange(len(lcz_n)),
           per_pop,
           color = [lcz_colors_dict.get(lcz) for lcz in lcz_n],
           marker = 'o',
           s = 200)
ax['b'].set_ylim(-0.05,0.32)
ax['b'].set_yticks(np.arange(0,0.4,0.1))
ax['b'].set_yticklabels(np.arange(0,40,10))
ax['b'].set_ylabel('Population covered [%]', fontsize = 16, color = 'dimgrey',
              rotation = 90, y = 0.5, x=-0.5)

### Plot the total surface and total population
for i in range(len(lcz_n)):
    ax['a'].text(s="{:.0f}".format(totSurf[i]),
            x=i, y=0.73,
            ha='center',
            color=lcz_colors_dict.get(lcz_n[i]), 
            weight='bold'
    )
    ax['b'].text(s="{:.2f}".format(totPop[i]/1e6),
            x=i, y=0.31,
            ha='center',
            color=lcz_colors_dict.get(lcz_n[i]), 
            weight='bold'
    )

ax['a'].text(s="Total" "\n" "surface (km²): ",
        x=-1.5,y=0.73,
        ha='left',
        color='dimgrey', 
        weight='bold', fontfamily='Arial'
    )
ax['b'].text(s="Total" "\n" "population " "\n" "(millions):",
        x=-1.5,y=0.31,
        ha='left',
        color='dimgrey', 
        weight='bold', fontfamily='Arial'
    )

fig.suptitle("Personal weather stations coverage" "\n" "per area and population in each Local Climate Zone", 
             x=0.5, y = 0.99,
             fontsize = 20, color = "dimgrey")
fig.savefig(savedir + 'LSOA_PWS_per_km2_pop_LCZ.png', dpi = 300)
fig.savefig(savedir + 'LSOA_PWS_per_km2_pop_LCZ.pdf')


# %%
### RELATIVE POPULATION PER DECILES OF EACH METRIC

var_dec = ["bh_dec", "bf_dec", "evi_dec", "alb_dec", "pop65_dec", "ethMin_dec"]
var_labels = ["bh_mean", "bf_mean", "evi_mean", "alb_mean", "Pop65Prop", "EthMinProp"]
var_name = [r"Building Height", r"Building Fraction", r"Enhanced Vegetation Index", 
            r"Shortwave Albedo", r"Population above 65",
            r"Ethnic Minorities"]

col=1
row=len(var_dec)
fig, ax = plt.subplots(row, col, figsize = (8,24))
fig.subplots_adjust(left=0.22, bottom=0.05, right=0.95, top=0.92)
ax=ax.flatten()

dec_label=[num2words(imd, to='ordinal_num') for imd in np.sort(np.unique(imd_lsoa[var_dec].values) + 1)]
dec_label[0] = dec_label[0] + "\n" "(Lowest)"
dec_label[-1] = dec_label[-1] + "\n" "(Highest)"

for i in range(len(var_dec)):
    perc_pop = [imd_lsoa[(~imd_lsoa.cws.isna()) & (imd_lsoa[var_dec[i]] == dec)].TotPop.sum() / 
            imd_lsoa[(imd_lsoa[var_dec[i]] == dec)].TotPop.sum() * 100 
            for dec in np.sort(imd_lsoa[var_dec[i]].unique())]
    
    counts = [imd_lsoa[(imd_lsoa[var_dec[i]] == dec)].TotPop.sum()
            for dec in np.sort(imd_lsoa[var_dec[i]].unique())]
    ax[i].scatter(np.sort(imd_lsoa[var_dec[i]].unique()),
                  perc_pop,
                  color = 'orange',
                  marker = 'o',
                  s = 100
                  )
    ax[i].set_ylim(0,35)
    ax[i].tick_params('both', bottom = False, labelbottom=False, 
                      labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 12)
    ax[i].spines[['top', 'right','bottom']].set_visible(False)
    ax[i].spines[['left']].set_color('dimgrey')
    ax[i].spines[['left']].set_position(('outward', 70))
    ax[i].spines[['bottom']].set_position(('outward', 10))
    ax[i].set_xticks(np.array(np.sort(imd_lsoa[var_dec[i]].unique().astype(int))))
    ax[i].set_xticklabels(dec_label)

    ### Plot the total population
    for dec in np.sort(imd_lsoa[var_dec[i]].unique()):
        ax[i].text(s="{:.2f}".format(imd_lsoa[(imd_lsoa[var_dec[i]] == dec)].TotPop.sum()/1e6),
                x=dec, y=1,
                horizontalalignment='center',
                color='dimgrey',
        )
    if i == range(len(var_dec))[-1]:
        ax[i].text(s="Total" "\n" "population " "\n" "(millions):",
                x=-1.7, y=1,
                color='dimgrey',
            )
    ### Metric name
    ax[i].text(-1.7, 36, var_name[i], 
               fontsize=16, color='dimgrey', ha='left')
    ### Integrate histogram of metric distribution and relative decile cuts
    c, x = np.histogram(imd_lsoa[var_labels[i]], 
         bins=100)
    x = [(x[i+1]+x[i])/2 for i in range(len(x)-1)]
    axhst = ax[i].inset_axes([-1.6, 25, 1.4, 10], transform=ax[i].transData)
    axhst.set_xlim(imd_lsoa[var_labels[i]].min(), imd_lsoa[var_labels[i]].max())
    axhst.fill_between(x, c, color = "grey", linewidth = 1, alpha = 0.5, step='pre')
    axhst.vlines(var_dec_bins[i][1:-1], 
                 ymin=[0]*len(var_dec_bins[i][1:-1]), 
                 ymax=[c[np.absolute(x-xbin).argmin()] for xbin in var_dec_bins[i][1:-1]], 
                 color ='orange', linewidth=2, zorder =1)
    axhst.spines[["bottom", "top", "left", "right"]].set_visible(False)
    axhst.tick_params('both', bottom=False, labelbottom=False,
                       left=False, labelleft=False)

ax[-1].tick_params('both', bottom = True, labelbottom=True, 
                      labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 12)
ax[-1].spines[['bottom']].set_visible(True)
ax[-1].set_xlabel('Deciles of each metric', fontsize = 16, color = 'dimgrey',
                  rotation = 0, y = -0.6, x=0.5)
ax[-1].set_ylabel('Population covered [%]', fontsize = 16, color = 'dimgrey',
                rotation = 90, y = 0.5, x=-0.5)

fig.suptitle("Personal weather station coverage" "\n" "per deciles of environmental and demographic metrics", 
             x=0.5, y = 0.98,
             fontsize = 20, color = "dimgrey")
fig.savefig(savedir + 'LSOA_PWS_per_covariate.png', dpi = 300)
fig.savefig(savedir + 'LSOA_PWS_per_covariate.pdf')


# %%
### VERTICAL HISTOGRAMS PER IMD DECILE
var_dec_labels = ["bh_dec", "bf_dec", "evi_dec", "alb_dec", "pop65_dec", "ethMin_dec"]
var_labels = ["bh_mean", "bf_mean", "evi_mean", "alb_mean", "Pop65Prop", "EthMinProp"]
var_name = [r"Building Height", r"Building Fraction", r"Enhanced Vegetation Index", 
            r"Shortwave Albedo", r"Population above 65",
            r"Ethnic Minorities"]
var_dic = dict(zip(var_dec_labels, var_name))

imd_label=[num2words(imd, to='ordinal_num') + " IMD Decile" for imd in np.sort(imd_lsoa.IMDdecile.unique())]
imd_label[0] = imd_label[0] + "\n" "(Most deprived)"
imd_label[-1] = imd_label[-1] + "\n" "(Least deprived)"

row = len(var_dec_labels)
col = len(imd_lsoa.IMDdecile.unique())
fig, ax = plt.subplots(row, col, figsize = (48,24))
ax = ax.flatten()
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.88)
i=0
for var in var_dec_labels:
    for imd in np.sort(imd_lsoa.IMDdecile.unique()):
        ### Percentage of total population in that IMD per deciles of environmental variable
        ### with a PWS
        perc_var_p = [imd_lsoa[(~imd_lsoa.cws.isna()) & (imd_lsoa.IMDdecile == imd) & 
                         (imd_lsoa[var] == dec)].TotPop.sum() / 
                      imd_lsoa[(imd_lsoa.IMDdecile == imd)].TotPop.sum() * 100
                      for dec in np.sort(imd_lsoa[var].unique())]
        ### Percentage of total population in that IMD per deciles of environmental variable
        ### without a PWS
        perc_var_a = [imd_lsoa[(imd_lsoa.cws.isna()) & (imd_lsoa.IMDdecile == imd) & 
                         (imd_lsoa[var] == dec)].TotPop.sum() / 
                      imd_lsoa[(imd_lsoa.IMDdecile == imd)].TotPop.sum() * 100
                      for dec in np.sort(imd_lsoa[var].unique())]
        ax[i].set_yticks(np.arange(len(perc_var_p)))
        ax[i].set_xlim(0,30)
        ax[i].barh(np.arange(len(perc_var_p)), 
                    perc_var_p,
                    height = 0.75,
                    color = 'orange',
                    label='Presence of personal weather station')
        ax[i].barh(np.arange(len(perc_var_p)), 
                    perc_var_a, left=perc_var_p,
                    height = 0.75, alpha = 0.5, 
                    color = 'dimgrey',
                    label='Absence of personal weather station')
        i+=1

for ax_i in ax:
    ax_i.tick_params('both', labelbottom = False, bottom = False, left = False, labelleft= False,
                     labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 16)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_color('dimgrey')

### Name of the variables
i=0
for ax_i in ax[0::col]:
    ax_i.text(0, 1.03, var_name[i], 
              transform=ax_i.transAxes, fontsize=20, color='dimgrey', ha='left')
    i+=1

ax[-col].tick_params('both', left = True, labelleft = True, bottom = True, labelbottom = True,
                     labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 14)
ax[-col].spines[['left', 'bottom']].set_visible(True)
ax[-col].spines[['left', 'bottom']].set_position(('outward', 10))
ax[-col].set_xlabel('Percentage of population [%]', fontsize = 16, color = 'dimgrey')
ax[-col].text(s='Deciles', fontsize = 16, color = 'dimgrey', 
              transform=ax[-col].transAxes,
              x = -0.3, y = 1.03, 
              rotation=0)
ax[-col].set_yticks([ax[-col].get_yticks()[i] for i in (0,-1)])
ax[-col].set_yticklabels(['Lowest ($1^{st}$)', 'Highest ($10^{th}$)'])

### Plot arrow indicating from lowest to highest decile
ax_ar = plt.axes([0.1, 0.915, 0.78, 0.01])
ax_ar.arrow(x=0, y=0.5, dx=1, dy=0, 
            width=.002, head_width=0.015, head_length=0.03, fc='dimgrey', ec='dimgrey')
ax_ar.tick_params('both', labelbottom = False, bottom = False, left = False, labelleft= False)
ax_ar.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
ax_ar.spines[['top', 'right', 'bottom', 'left']].set_position(('outward', 0))
fig.text(0.08, 0.92, imd_label[0], fontsize=24, color='dimgrey', ha='center', va='center')
fig.text(0.90, 0.92, imd_label[col-1], fontsize=24, color='dimgrey', ha='center', va='center')

### Plot legend of presence absence
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, 
           fontsize = 24, labelcolor = 'dimgrey', edgecolor = None, frameon=False)

fig.suptitle("Population spread across deciles of different demographic and " + 
             "environmental characteristics for each decile of the Indice of Multiple Deprivation", 
             y = 0.98,
             fontsize = 40, color = "dimgrey")
fig.savefig(savedir + 'LSOA_charac_dec_IMD_PWS_pres.png', dpi = 300)
fig.savefig(savedir + 'LSOA_charac_dec_IMD_PWS_pres.pdf')

# %%
### TABLE OF AVERAGE METRIC VALUES PER IMD AND PRESENCE/ABSENCE
var_labels = ["bh_mean", "bf_mean", "evi_mean", "alb_mean", "Pop65Prop", "EthMinProp"]
var_dic = dict(zip(var_dec_labels, var_name))

imd_label=[num2words(imd, to='ordinal_num') + " Decile" for imd in np.sort(imd_lsoa.IMDdecile.unique())]
imd_label[0] = imd_label[0] + " (Most deprived)"
imd_label[-1] = imd_label[-1] + " (Least deprived)"

row = len(var_dec_labels)
col = len(imd_lsoa.IMDdecile.unique())

### List of metrics to be calculated (Average Presence, Average Absence and Perkins Skill Score)
metrics = [r"$\overline{BH_{pws}}$", r"$\overline{BH_{abs}}$", r"$BH_{PSS}$",
           r"$\overline{BF_{pws}}$", r"$\overline{BF_{abs}}$", r"$BF_{PSS}$",
           r"$\overline{EVI_{pws}}$", r"$\overline{EVI_{abs}}$", r"$EVI_{PSS}$",
           r"$\overline{Alb_{pws}}$", r"$\overline{Alb_{abs}}$", r"$Alb_{PSS}$",
           r"$\overline{P65_{pws}}$", r"$\overline{P65_{abs}}$", r"$P65_{PSS}$",
           r"$\overline{Eth_{pws}}$", r"$\overline{Eth_{abs}}$", r"$Eth_{PSS}$"]
df_tab1 = pd.DataFrame(index = metrics, columns = imd_label)

i=0
for var in var_labels:
    for imd in np.sort(imd_lsoa.IMDdecile.unique()):
        ### Average metric in places where PWS are present
        x_avg_p=str(np.around(
                        imd_lsoa[(~imd_lsoa.cws.isna()) & (imd_lsoa.IMDdecile == imd)][var].mean(),
                                    decimals = 2)
                            )
        ### Average metric in places where PWS are absent
        x_avg_a=str(np.around(
                        imd_lsoa[(imd_lsoa.cws.isna()) & (imd_lsoa.IMDdecile == imd)][var].mean(),
                                    decimals = 2)
                            )
        ### Perkins skill score
        p_p, bins = np.histogram(imd_lsoa[(~imd_lsoa.cws.isna()) & (imd_lsoa.IMDdecile == imd)][var], 
                              bins=50, 
                              weights=imd_lsoa[(~imd_lsoa.cws.isna()) & (imd_lsoa.IMDdecile == imd)]["TotPop"], 
                              density=True)
        p_a, bins_a = np.histogram(imd_lsoa[(imd_lsoa.cws.isna()) & (imd_lsoa.IMDdecile == imd)][var], 
                              bins=bins, 
                              weights=imd_lsoa[(imd_lsoa.cws.isna()) & (imd_lsoa.IMDdecile == imd)]["TotPop"], 
                              density=True)
        ### Bins centers
        bins_c = [bins[i+1]-bins[i] for i in range(len(bins)-1)]
        PSS = np.sum(np.minimum(p_p, p_a)*(bins_c))
        df_tab1.iloc[i,imd-1] = x_avg_p
        df_tab1.iloc[i+1,imd-1] = x_avg_a
        df_tab1.iloc[i+2,imd-1] = np.around(PSS, decimals = 2)
    i+=3

df_tab1.to_csv(savedir + "Table_1.csv")

# %%
### Map all the acquired data
var_labels = ["bh_mean", "bf_mean", "evi_mean", "alb_mean", "Pop65Prop", "EthMinProp"]

### GeoPandas does not support categorical definition of colors
clr_abspres = {'Absence': 'grey', 'Presence':'orange'}
imd_lsoa["pres_clrs"] = imd_lsoa["cws_pres"].map(clr_abspres)

### Create named axes of different sizes: focus on Presence and IMD
fig, ax = plt.subplot_mosaic([['a', 'b', 'c'], 
                              ['d', 'e', 'f'], 
                              ['g', 'h', 'i']],
                              constrained_layout=False, figsize=(12,16))
fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.95)

for label, ax_i in ax.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax_i.text(0.0, 1.0, label, transform=ax_i.transAxes + trans,
            fontsize=12, verticalalignment='top', weight='bold', fontfamily='Arial',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    ax_i.tick_params('both', labelbottom = False, bottom = False, left = False, labelleft= False,
                     labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 16)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_color('dimgrey')

### Plot presence/absence per LSOA
imd_lsoa.plot(color = imd_lsoa["pres_clrs"],
    edgecolor = None, missing_kwds={'color': 'lightgrey'}, 
    ax=ax['a']
)

### Strategy around obtaining legend for categorical values
### See https://github.com/geopandas/geopandas/issues/2279
custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=color) 
                    for color in clr_abspres.values()]
ax['a'].legend(custom_points, clr_abspres.keys(), ncol=2, loc='lower center',
               fontsize = 10, labelcolor = 'dimgrey', edgecolor = None, frameon=False,
               bbox_to_anchor=(0.6, -0.03))

### Plot IMD deciles
imd_lsoa.plot(column = imd_lsoa["IMDdecile"].astype(float), cmap = cm.get_cmap('cividis', 10),
    edgecolor = None, missing_kwds={'color': 'lightgrey'},
    ax=ax['b']
)
mn = imd_lsoa.IMDdecile.min()
mx = imd_lsoa.IMDdecile.max()
norm = Normalize(vmin=mn, vmax=mx)
n_cmap = cm.ScalarMappable(norm=norm, cmap = cm.get_cmap('cividis', 10))
cax = inset_axes(ax['b'], width="90%", height="2%", loc='lower right') 
cbar = ax['b'].get_figure().colorbar(n_cmap, cax=cax, orientation='horizontal',
                              ticks=np.arange(1.45,10,0.9))
cbar.ax.tick_params(axis = "both", which='both', 
                    bottom=False, labelcolor = 'dimgrey', labelsize = 10)
cbar.ax.set_xticklabels([str(i) for i in np.arange(1,11,1)])

### Plot LCZ

lcz_short_labels = ['1','2','3','4','5','6','7','8','9','10',
              'A','B','C','D','E','F','G']
imd_lsoa.plot(column = imd_lsoa["lcz"], 
        cmap = cmap_lcz, vmin = 0, vmax = 18,
        edgecolor = None, missing_kwds={'color': 'lightgrey'},
        ax=ax['c'])
cax = inset_axes(ax['c'], width="90%", height="2%", loc='lower right') 
cbar = ax['c'].get_figure().colorbar(lcz_map, cax=cax, orientation='horizontal',
                              ticks = np.arange(1.5,18.5,1))
cbar.ax.tick_params(axis = "x", which='both', 
                    bottom=False, labelcolor = 'dimgrey', labelsize = 10)
cbar.ax.set_xticklabels(lcz_short_labels)

### Plot all remaining environmental and social variables
lst_ax = ['d', 'e', 'f', 'g', 'h', 'i']

for i in np.arange(len(lst_ax)):
    imd_lsoa.plot(
        var_labels[i], cmap = cm.get_cmap('cividis'),
    missing_kwds={'color': 'lightgrey'}, 
    ax=ax[lst_ax[i]]
    )
    mn = imd_lsoa[var_labels[i]].min()
    mx = imd_lsoa[var_labels[i]].max()
    norm = Normalize(vmin=mn, vmax=mx)
    n_cmap = cm.ScalarMappable(norm=norm, cmap = cm.get_cmap('cividis'))
    cax = inset_axes(ax[lst_ax[i]], width="90%", height="2%", loc='lower right') 
    cbar = ax[lst_ax[i]].get_figure().colorbar(n_cmap, cax=cax, orientation='horizontal',
                              ticks=([mn, mx]))
    cbar.ax.set_xticklabels([str(mn)[:4],str(mx)[:4]])
    cbar.ax.tick_params(axis = "both", labelcolor = 'dimgrey', labelsize = 12)

ax['g'].spines[['bottom', 'left']].set_visible(True)
ax['g'].spines[['bottom', 'left']].set_position(('outward', 15))
ax['g'].tick_params('both', labelbottom = True, bottom = True, left = True, labelleft= True,
                     labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 14)
ax['g'].set_xlabel("Longitude", color = 'dimgrey', fontsize=16)
ax['g'].set_ylabel("Latitude", color = 'dimgrey', fontsize=16)

### Add titles to each subplot
lst_titles = [r"Personal weather" "\n" " station coverage", r"Indices of Multiple Deprivation " "\n" "(Deciles)", r"Local Climate Zones",
              r"Building Height (m)", r"Building Fraction", r"Enhanced" "\n" " Vegetation Index", 
              r"Shortwave Albedo", r"Proportion of " "\n" "population above 65",
              r"Ethnic" "\n" "Minorities Proportion"]
i = 0
for ax_i in ax.values():
    ax_i.set_title(lst_titles[i], color='dimgrey', fontsize = 18)     
    i+=1

fig.savefig(savedir + 'LSOA_maps.png', dpi = 600)
fig.savefig(savedir + 'LSOA_maps.pdf')

# %%
### Distributions of metrics and deciles cut-offs
var_labels = ["bh_mean", "bf_mean", "evi_mean", "alb_mean", "Pop65Prop", "EthMinProp"]
var_name = [r"Building Height [m]", r"Building Fraction", r"Enhanced Vegetation Index", 
            r"Shortwave Albedo", r"Ratio of population above 65",
            r"Ratio of ethnic Minorities"]

col=1
row=len(var_dec)
fig, ax = plt.subplots(row, col, figsize = (8,24))
fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.92)
ax=ax.flatten()

for i in range(len(var_labels)):
    ### Histogram of metric distribution and relative decile cuts
    c, x = np.histogram(imd_lsoa[var_labels[i]], 
         bins=100)
    x = [(x[i+1]+x[i])/2 for i in range(len(x)-1)]
    ax[i].set_xlim(imd_lsoa[var_labels[i]].min(), imd_lsoa[var_labels[i]].max())
    ax[i].fill_between(x, c, color = "grey", linewidth = 1, alpha = 0.5, step='pre')
    ax[i].vlines(var_dec_bins[i][1:-1], 
                 ymin=[0]*len(var_dec_bins[i][1:-1]), 
                 ymax=[c[np.absolute(x-xbin).argmin()] for xbin in var_dec_bins[i][1:-1]], 
                 color ='orange', linewidth=2, zorder =1)
    ax[i].spines[["top", "right"]].set_visible(False)
    ax[i].tick_params('both', bottom=True, labelbottom=True,
                       left=True, labelleft=True, colors = 'dimgrey', labelsize = 12)
    ax[i].spines[['left', 'bottom']].set_color('dimgrey')
    ax[i].spines[['left']].set_position(('outward', 5))
    ax[i].spines[['bottom']].set_position(('outward', 5))
    ax[i].set_xticks(np.array([imd_lsoa[var_labels[i]].min(), 
                               imd_lsoa[var_labels[i]].max()]))
    ax[i].set_yticks(np.array([np.min(c), np.max(c)]))
    ax[i].set_xlabel(var_name[i], fontsize = 12, color = 'dimgrey', x=0.5, y=1)

ax[-1].set_ylabel('Counts', fontsize = 12, color = 'dimgrey',
                rotation = 90, y = 0.5)

fig.suptitle("Histograms of environmental and demographic metrics", 
             x=0.5, y = 0.95,
             fontsize = 16, color = "dimgrey")
fig.savefig(savedir + 'LSOA_covariate_hist.png', dpi = 300)
fig.savefig(savedir + 'LSOA_covariate_hist.pdf')


# %%
### GeoPandas does not support categorical definition of colors
clr_abspres = {'Absence': 'grey', 'PWS':'orange', 'AWS':'gold', 'Both':'darkred'}
imd_lsoa["pres_awscws_clrs"] = imd_lsoa["cws_aws_pres"].map(clr_abspres)

### Create named axes of different sizes: focus on Presence and IMD
fig, ax = plt.subplots(figsize = (12,8))
fig.subplots_adjust(left=0.10, bottom=0.1, right=0.90, top=0.90)
ax.tick_params('both', labelbottom = True, bottom = True, left = True, labelleft= True,
                    labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 16)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['top', 'right', 'bottom', 'left']].set_color('dimgrey')

### Plot presence/absence per LSOA
imd_lsoa.plot(color = imd_lsoa["pres_awscws_clrs"],
    facecolor="lightgrey", edgecolor="none", 
    ax=ax
)

### Strategy around obtaining legend for categorical values
### See https://github.com/geopandas/geopandas/issues/2279
custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=color) 
                    for color in clr_abspres.values()]
ax.legend(custom_points, clr_abspres.keys(), ncol=2, loc='lower center',
               fontsize = 10, labelcolor = 'dimgrey', edgecolor = None, frameon=False,
               bbox_to_anchor=(0.6, -0.03))

ax.spines[['bottom', 'left']].set_position(('outward', 15))
ax.set_xlabel("Longitude", color = 'dimgrey', fontsize=16)
ax.set_ylabel("Latitude", color = 'dimgrey', fontsize=16)

fig.suptitle(r"Presence or absence of: Met Office MIDAS automatic weather station (AWS)," "\n"
             "Netatmo personal weather stations (PWS), or both",
             x=0.5, y = 0.97,
             fontsize = 16, color = "dimgrey")
fig.savefig(savedir + 'LSOA_AWS_PWS_cov.png', dpi = 300)
fig.savefig(savedir + 'LSOA_AWS_PWS_cov.pdf')
# %%
### Relative coverage of AWS MIDAS per IMD Decile and N-counts of people
perc_pop = [imd_lsoa[(~imd_lsoa.aws.isna()) & (imd_lsoa.IMDdecile == imd)].TotPop.sum() / 
            imd_lsoa[(imd_lsoa.IMDdecile == imd)].TotPop.sum() * 100 
            for imd in np.sort(imd_lsoa.IMDdecile.unique())]
counts = [imd_lsoa[(imd_lsoa.IMDdecile == imd)].TotPop.sum()
            for imd in np.sort(imd_lsoa.IMDdecile.unique())]

fig, ax = plt.subplots(figsize = (12,8))
fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.87)

ax.scatter(np.sort(imd_lsoa.IMDdecile.unique()),
                perc_pop,
                color = 'orange',
                marker = 'o',
                s = 200
                )
ax.tick_params('both', labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 12)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_color('dimgrey')
ax.spines[['left']].set_position(('outward', 70))
ax.spines[['bottom']].set_position(('outward', 10))

imd_label=[num2words(imd, to='ordinal_num') for imd in np.sort(imd_lsoa.IMDdecile.unique())]
imd_label[0] = imd_label[0] + " Decile " "\n" "(Most deprived)"
imd_label[-1] = imd_label[-1] + " Decile " "\n" "(Least deprived)"
ax.set_xticks(np.array(np.sort(imd_lsoa.IMDdecile.unique().astype(int))))
ax.set_xticklabels(imd_label)

ax.set_ylim(0,1.5)
ax.set_ylabel('Population covered [%]', fontsize = 16, color = 'dimgrey',
              rotation = 90, y = 0.5, x=-0.5)

### Plot the total population
for imd in np.sort(imd_lsoa.IMDdecile.unique()):
    ax.text(s="{:.2f}".format(imd_lsoa[(imd_lsoa.IMDdecile == imd)].TotPop.sum()/1e6),
            x=imd, y=1.4,
            horizontalalignment='center',
            color='dimgrey',
    )
ax.text(s="Total" "\n" "population " "\n" "(millions):",
        x=-0.3, y=1.4,
        color='dimgrey',
    )

fig.suptitle("MIDAS automatic weather station coverage per deciles of Indice of Multiple Deprivation", 
             x=0.5, y = 0.98,
             fontsize = 18, color = "dimgrey")
fig.savefig(savedir + 'LSOA_AWS_per_IMD.png', dpi = 300)
fig.savefig(savedir + 'LSOA_AWS_per_IMD.pdf')
# %%
### Aggregate at larger NUTS level to see how results can be transferrable
nuts_gdf = gpd.read_file(datadir + 'NUTS_RG_20M_2021_4326/NUTS_RG_20M_2021_4326.shp')
nuts_gdf = nuts_gdf.to_crs(imd_lsoa.crs)
nuts2 = nuts_gdf[nuts_gdf.LEVL_CODE == 2].clip(imd_lsoa)
nuts3 = nuts_gdf[nuts_gdf.LEVL_CODE == 3].clip(imd_lsoa)

cws_nuts2 = nuts2.sjoin(cws_gdf)
nuts2 = (
    nuts2.set_index("NUTS_ID")
    .assign(cws=cws_nuts2.NUTS_ID.value_counts())
    .reset_index()
)
nuts2 = nuts2.assign(
    cws_pres=nuts2.cws.isna()
)
nuts2["cws_pres"].replace(False, 'Presence', inplace=True)
nuts2["cws_pres"].replace(True, 'Absence', inplace=True)

cws_nuts3 = nuts3.sjoin(cws_gdf)
nuts3 = (
    nuts3.set_index("NUTS_ID")
    .assign(cws=cws_nuts3.NUTS_ID.value_counts())
    .reset_index()
)
nuts3 = nuts3.assign(
    cws_pres=nuts3.cws.isna()
)
nuts3["cws_pres"].replace(False, 'Presence', inplace=True)
nuts3["cws_pres"].replace(True, 'Absence', inplace=True)

### GeoPandas does not support categorical definition of colors
clr_abspres = {'Absence': 'grey', 'Presence':'orange'}
nuts2["pres_clrs"] = nuts2["cws_pres"].map(clr_abspres)
nuts3["pres_clrs"] = nuts3["cws_pres"].map(clr_abspres)

### Plot the data
fig, ax = plt.subplot_mosaic([['a', 'b']],
                              constrained_layout=False, figsize=(12,8))
fig.subplots_adjust(left=0.10, bottom=0.1, right=0.90, top=0.90)

for label, ax_i in ax.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax_i.text(0.0, 1.0, label, transform=ax_i.transAxes + trans,
            fontsize=12, verticalalignment='top', weight='bold', fontfamily='Arial',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    ax_i.tick_params('both', labelbottom = False, bottom = False, left = False, labelleft= False,
                     labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 16)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_color('dimgrey')

nuts2.plot(color = nuts2["pres_clrs"],
    facecolor="lightgrey", edgecolor="darkgrey",
    ax=ax['a']
)
nuts3.plot(color = nuts3["pres_clrs"],
    facecolor="lightgrey", edgecolor="darkgrey",
    ax=ax['b']
)

### Strategy around obtaining legend for categorical values
### See https://github.com/geopandas/geopandas/issues/2279
custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=color) 
                    for color in clr_abspres.values()]
ax['a'].legend(custom_points, clr_abspres.keys(), ncol=2, loc='lower center',
               fontsize = 10, labelcolor = 'dimgrey', edgecolor = None, frameon=False,
               bbox_to_anchor=(0.6, -0.03))

ax['a'].spines[['bottom', 'left']].set_visible(True)
ax['a'].tick_params('both', labelbottom = True, bottom = True, left = True, labelleft= True)
ax['a'].spines[['bottom', 'left']].set_position(('outward', 15))
ax['a'].set_xlabel("Longitude", color = 'dimgrey', fontsize=16)
ax['a'].set_ylabel("Latitude", color = 'dimgrey', fontsize=16)

fig.suptitle(r"Presence or absence of Netatmo personal weather stations (PWS)" "\n" 
             "at NUTS 2 and NUTS 3 adminsitrative levels",
             x=0.5, y = 0.97,
             fontsize = 16, color = "dimgrey")
fig.savefig(savedir + 'NUTS_PWS_cov.png', dpi = 300)
fig.savefig(savedir + 'NUTS_PWS_cov.pdf')

# %%
### Map the proportions of each LCZ
### Create named axes of different sizes: focus on Presence and IMD
fig, ax = plt.subplot_mosaic([['a', 'b', 'c', 'd', 'e'], 
                              ['f', 'g', 'h', 'i', 'j'], 
                              ['k', 'l', 'm', 'n', 'o']],
                              constrained_layout=False, figsize=(24,16))
fig.subplots_adjust(left=0.10, bottom=0.1, right=0.90, top=0.90)

lcz_unique = np.unique(lcz_eu.values)[1:-1]

i = 0
for label, ax_i in ax.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax_i.text(0.0, 1.0, label, transform=ax_i.transAxes + trans,
            fontsize=12, verticalalignment='top', weight='bold', fontfamily='Arial',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    ax_i.tick_params('both', labelbottom = False, bottom = False, left = False, labelleft= False,
                     labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 16)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_color('dimgrey')
    imd_lsoa.plot(
        "lcz" + str(int(lcz_unique[i])) + "_p", cmap = cm.get_cmap('cividis'),
    missing_kwds={'color': 'lightgrey'}, 
    ax=ax_i
    )
    mn = imd_lsoa["lcz" + str(int(lcz_unique[i])) + "_p"].min()
    mx = imd_lsoa["lcz" + str(int(lcz_unique[i])) + "_p"].max()
    norm = Normalize(vmin=mn, vmax=mx)
    n_cmap = cm.ScalarMappable(norm=norm, cmap = cm.get_cmap('cividis'))
    cax = inset_axes(ax_i, width="90%", height="2%", loc='lower right') 
    cbar = ax_i.get_figure().colorbar(n_cmap, cax=cax, orientation='horizontal',
                              ticks=([mn, mx]))
    cbar.ax.set_xticklabels([str(mn)[:4],str(mx)[:4]])
    cbar.ax.tick_params(axis = "both", labelcolor = 'dimgrey', labelsize = 12)
    i+=1
    
ax['k'].spines[['bottom', 'left']].set_visible(True)
ax['k'].spines[['bottom', 'left']].set_position(('outward', 15))
ax['k'].tick_params('both', labelbottom = True, bottom = True, left = True, labelleft= True,
                     labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 14)
ax['k'].set_xlabel("Longitude", color = 'dimgrey', fontsize=16)
ax['k'].set_ylabel("Latitude", color = 'dimgrey', fontsize=16)

### Add titles to each subplot
lst_titles = ["LCZ 2", "LCZ 3", "LCZ 4", "LCZ 5", "LCZ 6",
              "LCZ 8", "LCZ 9", "LCZ 10", "LCZ A", "LCZ B",
              "LCZ C", "LCZ D", "LCZ E", "LCZ F", "LCZ G"]
i = 0
for ax_i in ax.values():
    ax_i.set_title(lst_titles[i], color='dimgrey', fontsize = 18)     
    i+=1

fig.suptitle(r"Proportion of each LCZ per LSOA",
             x=0.5, y = 0.97,
             fontsize = 22, color = "dimgrey")
fig.savefig(savedir + 'LSOA_LCZp_maps.png', dpi = 600)
fig.savefig(savedir + 'LSOA_LCZp_maps.pdf')

# %%
### Plot the cumulative distribution functions of each proportion of LCZ per modal LCZ

lcz_mod = np.sort(imd_lsoa.lcz.unique())
fig, ax = plt.subplot_mosaic([['a', 'b', 'c', 'd'], 
                              ['e', 'f', 'g', 'h'], 
                              ['i', 'j', 'k', 'l']],
                              constrained_layout=False, figsize=(24,16))
fig.subplots_adjust(left=0.10, bottom=0.1, right=0.90, top=0.90)

### Create temporary objects to plot the legend
for lcz in np.unique(lcz_eu.values)[1:-1]:
    ax['l'].plot(imd_lsoa["lcz" + str(int(lcz)) + "_p"], label = lcz_labels_dict.get(lcz),
                 color=lcz_colors_dict.get(lcz))

handles, labels = ax['l'].get_legend_handles_labels()
ax['l'].cla() 
ax['l'].remove()
i=0
for label, ax_i in ax.items():
    if label == "l":
        break
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax_i.text(0.0, 1.1, label, transform=ax_i.transAxes + trans,
            fontsize=12, verticalalignment='top', weight='bold', fontfamily='Arial',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    ax_i.tick_params('both', labelbottom = False, bottom = False, left = False, labelleft= False,
                     labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 16)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax_i.spines[['top', 'right', 'bottom', 'left']].set_color('dimgrey')
    
    for lcz in np.unique(lcz_eu.values)[1:-1]:
        if lcz == lcz_mod[i]:
            continue
        ### We limit the bins to 0.5 as past 0.5 one single LCZ is necessarily in majority
        count, bins = np.histogram(imd_lsoa.loc[imd_lsoa.lcz == lcz_mod[i],
                                                "lcz" + str(int(lcz)) + "_p"]*100, 
                                                bins=50, range=(0.0,50.0))
        bins_ticks = np.arange(bins.min() + (bins[1] - bins[0]) / 2, 
                            bins.max(), 
                            bins[1] - bins[0])
        # finding the PDF of the histogram using count values
        pdf = count / np.sum(count)

        # using numpy np.cumsum to calculate the CDF
        # We can also find using the PDF values by looping and adding
        cdf = np.cumsum(pdf)*100

        ax_i.plot(bins_ticks, cdf, label = lcz_labels_dict.get(lcz),
                color = lcz_colors_dict.get(lcz),
                zorder = 0, linewidth = 3)
    ax_i.set_xlim(0.0,50.0)
    ax_i.set_ylim(0.0,100.0)
    ax_i.vlines(100/3, 0, 100, linewidth = 2, linestyles='--', color = 'dimgrey')
    ax_i.hlines(50, 0, 100, linewidth = 2, linestyles='--', color = 'dimgrey')
    ax_i.hlines(80, 0, 100, linewidth = 4, linestyles='--', color = 'dimgrey')


    ax_i.set_title(lcz_labels_dict.get(lcz_mod[i]), color='dimgrey', fontsize = 18)

    i+=1

ax['i'].spines[['bottom', 'left']].set_visible(True)
ax['i'].spines[['bottom', 'left']].set_position(('outward', 15))
ax['i'].tick_params('both', labelbottom = True, bottom = True, left = True, labelleft= True,
                     labelcolor = 'dimgrey', colors = 'dimgrey', labelsize = 14)
ax['i'].set_xlabel("Proportion of the LSOA [%]", color = 'dimgrey', fontsize=16)
ax['i'].set_ylabel("Cumulative probability [%]", color = 'dimgrey', fontsize=16)

fig.suptitle(r"Cumulative density function of proportions of LCZ per modal LCZ at the LSOA level",
             x=0.5, y = 0.97,
             fontsize = 20, color = "dimgrey")
fig.legend(handles, labels, bbox_to_anchor=(0.87, 0.36), ncol=1, 
           fontsize = 14, labelcolor = 'dimgrey', edgecolor = None, frameon=False)

fig.savefig(savedir + 'CDF_LCZp.png', dpi = 600)
fig.savefig(savedir + 'CDF_LCZp.pdf')

# %%
