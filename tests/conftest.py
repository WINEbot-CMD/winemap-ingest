import os

# rasterio / pyproj wheels bring their own PROJ and GDAL data; a PROJ_LIB from another install (e.g. PostGIS on
# Windows) breaks EPSG lookups, so ignore it in the tests
for k in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(k, None)
