# RS_and_GIS_with_Python
# Abbreviations; RS = Remote Sensing, GIS = Geographic Information System
# Advantage of the python to do GIS analysis:
#      1. Customizable (i.e., argument can be changed in **args)
#      2. The thought of the programmer in easy form
#      3. Automation is possible through Object oriented programming

# Drawback of the python to do GIS analysis:
#      1. Computation power is not good enough for big data structure
#      2. File of more than 250 MB size can be read by the following code: 
          files = [i for i in geopandas.read_file("tmp.shp", chunksize=10)]
