
# sample command for analyzing an image sequence
python sprout_density_analyzer.py -fold imgs -exp Z229 -sep ph_ -field X03 -manu -thr 95

# making subfolder for storing results
mkdir -p ./imgproc_imgand/allres/

# copying density results into the same folder
cp ./imgproc_imgand/*/resfiles/*sprout_density_imgand.dat ./imgproc_imgand/allres/

# making front velocity analysis on density results
python front_velocity_analysis.py -field X03


