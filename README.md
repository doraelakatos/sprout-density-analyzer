
# Sprout Density Analyzer #
## sprout_density_analyzer.py ##

This Python script implements a segmentation algorithm for the sprouting process of gel-embedded endothelial cell aggregates. The segmentation process is optimized for phase contrast images recorded by time-lapse microscopy.


## Overview ##

To identify the initial aggregates, the first frame of each image sequence can be segmented by choosing one of the following methods: 
* Otsu's thresholding, 
* manual thresholding, 
* Hough Circle Transform, 
* adaptive thresholding of ROI. 

After segmentation, aggregates are reconstructed by using basic morphological operations and removal of small connected components. To detect sprout formation, images are preprocessed by extracting the background using a Gaussian mixture-based background segmentation algorithm [1,2]. The resulting foreground image is blended with the reconstructed  image of the initial aggregate, then basic morphological operations and small object removal are applied to reduce noise  and to remove sprouts belonging to other aggregates. The sprouting spheroid is identified as the largest cluster of connected pixels.

To quantify sprout density around the aggregates, the radial density profile (D0(r,t)) is calculated for each frame as:
> *D0(r, t) = A_{sprout}(r+w, r-w, t) / A_{ring}(r+w, r-w)* ,

where *r+w* and *r-w* are the outer and inner radii of a ring, respectively. The area of the ring is *A_{ring} = 4pi rw*. Whithin this ring, at a certain time *t* the area occupied by sprouts is denoted by *A_{sprout}(r+w, r-w, t)*. The ring has the same center as the minimal enclosing circle of the initial aggregate and *w = 5um*. To eliminate differences due to the initial conditions the radial density profiles is normalized as
> *D(r, t) = D0(r, t) - D0(r, t0)*

where *D0(r, t0)* is the density profile of the first frame.


## Dependencies ##

The script requires Python 2.7 and the following python modules:

* numpy >= 1.9.2 
* scipy >= 0.13.3
* cv2 >= 3.0.0
* skimage >= 0.11.3
* argparse >= 1.2.1


## Input files ##

Each image sequence should be kept in the same folder in jpg format.


## Output files ##

Output images and files are saved into a *subfolder* of ./imgproc_imgand , the *subfolder* contains the name of the experiment's field, the name of the segmentation method and the value of it's parameters (e.g.: ./imgproc_imgand/X08_roi_320_250_100_adapt_11_7/ ). The default output of the scripts are:
 * segmented image overlaid on the original image, saved into subfolder/resimgs/
 * sprout density files, saved into subfolder/resfiles/:
   * radial density profile of the central aggregate as a .dat file with 3 columns (#1: radius, #2: total area of ring, #3: density ratio of the initial aggregate),
   * radial density profile of the sprout on each image in image sequence as a .dat file with 3+*frame number* columns (#1: radius, #2: total area of ring, #3: density ratio of the initial aggregate, #4-: density ratio of the sprout),
   * a .dat file containing the segmentation parameters.
 * if any of *addimgs* or *test* options are used additional images are saved into subfolder/addimgs/.


## Usage ##

1. Download the sprout_density_analyzer.py file for image segmentation to the desired location. 
2. Run the Python script as `python sprout_density_analyzer.py -h` to reach the help menu.
 * Give the path of the folder containing the image sequence and the template image name as:
     
    > PATH / FOLD / EXP + \'-\' + SEP + FIELD + _%03d.jpg
     
    >  e.g.: ./imgs/Z229-ph_X03_001.jpg 
     
    > `python sprout_density_analyzer.py -fold imgs -exp Z229 -sep ph_ -field X03`

 * Give the total frame number within an image sequence with optional argument `-frame`.
 * The segmentation method can be selected by using one of the arguments from this mutually exclusive argument group:
   * Otsu segmentation: `-otsu`
     * default segmentation method
      * no additional parameter needed
      * e.g.: `-otsu`
   * Manual segmentation: `-manu`
     * 1 optional parameter: threshold value
      * e.g.: `-manu -thr 95`
   * Hough Circle Transform: `-hough`
     * 4 optional parameters:
       * Canny edge detector's higher threshold
        * accumulator threshold for center of the circles 
        * minimal and maximal circle radius
      * e.g.: `-hough -canny 130 -acc 15 -minr 70 -maxr 90`
   * Adaptive segmentation with ROI: `-roi`
     * 5 optional parameters:
       * ROI rectangle's center coordinates (x,y) and half side size in y direction
        * block size and subtracted value for adaptive segmentation
      * e.g.: `-roi -ox 320 -oy 250 -hsy 100 -block 11 -sub 7`
 * Additional features:
   * Making test segmentation on the first frame: `-test`
   * Saving additional images such as foreground masked images: `-addimgs`

3. Download the front_velocity_analysis.py script for making velocity analysis of the density front, follow the steps written in sprout_front_analysis.sh bash script to to run it properly.


## Test ##

Sample images are avilable in imgs/ folder, by downloading the repository and running the bash script as `./sprout_front_analysis.sh` test results can be compared with output files in imgproc_imgand_orig/ folder.


![alt tag](./sample_gif/Sprout_density_analyzer_sample_100ms_loop.gif "Sprout segmentation sample")



## Bibliography ##

[1] Z Zivkovic: Improved adaptive Gaussian mixture model for background subtraction.
	   *Proceedings of the 17th International Conference on Pattern Recognition, 2004.* (2004)

[2] Z Zivkovic and F van der Heijden: Efficient adaptive density estimation per image pixel for the task of background subtraction.
	   *Pattern Recognition Letters* **7** (27) (2005): 773-780.
