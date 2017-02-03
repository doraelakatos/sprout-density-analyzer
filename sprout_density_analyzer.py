import numpy as np
import cv2
from skimage import morphology, img_as_ubyte
from scipy import ndimage
import os, sys, argparse
import errno


class SmartFormatter(argparse.HelpFormatter):
    """ Subclass of argparse.HelpFormatter, for new line insertion.
    Help text marked at the beginning with R| will be formatted raw,
    i.e. won't wrap and fill out, observer newline.
    From Ruamel/Anthon van der Neut.
    """
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def sprout_morpho(img_bckg, img_stored):
    """ Removes small objects on background subtracted image, computes
    the per-element bit-wise logical disjunction for the removed and
    the accumulated image, and removes the small components on the
    accumulated image.

    Parameters:
    ----------
    img_bckg : ndarray (arbitrary shape, uint8 type)
        Background subtracted image ('white' or 'gray' version),
        originated from the actual frame.
    img_stored : nd_array (same shape and type as `img_bckg`)
        Accumulated image (same version as `img_bckg`).

    Returns:
    --------
    img_rso : ndarray (same shape and type as input `img_bckg`)
        The input `img_bckg` after small object removal.
    img_stored : ndarray (same shape and type as input `img_stored`)
        The input `img_stored` accumulated with the output `img_rso`,
        the small objects are removed.
    """
    # parameters for small object removal (=rso) (minimal size, connectivity)
    min_s1 = 30
    conn_s1 = 5
    min_s2 = 2000
    conn_s2 = 8
    
    # removal of small objects (#1) on image
    img_min = morphology.remove_small_objects(img_bckg.astype(bool), min_s1, conn_s1)
    img_rso = (img_min*img_bckg).astype(np.uint8)
    
    # image accumulation
    img_stored_im = cv2.bitwise_or(img_stored, img_rso)
    
    # removal of small objects (#2) on accumulated image
    img_min = morphology.remove_small_objects(img_stored_im.astype(bool), min_s2, conn_s2)
    img_stored = (img_min*img_stored_im).astype(np.uint8)

    return img_rso, img_stored


def sprout_morpho_combo(img_gray, img_whtst, img_combst, fr_id, fr_lbld):
    """ Computes per-element bit-wise conjunction of actual denoised,
    background subtracted gray image and accumulated white image resulting in
    the combined version of actual image. Computes the per-element bit-wise
    logical disjunction for the combined and the accumulated combined image.
    Basic morphological operations and small object removal are made through
    the combining process for additional denoising of image.

    Parameters:
    ----------
    img_gray : ndarray (arbitrary shape, uint8 type)
        Background subtracted and small object removed image ('gray'),
        originated from the actual frame. `img_rso` output of sprout_morpho().
    img_whtst : ndarray (same shape and type as `img_gray`)
        Accumulated image ('white'), `img_stored` output of sprout_morpho().
    img_combst : ndarray (same shape and type as `img_gray`)
        Accumulated image ('combined' ('gray' and 'white') version).
    fr_id : int
        Frame number.
    fr_lbld : ndarray (same shape and type as `img_gray`)
        Labeled image of the central aggregate (first frame).

    Returns:
    --------
    img_combst : ndarray, same shape and type as input `img_gray`
        The input `img_stored` accumulated with the output `img_rso`,
        the small objects are removed.

    """
    # parameters and structuring elements for morphological operations
    num_s = 3
    num_b = 4
    kern_s = np.ones((num_s, num_s), np.uint8)
    kern_b = np.ones((num_b, num_b), np.uint8)

    # parameters for small object removal (minimal size, connectivity)
    min_s1 = 10000
    conn_s1 = 3
    min_s2 = 30000
    conn_s2 = 3
    
    # white accumulated image dilation
    # gray image and dilated white accumulated image addition
    img_whtst_dil = cv2.dilate(img_whtst, kern_b, iterations=1)
    img_combo_dil = cv2.addWeighted(img_whtst_dil, 0.5, img_gray, 0.5, 0)
    
    # removal of small objects (#1) on accumulated image
    # gray image masking with the rso accumulated image
    img_min = morphology.remove_small_objects(img_combo_dil.astype(bool), min_s1, conn_s1)
    img_combo_dil = (img_min*img_combo_dil).astype(np.uint8)
    img_gray_rem = cv2.bitwise_and(img_gray, img_combo_dil)

    # masked gray image combination with the white accumulated image,
    # manual segmentation,
    # dilation and erosion with different kernels
    img_combo_rem = cv2.bitwise_or(img_gray_rem, img_whtst)
    rt, img_combo_rem_thr = cv2.threshold(img_combo_rem, 50, 255, cv2.THRESH_BINARY)
    img_combo_rem_thr = cv2.dilate(img_combo_rem_thr, kern_b, iterations=1)
    img_combo_rem_thr = cv2.erode(img_combo_rem_thr, kern_s, iterations=1)
    
    # removal of small objects (#2) on combined image
    # dilation of rso combined image and filling holes
    img_min = morphology.remove_small_objects(img_combo_rem_thr.astype(bool), min_s2, conn_s2)
    img_combo_rem_thr = (img_min*img_combo_rem_thr).astype(np.uint8)
    img_combo_rem_thr = cv2.dilate(img_combo_rem_thr, kern_s, iterations=1)
    img_combo_rem_thr = img_as_ubyte(ndimage.morphology.binary_fill_holes(img_combo_rem_thr))

    # combined image accumulation:
    # changes on the first 5 frames are ignored,
    if fr_id >= 4:
        img_combst = cv2.bitwise_or(img_combo_rem_thr, img_combst)
    else:
        img_combst = cv2.bitwise_or(fr_lbld, img_combst)
    
    # final morphological operations (dilation, erosion, filling holes)
    img_combst = cv2.dilate(img_combst, kern_s, iterations=1)
    img_combst = cv2.erode(img_combst, kern_s, iterations=1)
    img_combst = img_as_ubyte(ndimage.morphology.binary_fill_holes(img_combst))

    return img_combst
    

def img_roi(img_orig, roipar):
    """ Selecting region of interest of original image for segmentation.

    Parameters:
    ----------
    img_orig : ndarray (arbitrary shape, uint8 type)
        Original grayscale image (first frame).
    roipar : ndarray (1x3 shape, int type)
        Center coordinates of ROI rectangle and half side size in y (raw)
        direction.

    Returns:
    --------
    img_out : ndarray, (shape is obtained by roi_pars, uint8 type)
        Rectangle shaped part cut from `img_orig`, position and size is
        obtained by `roi_pars`.
    roi_pars : ndarray (1x4 shape, int type)
        Y_min, Y_max, X_min, X_max coordinates of ROI rectangle.
    """
    # initial roi parameters: center coordinates and y half side size
    ox = roipar[0]
    oy = roipar[1]
    hsy = roipar[2]

    # calculating the rectangle's half side size in x direction
    hsx = np.round(float(hsy)*1.33, decimals=0).astype(np.int)

    # calculating the coordinates of the corners
    # cutting-out of the original image
    roi_pars = np.array([oy - hsy, oy + hsy, ox - hsx, ox + hsx])
    img_out = img_orig[roi_pars[0]:roi_pars[1], roi_pars[2]:roi_pars[3]]
    
    return img_out, roi_pars


def central_circle(img_thresh, kern_s, kern_b):
    """ Denoising the segmented image, labeling the central aggregate, finding
    minimal enclosing circle of the aggregate. Making a black image with
    concentric circles, and drawing it on the labeled image for visualization.
    Making density calculation with density_calc().

    Parameters:
    ----------
    img_thresh : ndarray (arbitrary shape, uint8 type)
        Segmented original image (first frame).
    kern_s, kern_b : ndarrays (NxN shape, int type)
        Structuring elements used for basic morphological operations.

    Returns:
    --------
    img_lbld_fin : ndarray (same shape and type as input `img_thresh`)
        The labeled and denoised (with removal of small objects and basic
        morphological operations) `img_thresh`.
    img_circled : ndarray (same shape and type as input `img_thresh`)
        `img_lbld_fin` with concentric circles drawn on it with the center
        of the minimal enclosing circle, output of density_calc().
    img_empty_circled : ndarray (same shape and type as input `img_thresh`)
        Black image with concentric circles drawn as on `img_circled`.
    ox, oy : int
        Center coordinates of minimal enclosing circle around the aggregate.
    maxrad : int
        Radius of the largest concentric circle that fits the image.
    center_dens : ndarray (maxrad x 3 shape, float type)
        Density function of central aggregate, output of density_calc().
    """

    # dilation of segmented image
    img_closed = cv2.dilate(img_thresh, kern_s, iterations=1)
    
    # small object removal on dilated image
    min_s = 20000
    conn_s = 4
    img_min = morphology.remove_small_objects(img_closed.astype(bool), min_s, conn_s)
    img_rso = (img_min*img_thresh).astype(np.uint8)

    # dilation and erosion of the rso image,
    # filling holes, labeling blob(s),
    # segmentation, erosion
    img_closed = cv2.dilate(img_rso, kern_b, iterations=1)
    img_closed = cv2.erode(img_closed, kern_s, iterations=1)
    img_filled = ndimage.binary_fill_holes(img_closed.astype(bool)).astype(np.uint8)
    img_lbld, num_lbld = ndimage.measurements.label(img_filled)
    rt, img_thresh = cv2.threshold(img_lbld.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    img_lbld_fin = cv2.erode(img_thresh, kern_s, iterations=1)

    # finding minimal enclosing circle of the central blob
    img_cont, conts, hier = cv2.findContours(img_thresh, cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_NONE)
    if len(conts) != 0:
        encx, ency, encradius = 346.0, 260.0, 260.0
        for c in conts:
            (encx, ency), encradius = cv2.minEnclosingCircle(c)
        ox = np.around(encx, decimals=0).astype(np.int)
        oy = np.around(ency, decimals=0).astype(np.int)
        maxrad = np.minimum(img_thresh.shape[0] - oy, oy).astype(np.int)
    else:
        sys.exit("Couldn\'t find any circles, try other parameters!")

    # creating an empty image with circles
    img_empty = np.zeros_like(img_thresh).astype(np.uint8)
    img_empty_circled = cv2.cvtColor(img_empty, cv2.COLOR_GRAY2BGR)
    for radi in range(0, maxrad, 10):
        if (radi/10) % 5 == 0:
            img_empty_circled = cv2.circle(img_empty_circled, (ox, oy),
                                           radi, (40, 120, 0), 2)
        else:
            img_empty_circled = cv2.circle(img_empty_circled, (ox, oy),
                                           radi, (141, 90, 4), 1)

    # making density calculation on final image
    center_dens, img_circled = density_calc(img_lbld_fin, img_empty_circled, ox, oy, maxrad)

    return (img_lbld_fin, img_circled, img_empty_circled), (ox, oy, maxrad), center_dens


def central_morpho_roi(img_orig, roipar, adapt_p):
    """ Selecting area of interest on image by calling img_roi(),
    segmenting with adaptive threshold method, denoising with small object
    removal and basic morphological operations, calling central_circle()
    for making additional denoising and density calculation.

    Parameters:
    ----------
    img_orig : ndarray (arbitrary shape, uint8 type)
        Original grayscale image (first frame).
    roipar : ndarray (1x3 shape, int type)
        Center coordinates of ROI rectangle and half side size in y direction.
    adapt_p : ndarray (1x2 shape, int type)
        Parameters for adaptive segmentation: block size, subtracted value.

    Returns:
    --------
    imgs : ndarrays
        Return values from central_circle():
        img_lbld_fin, img_circled, img_empty_circled
    cpars : ints
        Return values from central_circle():
        ox, oy, maxrad
    center_dens :
        Return values from central_circle().

    """
    # parameters and structuring elements for morphological operations
    num_s = 4
    num_b = 8
    kern_s = np.ones((num_s, num_s), np.uint8)
    kern_b = np.ones((num_b, num_b), np.uint8)

    # parameters for small object removal (minimal size, connectivity)
    min_s = 4000
    conn_s = 3

    # ROI selection
    # (#1) selecting area of interest by calling img_roi() function
    # (#2) segmenting image with adaptive threshold method
    # (#3) blurring image with gaussian kernel
    # (#4) segmenting blurred img with Otsu method
    # (#5) removal of small objects on segmented image
    imgroi, rps = img_roi(img_orig, roipar)
    imgroi_adth = cv2.adaptiveThreshold(imgroi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, adapt_p[0], adapt_p[1])
    imgroi_blur = cv2.GaussianBlur(imgroi_adth, (11, 11), 0)
    rt, imgroi_otsu = cv2.threshold(imgroi_blur, 90, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_min = morphology.remove_small_objects(imgroi_otsu.astype(bool), min_s, conn_s)
    imgroi_otsu = (img_min*imgroi_otsu).astype(np.uint8)

    # pasting segmented roi image into an empty image
    img_thresh = np.zeros_like(img_orig).astype(np.uint8)
    img_thresh[rps[0]:rps[1], rps[2]:rps[3]] = imgroi_otsu

    # calling central circle function
    imgs, cpars, center_dens = central_circle(img_thresh, kern_s, kern_b)

    return imgs, cpars, center_dens


def central_morpho(img_orig, cmode, thrhold, hough_p):
    """ Segmenting original image with the chosen segmentation method:
    (0) Otsu, (1) Manual segmentation, (2) Hough Circle Transform.
    Making additional denoising and density calculation by calling
    central_circle() function.

    Parameters:
    ----------
    img_orig : ndarray (arbitrary shape, uint8 type)
        Original grayscale image (first frame).
    cmode : int
        Parameter for choosing segmentation method.
    thrhold : int
        Threshold value for manual segmentation.
    hough_p : ndarray (1x4 shape, int type)
        Parameters for Hough Circle Transform: Canny parameter, accumulator
        threshold, minimum circle radius, maximum circle radius.

    Returns:
    --------
    imgs : ndarrays
        Return values from central_circle():
        img_lbld_fin, img_circled, img_empty_circled
    cpars : ints
        Return values from central_circle():
        ox, oy, maxrad
    center_dens :
        Return values from central_circle().

    """
    # parameters and structuring elements for morphological operations
    num_t = 2
    num_s = 4
    num_b = 8
    kern_t = np.ones((num_t, num_t), np.uint8)
    kern_s = np.ones((num_s, num_s), np.uint8)
    kern_b = np.ones((num_b, num_b), np.uint8)

    # parameters for small object removal (minimal size, connectivity)
    min_s1 = 60
    conn_s1 = 8
    min_s2 = 5000
    conn_s2 = 3

    img_fth = np.copy(img_orig)
    
    if cmode == 0:
        # Otsu segmentation
        rt, img_thresh = cv2.threshold(img_fth, thrhold, 255,
                                       cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    elif cmode == 1:
        # Manual image segmentation
        rt, img_thresh = cv2.threshold(img_fth, thrhold, 255,
                                       cv2.THRESH_BINARY_INV)
    
    elif cmode == 2:
        # Hough Circle Transform
        # Image preparation:
        # (#1) blurring image with gaussian kernel,
        # (#2) laplace transformation, (#3) rescaling image,
        # (#4) segmenting image with Otsu method
        # (#5) morphological opening
        img_blur = cv2.GaussianBlur(img_fth, (7, 7), 0)
        img_lapl = cv2.Laplacian(img_blur, cv2.CV_8U)
        img_lapl = cv2.equalizeHist(img_lapl)
        rt, img_lthr = cv2.threshold(img_lapl, 30, 255,
                                     cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_lthr = cv2.morphologyEx(img_lthr, cv2.MORPH_OPEN, kern_t)

        # preparation for Hough transform
        # (#1) making a copy of the prepared image (img_htr)
        # (#2) removal of small objects
        # (#3) blurring rso image
        # (#4) Hough transform
        img_htr = np.copy(img_lthr)
        img_min = morphology.remove_small_objects(img_htr.astype(bool),
                                                  min_s1, conn_s1)
        img_htr = (img_min*img_htr).astype(np.uint8)
        img_crc = cv2.GaussianBlur(img_htr, (15, 15), 0)
        circles = cv2.HoughCircles(img_crc, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=hough_p[0], param2=hough_p[1],
                                   minRadius=hough_p[2], maxRadius=hough_p[3])

        # converting circle's center coordinates and radius to integers
        if circles is not None and circles.size == 3:
            circles = np.round(circles[0, :]).astype(int)
            ohx, ohy, radh = circles[0]
        elif circles is None:
            sys.exit("Couldn\'t find any circles, try other Hough parameters!")
        else:
            sys.exit("Too many circles were found, try other Hough parameters!")

        # drawing filled circle on prepared image
        # final removal of small objects
        img_prep = cv2.circle(img_lthr, (ohx, ohy), radh, 255, -1)
        img_min = morphology.remove_small_objects(img_prep.astype(bool), min_s2, conn_s2)
        img_prep = (img_min*img_prep).astype(np.uint8)
        img_thresh = img_prep

    # calling central circle function
    imgs, cpars, center_dens = central_circle(img_thresh, kern_s, kern_b)

    return imgs, cpars, center_dens


def density_calc(img, img_circle, orx, ory, radmax):
    """ Making density calculation on input image and drawing circles
    on it for visualization.

    Parameters:
    ----------
    img : ndarray (arbitrary shape, uint8 type)
        Segmented image.
    img_circle : ndarray (same shape and type as `img`)
        Black image with concentric circles.
    orx, ory : int
        Center coordinates of minimal enclosing circle of central
        aggregate on the first frame.
    radmax : int
        Radius of the largest concentric circle that fits the image.

    Returns:
    --------
    densvec : ndarray (radmax x 3 shape, float type)
        Density function of sprouting aggregate, contains 3 columns
        0: radius, 1: area of ring (A_ring(radius, radius-1)),
        2: ratio of sprout occupied area and ring area (A_sprout/A_ring)
    img_circled : ndarray (same shape and type as `img`)
        Segmented image with concentric circles.

    """
    # calculating number of pixels in rings
    densvec = np.zeros((radmax, 3)).astype(float)
    for r in range(radmax):
        img_empty = np.zeros_like(img).astype(np.uint8)
        img_circ = cv2.circle(img_empty, (orx, ory), r+1, 1, 10)
        img_rem = cv2.bitwise_and(img, img_circ)
        area_circ = np.sum(img_circ).astype(float)
        densvec[r, :] = [r+1, area_circ, np.sum(img_rem).astype(float)/area_circ]

    # making circled image for visualization
    img_color = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img_circled = cv2.bitwise_or(img_color, img_circle)
    img_circled = cv2.addWeighted(img_circled, 0.5, img_circle, 0.5, 0)

    return densvec, img_circled
    

def mkdirs(newdir, mode=0777):
    # Making new directory.
    try:
        os.makedirs(newdir, mode)
    except OSError, err:
        # Reraise the error unless it's about an already existing directory 
        if err.errno != errno.EEXIST or not os.path.isdir(newdir): 
            raise


def img_combine(img_orig, img_lbld, folder, imgn_beg, fr_id, imgn_end):
    """ Overlaying labeled image on original image and saving it.

    Parameters:
    ----------
    img_orig : ndarray (arbitrary shape, uint8 type)
        Original RGB image.
    img_lbld : ndarray (same shape and type as `img_orig`)
        Labeled version of original image.
    folder : str
        Folder name with path.
    imgn_beg : str
        Beginning of image name.
    fr_id : int
        Frame number.
    imgn_end: str
        End of image name.
    """

    img_found = np.zeros_like(img_orig)
    rt, img_found[:, :, 0] = cv2.threshold(img_lbld, 0, 255, cv2.THRESH_BINARY)
    img_found = cv2.addWeighted(img_orig, 0.7, img_found, 0.3, 0)
    img_save(img_found, folder, imgn_beg, fr_id, imgn_end)


def img_save(img, folder, imgn_beg, fr_id, imgn_end):
    # Saving an image.
    cv2.imwrite('%(fold)s%(img_beg)s%(a)03d_%(img_end)s.png' % {'fold': folder,
                'img_beg': imgn_beg, 'a': fr_id+1, 'img_end': imgn_end}, img)


##############################################################################
# Argument parser
p = argparse.ArgumentParser(description='Sprout density analyzer',
                            formatter_class=SmartFormatter)
gr_sys = p.add_argument_group('Arguments related to image name and path \n'
                              'PATH / FOLD / EXP + \'-\' + SEP + FIELD + _%03d.jpg \n'
                              'e.g.: ./imgs/Z229-ph_X03_001.jpg')
gr_sys.add_argument('-path', action='store', dest='path', type=str, default='./',
                    help='R|Path to image containing folder \ndefault: ./')
gr_sys.add_argument('-fold', action='store', dest='fold', type=str, default='imgs/',
                    help='R|Name of image containing folder \ndefault: imgs')
gr_sys.add_argument('-exp', action='store', dest='exp', type=str, required=True,
                    help='R|Name/number of experiment \ne.g., Z229')
gr_sys.add_argument('-sep', action='store', dest='sep', type=str, default='ph_',
                    help=('R|Separating string between experiment and field name'
                          '\ndefault: ph_'))
gr_sys.add_argument('-field', action='store', dest='field', type=str, required=True,
                    help='R|Name/number of field \ne.g., X03 ')
gr_sys.add_argument('-frame', action='store', dest='frame', type=int, default=30,
                    help=('R|Total number of frames'))

# Additional image, segmentation testing
gr_add = p.add_argument_group('Additional features')
gr_add.add_argument('-addimgs', action='store_true', default=False,
                    help='Saving additional images from image analysis')
gr_add.add_argument('-test', action='store_true', default=False,
                    help='Making only the segmentation on the first image')

# Mutually exclusive group: selecting from 4 segmentation method
gr_mutex = p.add_mutually_exclusive_group(required=True)
gr_mutex.add_argument('-otsu', action='store_true', dest='otsu', default=False,
                      help='R|Otsu segmentation method, \nno additional parameter needed')
gr_mutex.add_argument('-manu', action='store_true', dest='manu', default=False,
                      help='R|Manual segmentation method, \n1 optional parameter')
gr_mutex.add_argument('-hough', action='store_true', dest='hough', default=False,
                      help='R|Hough Circle Transform method, \n4 optional parameters')
gr_mutex.add_argument('-roi', action='store_true', dest='roi', default=False,
                      help=('R|ROI selection and adaptive segmentation method, \n'
                            '5 optional parameters'))

# Manual segmentation, optional parameter
gr_manu = p.add_argument_group('Optional parameters for manual segmentation')
gr_manu.add_argument('-thr', action='store', dest='thr', type=int,
                     help='R|Threshold value for manual segmentation \ndefault: 80')

# Hough circle transform, optional parameters
gr_hough = p.add_argument_group('Optional parameters for Hough Circle Transform')
gr_hough.add_argument('-canny', action='store', dest='canny', type=int, default=False,
                      help=('R|The higher threshold of the two passed to the Canny() \n'
                            'edge detector (the lower one is 2x smaller) \ndefault: 120'))
gr_hough.add_argument('-acc', action='store', dest='acc', type=int, default=False,
                      help=('R|Accumulator threshold for the circle centers '
                            'at the \ndetection stage \ndefault: 20'))
gr_hough.add_argument('-minr', action='store', dest='minr', type=int, default=False,
                      help='R|Minimum circle radius \ndefault: 75')
gr_hough.add_argument('-maxr', action='store', dest='maxr', type=int, default=False,
                      help='R|Maximum circle radius \ndefault: 90')

# ROI segmentation, optional parameters
gr_roi = p.add_argument_group('Optional parameters for ROI and adaptive segmentation')
gr_roi.add_argument('-ox', action='store', dest='ox', type=int, default=False,
                    help='R|X coordinate of roi rectangle\'s center \ndefault: 346')
gr_roi.add_argument('-oy', action='store', dest='oy', type=int, default=False,
                    help='R|Y coordinate of roi rectangle\'s center \ndefault: 260')
gr_roi.add_argument('-hsy', action='store', dest='hsy', type=int, default=False,
                    help='R|Rectangle\'s half side size in y direction \ndefault: 100')
gr_roi.add_argument('-block', action='store', dest='block', type=int, default=False,
                    help='R|Block size for adaptive segmentation \ndefault: 11')
gr_roi.add_argument('-sub', action='store', dest='sub', type=int, default=False,
                    help='R|Subtracted value for adaptive segmentation \ndefault: 8')
args = p.parse_args()

##############################################################################
# Default values for segmentation
# Segmentation method selection from Otsu-Manu-Hough:
#   THR_MODE: 0: Otsu, 1: Manual, 2: Hough
# Manual segmentation:
#   MANU_PAR: threshold value for manual segmentation
# Hough circle transform:
#   HOUGH_PAR: Canny parameter, accumulator threshold,
#              min circle, max circle radius
# ROI segmentation:
#   ROI_PAR: x coord, y coord, y half side size
#   ADAPT PAR: block size, subtracted value
# Segmentation parameters containers:
#   FIELD_THR_NAME: folder name part, contains segmentation parameters
#   THR_PARAMETERS: information and parameters of segmentation
THR_MODE = 0
MANU_PAR = 80
HOUGH_PAR = np.array([120, 20, 75, 90])
ROI_PAR = np.array([346, 260, 100])
ADAPT_PAR = np.array([11, 8])
THR_PARAMETERS = []
FIELD_THR_NAME = ''

###############################################################################
# Selecting segmentation method
if args.otsu:
    THR_MODE = 0
    THR_PARAMETERS.append('OTSU SEGMENTATION;')
    FIELD_THR_NAME += 'Otsu'
    print 'Otsu segmentation'

elif args.manu:
    THR_MODE = 1
    if args.thr is not None:
        MANU_PAR = args.thr
    pams = 'MANUAL SEGMENTATION' + str(MANU_PAR)
    THR_PARAMETERS.append(str(pams))
    FIELD_THR_NAME += 'Manu_' + str(MANU_PAR)
    print 'Manual segmentation with threshold value: %s ' % MANU_PAR

elif args.hough:
    THR_MODE = 2
    if args.canny is not None:
        HOUGH_PAR[0] = args.canny
    if args.acc is not None:
        HOUGH_PAR[1] = args.acc
    if args.minr is not None:
        HOUGH_PAR[2] = args.minr
    if args.maxr is not None:
        HOUGH_PAR[3] = args.maxr
    pams = 'HOUGH TRANSFORM' + str(HOUGH_PAR.tolist())
    THR_PARAMETERS.append(str(pams))
    FIELD_THR_NAME += 'Hough_' + '_'.join(HOUGH_PAR.tolist())
    print 'Hough transform with the following parameters: %s' % HOUGH_PAR

elif args.roi:
    if args.ox is not None:
        ROI_PAR[0] = args.ox
    if args.oy is not None:
        ROI_PAR[1] = args.oy
    if args.hsy is not None:
        ROI_PAR[2] = args.hsy
    if args.block is not None:
        ADAPT_PAR[0] = args.block
    if args.sub is not None:
        ADAPT_PAR[1] = args.sub

    pams = np.concatenate((np.array(['center, y half size:']), ROI_PAR,
                           np.array(['adapt  parameters:']), ADAPT_PAR))
    pams_file = ' '.join(pams.tolist())
    THR_PARAMETERS.append('ROI ADAPTIVE SEGMENTATION;')
    THR_PARAMETERS.append(str(pams_file))

    pams_npa = np.concatenate((np.array(['roi']), ROI_PAR, np.array(['adapt']), ADAPT_PAR))
    pams_name = '_'.join(pams_npa.tolist())
    FIELD_THR_NAME += str(pams_name)
    print 'ROI and adaptive segmentation'
    print 'The center coordinates and y half side size of roi rectangle are: ' % ROI_PAR
    print 'Adaptive segmentation\'s block size and subtracted value are: %s' % ADAPT_PAR

##############################################################################
# Other optional arguments - additional image and test mode selection
ADD_IMGS = args.addimgs
TEST = args.test

##############################################################################
# Creating folder names from folder path, exp an field names
fold_with_path = str(args.path) + '/' + str(args.fold) + '/'
fold_imgproc = str(args.path) + 'imgproc_imgand/'
fold_field = str(fold_imgproc) + str(args.field) + '_' + str(FIELD_THR_NAME) + '/'
fold_resimgs = str(fold_field) + 'resimgs/'
fold_resfiles = str(fold_field) + 'resfiles/'
fold_addimgs = str(fold_field) + 'addimgs/'
# Creating image name
img_name_beg = str(args.exp) + '-' + str(args.sep) + str(args.field) + '_'

##############################################################################
# Loading image sequence
# Creating background subtractor operator
# Calculating the length of image sequence
cap_name = str(fold_with_path) + str(img_name_beg) + '%03d.jpg'
cap = cv2.VideoCapture(cap_name)
fgbg = cv2.createBackgroundSubtractorMOG2()
fr_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if TEST:
    fr_length = 1
else:
    fr_length = args.frame

##############################################################################
# Making sprout density analysis

for frame_id in range(fr_length):

    print frame_id
    # reading a frame, converting it to grayscale
    ret1, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # image analysis of the first frame: segmentation, density calculation
    if frame_id == 0:
        # creating numpy arrays for accumulating gray, white and combined images
        img_gray_acc = np.zeros(frame_gray.shape, dtype=np.uint8)
        img_white_acc = np.copy(img_gray_acc)
        img_combo_acc = np.copy(img_gray_acc)

        # finding central aggregate by using the chosen segmentation method
        # making density analysis on the labeled frame
        if args.roi:
            cimgs, cparams, cdens = central_morpho_roi(frame_gray, ROI_PAR, ADAPT_PAR)
        else:
            cimgs, cparams, cdens = central_morpho(frame_gray, THR_MODE, MANU_PAR, HOUGH_PAR)

        # unpacking results
        frame_lbld, img_wcircles, img_onlycircles = cimgs
        origx, origy, maxirad = cparams
        center_densfunc = cdens

        # creating numpy arrays for storing density values and header
        # densfunc_fin by columns: #1 radius, #2 area of the circle,
        #                          #3 density results of central aggregate
        #                          #4 - density results of each frame (0 - fr_length)
        densfunc_fin = np.zeros((center_densfunc.shape[0], 3 + fr_length))
        densfunc_fin[:, 0:3] = center_densfunc
        densfunc_head = np.zeros((1, 3 + fr_length)).astype('|S10')
        densfunc_head[0:1, 0:3] = ['#radi', 'circarea', 'central']

        # creating folders
        mkdirs(str(fold_imgproc))
        mkdirs(str(fold_field))
        mkdirs(str(fold_resimgs))
        mkdirs(str(fold_resfiles))

        # blending original and labeled frame together, saving it
        img_combine(frame, frame_lbld, fold_addimgs, img_name_beg, frame_id, 'overlay')

        # saving additional images and density results of central aggregate
        if ADD_IMGS or TEST:
            mkdirs(str(fold_addimgs))
            img_save(frame_lbld, fold_addimgs, img_name_beg, frame_id, 'central_lab')
            img_save(img_wcircles, fold_addimgs, img_name_beg, frame_id, 'central_circles')
            np.savetxt('%(fold)s%(img)s_central_density_imgand.dat' %
                       {'fold': fold_resfiles, 'img': img_name_beg}, center_densfunc,
                       fmt='%.4f', delimiter='\t', header='#radi, Tcirc, Dsprout')

    # gray image processing
    # (#1) applying background subtraction on frame
    # (#2) adding the labeled central aggregate to the gray foreground image
    # (#3) denoising and accumulating gray image with sprout_morpho()
    fgmask_bckgsub = fgbg.apply(frame)
    fgmask_gray = cv2.addWeighted(fgmask_bckgsub, 0.9, frame_lbld, 0.1, 0)
    img_gray_denoi, img_gray_acc = sprout_morpho(fgmask_gray, img_gray_acc)

    # white image processing
    # (#1) selecting white points from the grayscale foreground masked image
    # (#2) combining it with the labeled central aggregate
    # (#3) denoising and accumulating white image with sprout_morpho()
    fgmask_white = np.zeros_like(fgmask_bckgsub, dtype=np.uint8)
    fgmask_white[np.where(fgmask_bckgsub == 255)] = 255
    fgmask_white = cv2.bitwise_or(fgmask_white, frame_lbld)
    img_white_denoi, img_white_acc = sprout_morpho(fgmask_white, img_white_acc)

    # combo image processing
    # (#1) segmenting the gray foreground masked image
    # (#2) combining the denoised gray and the accumulated white image
    #      with sprout_morpho_combo()
    # (#3) combining combo image with the segmented one with bitwise_and
    # (#4) combining original image and final results in an rgb image, saving it
    ret2, fgmask_gray_thr = cv2.threshold(fgmask_gray, 0, 255, cv2.THRESH_BINARY)
    img_combo_acc = sprout_morpho_combo(img_gray_denoi, img_white_acc,
                                        img_combo_acc, frame_id, frame_lbld)
    img_final = cv2.bitwise_and(fgmask_gray_thr, img_combo_acc)
    img_combine(frame, img_final, fold_resimgs, img_name_beg, frame_id, 'overlay')

    # making density calculation of the final image, storing it
    sprout_densfunc, img_sprout_circled = density_calc(img_final, img_onlycircles,
                                                       origx, origy, maxirad)
    densfunc_fin[:, frame_id + 3] = sprout_densfunc[:, 2]
    densfunc_head[0:1, frame_id + 3] = 't=' + str(frame_id+1)

    # saving additional images
    if ADD_IMGS:
        img_save(img_final, fold_addimgs, img_name_beg, frame_id, 'img_final')
        img_save(img_sprout_circled, fold_addimgs, img_name_beg, frame_id, 'sprout_circled')

    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# saving segmentation parameters and density function
denstosave = np.vstack((densfunc_head, densfunc_fin.astype('|S8')))
np.savetxt('%(fold)s%(img)s_sprout_density_imgand.dat' % {'fold': fold_resfiles,
           'img': img_name_beg}, denstosave, fmt='%s', delimiter='\t')

params = np.asarray(THR_PARAMETERS, dtype=np.str)
params = params[:, np.newaxis]
np.savetxt('%(fold)s%(img)s_sprout_params_imgand.dat' % {'fold': fold_resfiles,
           'img': img_name_beg}, params, fmt='%s', delimiter='\t')
