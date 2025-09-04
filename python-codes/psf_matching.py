# -*- coding: utf-8 -*-
"""
Created on Tue May 20 08:49:36 2025

@author: zacha
"""

import os
# import subprocess
import numpy as np
import astropy.io
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from reproject import reproject_exact
import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable # Utile pour placer la colorbar
from astropy.convolution import convolve_fft, convolve
from photutils.psf.matching import create_matching_kernel, SplitCosineBellWindow
# from scipy.signal import convolve2d
# from scipy.signal import fftconvolve

sex_config_file = 'default.sex' # Configuration file for Sextractor
psfex_config_file = 'default.psfex' # Configuration file for PSFextractor

# === INPUT VARIABLES R0600-ID67 ===

# direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//R0600-ID67//jwst_test//' # Directory in which are all .fits files + all Sextractor and PSFextractor files
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//R0600-ID67//jwst_test_2//' # Directory in which are all .fits files + all Sextractor and PSFextractor files
psf_ref = 'rxcj0600-grizli-v5.0-f444w-clear_drc_sci_psf.npy'    # reference image for the PSF matching
x0, y0 = 6734, 4057                                             # center coordinates
size = 400                                                      # region size (lentgh = height) in pixels for the new images
end_alma_name = 'image.pbcor.mJyppix.fits'
alpha, beta = 0.5, 0.6 # alpha, beta = np.round(np.linspace(0, 1, 1), 2), np.round(np.linspace(0, 1, 1), 2)

"""
# === INPUT VARIABLES A0102-ID224 ===
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//A0102-ID224//JWST' # Directory in which are all .fits files + all Sextractor and PSFextractor files
psf_ref = 'elgordo-grizli-v7.0-f444w-clear_drc_sci_psf.npy'     # reference image for the PSF matching
x0, y0 = 3300, 3600                                             # center coordinates
size = 400                                                      # region size (lentgh = height) in pixels for the new images
end_alma_name = 'image.pbcor.mJyppix.fits'
alpha, beta = 0.2, 0.5 # alpha, beta = np.round(np.linspace(0, 1, 1), 2), np.round(np.linspace(0, 1, 1), 2)
"""
"""
# === INPUT VARIABLES M0417-ID46 ===
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//M0417-ID46//JWST' # Directory in which are all .fits files + all Sextractor and PSFextractor files
psf_ref = "hlsp_canucs_jwst_macs0417-clu-40mas_f444w_v1_sci_psf.npy"    # reference image for the PSF matching
x0, y0 = (6030, 4435)                                                   # center coordinates
size = 400                                                              # region size (lentgh = height) in pixels for the new images
end_alma_name = 'image.pbcor.mJyppix.fits'
alpha, beta = 0.35, 0.5 # alpha, beta = np.round(np.linspace(0, 1, 1), 2), np.round(np.linspace(0, 1, 1), 2)
"""
# === CONFIGURATION ===
os.chdir(direc) # Change to working directory
print(f"now into directory: {os.getcwd()}")

def get_psf_from_npy(direc_init):
    """
    Load the first component of all .npy PSF files in a directory, ignoring subdirectories.

    Parameters:
        direc_init (str): Path to the directory containing .npy PSF files.

    Returns:
        psf_files (list): List of .npy filenames.
        psf_imgs (list): List of 2D numpy arrays (first component of each PSF).
    """
    psf_files = sorted([
        f for f in os.listdir(direc_init)
        if f.endswith('.npy') and os.path.isfile(os.path.join(direc_init, f))
    ])
    psf_imgs = [np.load(os.path.join(direc_init, f))[0] for f in psf_files]
    return psf_files, psf_imgs

def go_to_parent_and_into(folder_name):
    """
   Changes the current working directory to the parent directory,
   then attempts to enter a specified subfolder within that parent directory.

   Parameters:
   -----------
   folder_name : str
       The name of the subdirectory (inside the parent directory) to move into.

   Behavior:
   ---------
   - Moves up one level from the current working directory.
   - If the specified folder exists in the parent directory, changes into it.
   - Otherwise, prints a message and remains in the parent directory.
   """
    # Step 1: Move to parent directory
    os.chdir("..")
    print(f"Now in parent directory: {os.getcwd()}")

    # Step 2: Try to enter the specified folder
    if os.path.isdir(folder_name):
        os.chdir(folder_name)
        print(f"Successfully moved into: {os.getcwd()}")
    else:
        print(f"Folder '{folder_name}' not found in parent directory.")

def reshape_psf(image):
    """
    Extracts a LxL pixel window centered around the only pixel in the image with value 1. 
    (L is the size of the reference psf)

    Parameters:
    -----------
    image : np.ndarray
        2D array representing the image. Must contain exactly one pixel with value 1.

    Returns:
    --------
    window : np.ndarray
        A LxL array centered on the pixel with value 1.
    """
    # Choose the reference (target) PSF
    cube_target = np.load('psf_cubes/' + psf_ref)
    psf_target = cube_target[0]  # take first component
    L = psf_target.shape[0]
    
    # Find the coordinates of the pixel with value 1
    coords = np.argwhere(image == 1)
    
    if coords.shape[0] != 1:
        raise ValueError("The image must contain exactly one pixel with value 1.")

    center_x, center_y = coords[0]

    # Half window size
    half = L // 2

    # Define window bounds
    y_min = center_y - half
    y_max = center_y + half + 1
    x_min = center_x - half
    x_max = center_x + half + 1

    # Extract the 25x25 window
    window = image[x_min : x_max,
                    y_min : y_max]

    return window

def create_matching_kernel_astropy(psf_source, psf_target, eps=1e-6):
    # Shift the PSFs so that their center is at the origin (for accurate FFT)
    psf_source /= np.sum(psf_source)
    psf_target /= np.sum(psf_target)
    psf_source = np.fft.ifftshift(psf_source)
    psf_target = np.fft.ifftshift(psf_target)

    # Compute the Fourier transforms of both PSFs
    fft_source = np.fft.fft2(psf_source)
    fft_target = np.fft.fft2(psf_target)

    # Compute the matching kernel in Fourier space and transform back
    kernel_fft = fft_target / (fft_source + eps)
    kernel = np.fft.ifft2(kernel_fft).real
    kernel = np.fft.fftshift(kernel)  # Shift back to center the kernel

    # Normalize the kernel so that convolution conserves flux
    kernel /= np.sum(kernel)
    return kernel

def cutout_convol(kernel, center_coord, ratio, size, image_data):
    size = int(size * np.sqrt(ratio))
    x0, y0 = center_coord
    x_start = max(x0 - size//2, 0)
    y_start = max(y0 - size//2, 0)
    x_end = x_start + size
    y_end = y_start + size
    
    # Extract the sub-image using NumPy slicing (note: order is [y, x])
    galaxy_patch = image_data[y_start : y_end, x_start : x_end]

    # Apply the matching kernel via convolution
    matched_galaxy = convolve_fft(galaxy_patch, kernel, normalize_kernel = False, boundary = 'fill', fill_value = 0.0, nan_treatment = 'fill', allow_huge=True)
    
    # Sustitute the initial pixels by the convoluted ones
    matched_image = np.copy(image_data)
    matched_image[y_start : y_end, x_start : x_end] = matched_galaxy
    return matched_image

def get_pixel_size_ratio(hdu1, hdu2):
    """
    Return the ratio between the pixel sizes of two FITS images using CDELT1 or CD1_1.

    Parameters:
        hdu1 : hduist or HDU object from the first FITS file
        hdu2 : hduist or HDU object from the second FITS file

    Returns:
        float : pixel size ratio = pixel_size1 / pixel_size2
    """
    """
    # Extract headers
    header1 = hdu1[0].header if isinstance(hdu1, fits.HDUList) else hdu1.header
    header2 = hdu2[0].header if isinstance(hdu2, fits.HDUList) else hdu2.header
    """
    # Get pixel sizes
    pix_size1 = get_pixel_size(hdu1)
    pix_size2 = get_pixel_size(hdu2)

    return (pix_size2 / pix_size1)**2

# Apply the matching kernel via convolution on a size*size pixels part of the image, create a new HDU with the modified data and original header
def convolve_target(kernel, hdu, hdu_header, center_coord, size): # region size (lentgh = height)

    if type(hdu) == astropy.io.fits.hdu.hdulist.HDUList:
        image_data = hdu[0].data.astype(np.float32).copy()  # make a copy to avoid modifying the original in memory
        header = hdu[0].header  # keep the header
    else:
        image_data = hdu.data.astype(np.float32).copy()  # make a copy to avoid modifying the original in memory
        header = hdu.header  # keep the header
    """
    # Load the associated reference image
    image_ref = psf_ref.replace('_psf.npy', '.fits')
    hdu_ref = fits.open(image_ref, memmap=True)
    hdu_ref_header = hdu_ref[0].header
    # Define the center (x0, y0) and size (width, height) of the region to extract
    ratio = get_pixel_size_ratio(hdu_header, hdu_ref_header)
    
    matched_image = cutout_convol(kernel, center_coord, ratio, size, image_data)
    """
    matched_image = convolve(image_data, kernel, normalize_kernel = False, boundary = 'fill', fill_value = 0.0, nan_treatment = 'fill')
    
    # Create a new HDU with the modified data and original header
    hdu = fits.PrimaryHDU(data = matched_image, header = header)
    return hdu

def save_fits(hdu_final, image_path):
    # output directory for the convolved images (.fits format)
    output_dir = os.path.join(direc, "matched_images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"matched_{os.path.basename(image_path)}")

    # Save the resulting convolved image
    hdu_final.writeto(output_path, overwrite=True)
    print(f"Final image saved in: {output_path}")

def radial_profile(image, nbins, center=None):
    """
    Compute the radial profile of a 2D image, returning only values
    where r_centers < nbins.
    
    Parameters:
        image (2D np.array): Input image (e.g. PSF).
        center (tuple or None): (x, y) coordinates of the center. 
                                If None, defaults to image center.
        nbins (int): Number of radial bins.
    
    Returns:
        r_filtered (np.array): Radii (bin centers) < nbins.
        profile_filtered (np.array): Corresponding mean values.
    """
    y, x = np.indices(image.shape)
    if center is None:
        center = (image.shape[1] // 2, image.shape[0] // 2)

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.flatten()
    img = image.flatten()

    # Bin the radii
    max_radius = r.max()
    bins = np.linspace(0, max_radius, nbins + 1)
    inds = np.digitize(r, bins)

    profile = np.zeros(nbins)
    for i in range(1, nbins + 1):
        mask = inds == i
        if np.any(mask):
            profile[i - 1] = img[mask].mean()
        else:
            profile[i - 1] = np.nan  # or 0

    # Compute bin centers
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    # Filter: keep only r_centers < nbins
    mask_valid = r_centers < nbins
    r_filtered = r_centers[mask_valid]
    profile_filtered = profile[mask_valid]

    return r_filtered, profile_filtered

def plot_psf_profiles(psf_src, psf_target, kernel, alpha, beta, src_path, nbins):
    """
    Compute the mean squared error (MSE) between the source PSF after convolution
    and the target PSF, using a Split Cosine Bell window for kernel regularization.
    Plot radial profiles of the input PSF and PSF convolved with kernel.
    
    Parameters:
        psf (2D np.array): Original PSF image.
        kernel (2D np.array): Convolution kernel.
        target_psf (2D np.array): The reference PSF to match to.
        alpha (float): Fraction of the window used for the cosine tapering (start).
        beta (float): Fraction of the window used for the cosine tapering (end).
        nbins (int): Number of radial bins for the profiles.
    
    Returns:
        None. Shows a matplotlib plot with
            mse (float): Mean squared error between convolved and target PSF.
            radial profiles
        
    """
    # Convolve the PSF with the kernel (mode='same' to keep size)
    psf_conv = convolve(psf_src, kernel, normalize_kernel = False, boundary = 'fill', fill_value = 0.0, nan_treatment = 'fill')
    
    # Compute the MSE between convolved PSF and target
    mse = np.mean((psf_conv - psf_target) ** 2)
    
    # Compute radial profiles
    r, prof_psf = radial_profile(psf_target, nbins = nbins)
    _, prof_conv = radial_profile(psf_conv, nbins = nbins)
    r *= 0.04 # Convert in arcsec
    # Compute fractional residual
    res = np.where(prof_psf != 0, (prof_psf - prof_conv)/prof_psf * 100, 0) # [%]
    
    # Plotting
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # Le premier subplot est 3x plus haut que le second

    # Premier subplot
    ax1 = fig.add_subplot(gs[0])
    ax1.loglog(r, prof_psf, label='Original PSF', lw=2)
    ax1.loglog(r, prof_conv, label='Convolved PSF', lw=2, linestyle='--')
    ax1.set_ylabel('PSF radial intensity', size=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_title(f'Radial Profile of PSF and Convolved PSF, alpha = {alpha}, beta = {beta}, MSE = {mse}')
    ax1.legend()

    # Deuxième subplot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(r, res, 'black')
    ax2.axhspan(-5, 5, color = 'gray', alpha = 0.4)
    ax2.set_ylim(-20, 40)  # Contraint les ordonnées entre -20 et 40
    ax2.set_ylabel('Fractional residual (%)', size=20)
    ax2.set_xlabel('Radius (arcseconds)', size=20)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.legend()
    plt.tight_layout()

    fig.canvas.manager.set_window_title(f"Radial Profile of {src_path}, alpha = {alpha}, beta = {beta}")
    #plt.subplots_adjust(left=0.07, bottom=0.2, right=0.96, top=0.8, wspace=0.4, hspace=0.4) # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    os.makedirs("test_windows", exist_ok=True)
    plt.savefig(os.path.join("test_windows", f"Radial Profile of {src_path}, alpha = {alpha}, beta = {beta}.png"))
    plt.close()
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 9))
    axs[0].imshow(psf_conv)
    axs[0].set_title('Convolved 4.44 µm')
    axs[1].imshow(psf_target)
    axs[1].set_title('Original 4.44 µm')
    fig.canvas.manager.set_window_title(f"Radial Profile of {src_path}, alpha = {alpha}, beta = {beta}")
    """
    

def replace_image_convolute(hdu_header, hdu, image_path, psf_src, center_coord, size, alpha, beta):
    """
    Replace a region of the input image with a convolution by a matching kernel
    that transforms the source PSF into a target PSF.

    Parameters:
        hdu (astropy.io.fits.hdulist): FITS HDU list containing the image data to be processed.
        image_path (str): Path or identifier of the image file (used for logging).
        psf_src (np.array): Source PSF array (the PSF of the input image region).
        center_coord (tuple): (x, y) pixel coordinates defining the center of the region to convolve.
        size (int): Size of the square region to convolve (length = height in pixels).
        alpha (float): Parameter controlling the shape of the Split Cosine Bell window used in kernel creation.
        beta (float): (Unused in the provided snippet, but presumably controls window or kernel smoothing).

    Returns:
        astropy.io.fits.PrimaryHDU: A new HDU containing the convolved image region with the original header preserved.

    Description:
        This function loads the target PSF cube, extracts the reference PSF component,
        calculates the matching kernel to convert the source PSF into the target PSF using a Split Cosine Bell window,
        then convolves a square region of the image centered on `center_coord` of size `size` pixels,
        and finally returns a new HDU with the convolved data. The image path is printed for logging.
    """

    # Choose the reference (target) PSF
    cube_target = np.load('psf_cubes/' + psf_ref)
    psf_target = cube_target[0]  # take first component
    psf_target /= np.sum(psf_target)

    # Compute matching kernel to convert psf_src into psf_target
    window = SplitCosineBellWindow(alpha = alpha, beta = beta)
    kernel = create_matching_kernel(psf_src, psf_target, window = window)
    
    # To analyse the best parmameters alpha and beta for the Split Cosine Bell window 
    plot_psf_profiles(psf_src, psf_target, kernel, alpha, beta, image_path, nbins = 25)
    
    # Apply the matching kernel via convolution on a size*size pixels part of the image, create a new HDU with the modified data and original header
    hdu = convolve_target(kernel, hdu, hdu_header, center_coord, size)
    print(f"{image_path} has been convolved with respect to the given reference")
    return hdu
    
def match_coord2(center_coord, hdu_src):
    # Load the associated ref galaxy image
    ref_path = psf_ref.replace('_psf.npy', '.fits')
    hdu_ref = fits.open(ref_path, memmap=True)
    
    # === 1. Get the pixel/world coordinates system (WCS) ===
    wcs_ref = WCS(hdu_ref[0].header)
    if type(hdu_src) == astropy.io.fits.hdu.hdulist.HDUList:
        wcs = WCS(hdu_src[0].header)
    else:
        wcs = WCS(hdu_src.header)   # type 'hdulist'
    
    # === 2. Convert pixel coordinates in ref to world coordinates (RA, Dec) ===
    world_coords = wcs_ref.pixel_to_world(center_coord[0], center_coord[1])  # pixel_to_world okay for just one or two pixels

    # === 3. Convert world coordinates to pixel coordinates in image with higher resolution ===
    coords = wcs.world_to_pixel(world_coords)

    # === 4. Round to integer pixel values and filter valid pixels inside image with higher resolution ===
    coords_int = np.round(coords).astype(int)
    return coords_int

def get_pixel_size(header):
    cdelt = header.get("CDELT1")
    if cdelt is not None and abs(cdelt) <= 0.001:
        return abs(cdelt)
    else:
        cd = header.get("CD1_1")
        if cd is not None:
            return abs(cd)
        else:
            raise ValueError("Neither suitable CDELT1 nor CD1_1 found in header.")

def replace_wcs(header_old, wcs_new):
    """
    Replace the WCS in a FITS header with a new one, skipping any values 
    that contain non-ASCII or non-printable characters.

    Parameters
    ----------
    header_old : astropy.io.fits.Header
        Original header with metadata and WCS.

    wcs_new : astropy.wcs.WCS
        New WCS object to insert.

    Returns
    -------
    header_new : astropy.io.fits.Header
        Cleaned header with new WCS and preserved non-WCS values.
    """

    def is_fits_compatible(value):
        if not isinstance(value, str):
            return True
        return all(c in string.printable for c in value)

    # WCS keywords to remove
    wcs_keys = WCS(header_old).to_header().keys()

    header_clean = Header()
    for key in header_old:
        if key not in wcs_keys and key != '':
            try:
                val = header_old[key]
                if is_fits_compatible(val):
                    header_clean[key] = val
                # else:
                    # print(f"⚠️ Skipped non-ASCII header entry: {key} = {val}")
            except Exception as e:
                print(f"⚠️ Error processing header key '{key}': {e}")

    # Add new WCS
    header_wcs = wcs_new.to_header()
    for key in header_wcs:
        header_clean[key] = header_wcs[key]

    return header_clean

def replace_wcs2(header_old, wcs_new):
    """
    Replace the WCS in a FITS header with a new one, skipping any values 
    that contain non-ASCII or non-printable characters.

    Parameters
    ----------
    header_old : astropy.io.fits.Header
        Original header with metadata and WCS.

    wcs_new : astropy.wcs.WCS
        New WCS object to insert.

    Returns
    -------
    header_new : astropy.io.fits.Header
        Cleaned header with new WCS and preserved non-WCS values,
        but preserving CDELT1 and CDELT2 from the original header.
        The complete list of modified headers is here:
        [
            'CTYPE1', 'CTYPE2',
            'CRVAL1', 'CRVAL2',
            'CUNIT1', 'CUNIT2',
            'CDELT1', 'CDELT2',
            'CRPIX1', 'CRPIX2',
            'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
            'WCSAXES',
            'RADESYS', 'LONPOLE', 'LATPOLE'
        ]
    """

    def is_fits_compatible(value):
        if not isinstance(value, str):
            return True
        return all(c in string.printable for c in value)

    # WCS keywords to remove (keys in the old WCS)
    wcs_keys = WCS(header_old).to_header().keys()

    header_clean = Header()
    for key in header_old:
        if key not in wcs_keys and key != '':
            try:
                val = header_old[key]
                if is_fits_compatible(val):
                    header_clean[key] = val
                else:
                    print(f"⚠️ Skipped non-ASCII header entry: {key} = {val}")
            except Exception as e:
                print(f"⚠️ Error processing header key '{key}': {e}")
                """
    # Add new WCS keys, but for 'CDELT1' and 'CDELT2' use original header values
    header_wcs = wcs_new.to_header()
    for key in header_wcs:
        if key in ['CDELT1', 'CDELT2']:
            # Use value from old header if exists, else new WCS value
            val = header_old.get(key, header_wcs[key])
            header_clean[key] = val
        else:
            header_clean[key] = header_wcs[key]
"""
    return header_clean
    
def cutout_fits(hdu_ref, center_coord, hdu_source, size): # region size (lentgh = height) in pixels
    """
    Performs a cutout around a given sky position and reprojects a high-resolution source image
    onto the WCS of a lower-resolution reference image.

    Parameters
    ----------
    hdu_ref : astropy.io.fits.hduist
        Reference image (typically lower resolution) containing the target WCS.

    center_coord : astropy.coordinates.SkyCoord
        Sky coordinate (e.g., ICRS) of the center of the cutout.

    hdu_source : astropy.io.fits.ImageHDU or PrimaryHDU
        Source image (typically higher resolution) to be reprojected.

    size : int
        Output cutout size in pixels (the final image will be size × size pixels).
        The region extracted from the source image is scaled according to pixel scale ratio.

    Returns
    -------
    reprojected_data : numpy.ndarray
        The source image data, reprojected onto the reference WCS.

    cutout_header : astropy.io.fits.Header
        FITS header containing the WCS of the cutout and selected metadata.

    Notes
    -----
    - The function first creates a WCS cutout in the reference image to define the target projection.
    - It then extracts a region from the source image around the same sky position, scaled by the resolution ratio.
    - The source cutout is reprojected onto the target WCS using `reproject_exact`.
    - The resulting header includes WCS information and selected metadata (e.g., 'BUNIT', 'EXPTIME') from the source image if available.
    """

    # --- Target WCS (low resolution image: 0.04"/px)
    ref_wcs = WCS(hdu_ref[0].header)
    
    # --- Step 1: Make a WCS cutout in the target image (this defines the output WCS)
    cutout_target = Cutout2D(hdu_ref[0].data, position=center_coord, size=size, wcs=ref_wcs)
    cutout_header = cutout_target.wcs.to_header()
    cutout_wcs = WCS(cutout_header)
    
    # --- Step 2: Cut the source image around the same sky position
    if type(hdu_source) == astropy.io.fits.hdu.hdulist.HDUList:
        src_header = hdu_source[0].header
        source_data = hdu_source[0].data
    else:
        src_header = hdu_source.header   # type 'hdulist'
        source_data = hdu_source.data
    source_wcs = WCS(src_header)

    # --- Step 3: Convert the sky position into pixel position in the source image
    source_center_px = match_coord2(center_coord, hdu_source)

    # Cut a region in the source image (larger if needed, depending on resolution ratio)
    ratio = get_pixel_size_ratio(src_header, hdu_ref[0].header)
    cutout_source = Cutout2D(source_data, position=source_center_px, size= int(size * np.sqrt(ratio)), wcs=source_wcs)
    # --- Step 4: Reproject the source cutout onto the cutout WCS
    reprojected_data, footprint = reproject_exact((cutout_source.data, cutout_source.wcs), cutout_header, shape_out= (int(size), int(size)))
            
    new_header = replace_wcs(src_header, cutout_wcs)
    return reprojected_data * ratio, new_header

    
def replace_image_projection(image_path, hdu_convoluted, center_coord, size): # region size (lentgh = height) in pixels
    hdu_src = hdu_convoluted
    # Load the associated reference image
    image_ref = psf_ref.replace('_psf.npy', '.fits')
    hdu_ref = fits.open(image_ref, memmap=True)
    
    reprojected_data, cutout_header = cutout_fits(hdu_ref, center_coord, hdu_src, size)
    # Create a new HDU with the modified data and original header
    reprojected_hdu = fits.PrimaryHDU(data = reprojected_data, header = cutout_header)
    print(f"{image_path} has been projected onto the given reference")
    return reprojected_hdu

    
def main():
    """Main function to process all FITS files in the directory."""
    
    # === CONFIGURATION ===
    center_coord = (x0, y0)
    npy_direc = "psf_cubes"
    psf_files, psf_imgs = get_psf_from_npy(npy_direc)
    target_index = psf_files.index(psf_ref)
    # plt_psf_prez(psf_imgs, psf_files[0])
    
    # Loop over all PSF images and apply matching
    for i, psf_src in enumerate(psf_imgs):
        
        psf_src /= np.sum(psf_src)
        # Step 1 : Load the associated galaxy image
        image_path = psf_files[i].replace('_psf.npy', '.fits')
        hdu = fits.open(image_path, memmap=True)
        hdu_header = hdu[0].header
        
        if i == target_index: # the reference fits, that must not change
            print(f"{image_path} has not been convolved because it was the given reference")

            # save_fits(hdu, image_path)
            hdu_projected = replace_image_projection(image_path, hdu, center_coord, size)
            
            # Step 3 : Save the resulting convolved image
            save_fits(hdu_projected, image_path)

        else:
            # Step 2 : Convolution and projection
            center_coord_1 = match_coord2(center_coord, hdu)
            hdu_projected = replace_image_projection(image_path, hdu, center_coord, size)
            hdu_convoluted = replace_image_convolute(hdu_header, hdu_projected, image_path, psf_src, center_coord_1, size, alpha = alpha, beta = beta)
            
            # Step 3 : Save the resulting convolved image
            save_fits(hdu_convoluted, image_path)
        
    """
        # if image_path == "rxcj0600-grizli-v5.0-f150w-clear_drc_sci.fits":
        if image_path == "hlsp_canucs_jwst_macs0417-clu-40mas_f150w_v1_sci.fits":
            for alph in alpha:
                for bet in beta:
                    center_coord_1 = match_coord2(center_coord, hdu)
                    hdu_projected = replace_image_projection(image_path, hdu, center_coord, size)
                    hdu_convoluted = replace_image_convolute(hdu_header, hdu_projected, image_path, psf_src, center_coord_1, size, alpha = alph, beta = bet)
"""
    # For ALMA images
    
    # Step 1 : Load the associated galaxy image
    go_to_parent_and_into("ALMA") 
    fits_files_path = [f for f in os.listdir(os.getcwd()) if f.endswith(end_alma_name)]
    # psf_files_path = [f for f in os.listdir(os.getcwd()) if f.endswith('psf_51pix.fits')]
    fits_files = [fits.open(f, memmap=True) for f in fits_files_path]
    psf_fits_files = [fits.open(f, memmap=True) for f in os.listdir(os.getcwd()) if f.endswith('psf_51pix.fits') or f.endswith('psf.fits')]
    os.chdir(direc) # Change to working directory
    print(f"now into directory: {os.getcwd()}")
    
    if not fits_files:
        print("[WARN] No FITS files found.")
    for i, fits_image in enumerate(fits_files):
        image_path = fits_files_path[i].replace('.fits', '.fits')
        hdu = fits_image
        hdu_header = hdu[0].header
        psf_src = psf_fits_files[i][0].data
        # psf_src = reshape_psf(psf_src_orig)
        psf_src /= np.sum(psf_src)
        # plt.imshow(psf_src)
        
        """
        for alph in alpha:
            for bet in beta:
                center_coord_1 = match_coord2(center_coord, hdu)
                hdu_projected = replace_image_projection(image_path, hdu, center_coord, size)
                hdu_convoluted = replace_image_convolute(hdu_header, hdu_projected, image_path, psf_src, center_coord_1, size, alpha = alph, beta = bet)
        """

        # Step 2 : Convolution and projection
        center_coord_1 = match_coord2(center_coord, hdu)
        hdu_projected = replace_image_projection(image_path, hdu, center_coord, size)
        hdu_convoluted = replace_image_convolute(hdu_header, hdu_projected, image_path, psf_src, center_coord_1, size, alpha = alpha, beta = beta)
        
        # Step 3 : Save the resulting convolved image
        save_fits(hdu_convoluted, image_path)  

if __name__ == '__main__':
    main()
    """
    taper = SplitCosineBellWindow(alpha=0.2, beta=0.8)
    data = taper((101, 101))
    plt.plot(data[50, :])
    """
