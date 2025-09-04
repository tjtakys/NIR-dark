#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:43:59 2025

@author: zlemoult
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable # Utile pour placer la colorbar
import pandas as pd
import re
import csv
import astropy.io
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.cosmology import Planck18 as cosmo
from astropy.visualization import make_lupton_rgb
from matplotlib.patches import Circle
# import matplotlib.transforms as transforms
from skimage import measure
from astropy.table import Table

"""
# === INPUT VARIABLES R0600-ID67 ===
# direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//R0600-ID67//jwst_test//' # Directory in which are all .fits files + all Sextractor and PSFextractor files
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//R0600-ID67//jwst_test_2//' # Directory in which are all .fits files + all Sextractor and PSFextractor files
psf_ref = 'rxcj0600-grizli-v5.0-f444w-clear_drc_sci_psf.npy'    # reference image for the PSF matching
x0, y0 = 6734, 4057                                             # center coordinates
size = 2 / 0.04                                                 # N / 0.04 donne une image de N"
scl = 1                                                         # For the scale in the final images
stretch1, stretch2, Q1, Q2 = 0.2, 0.2, 4, 2                    # Parameters for the RBG image
end_alma_name = 'image.pbcor.mJyppix.fits'
Galaxy_name = 'R0600-ID67'
z_spec = 4.80
"""

# === INPUT VARIABLES A0102-ID224 ===
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//A0102-ID224//JWST' # Directory in which are all .fits files + all Sextractor and PSFextractor files
psf_ref = 'elgordo-grizli-v7.0-f444w-clear_drc_sci_psf.npy'     # reference image for the PSF matching
x0, y0 = 3300, 3600                                             # center coordinates
size = 6 / 0.04                                                 # N / 0.04 donne une image de N"
scl = 5                                                         # For the scale in the final images
stretch1, stretch2, Q1, Q2 = 0.8, 0.8, 10, 6                    # Parameters for the RBG image
end_alma_name = 'image.pbcor.mJyppix.fits'
Galaxy_name = 'A0102-ID224'
z_spec = 4.33

"""
# === INPUT VARIABLES M0417-ID46 ===
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//M0417-ID46//JWST' # Directory in which are all .fits files + all Sextractor and PSFextractor files
psf_ref = "hlsp_canucs_jwst_macs0417-clu-40mas_f444w_v1_sci_psf.npy"    # reference image for the PSF matching
x0, y0 = 6030, 4435                                                     # center coordinates   
size = 4 / 0.04                                                         # N / 0.04 donne une image de N"                                                        # region size (lentgh = height) in pixels for the new images
scl = 3                                                                 # For the scale in the final images
stretch1, stretch2, Q1, Q2 = 0.8, 0.8, 10, 6                            # Parameters for the RBG image
end_alma_name = 'image.pbcor.mJyppix.fits'
Galaxy_name = 'M0417-ID46'
z_spec = 3.65
"""


# === CONFIGURATION ===
os.chdir(direc) # Change to working directory
fits_jwst_coord = psf_ref.replace('_psf.npy', '.fits') # reference fits relative path for the PSF matching
hdu_jwst_coord = fits.open(fits_jwst_coord, memmap = True) # reference fits for the PSF matching
sex_config_file = 'default.sex' # Configuration file for Sextractor

# === INPUT VARIABLES ===
alma_frequencies_ghz = [(223.015, 243.002)]
# alma_frequencies_ghz = [(223.015, 227.002), (241.015, 245.002)]

def check_gal_coord(hdu, hdu_ref, coords_ref):
    # === 1. Load fits ===
    if type(hdu) == astropy.io.fits.hdu.hdulist.HDUList:
        wcs = WCS(hdu[0].header)
        data = hdu[0].data.astype(np.float32)
    else:
        wcs = WCS(hdu.header)   # type 'hduist'
        data = hdu.data.astype(np.float32)
    
    wcs_ref = WCS(hdu_ref[0].header)

    # === 2. Convert pixel coordinates in ref to world coordinates (RA, Dec) ===
    world_coords = wcs_ref.wcs_pix2world(coords_ref, 1)

    # === 3. Convert world coordinates to pixel coordinates in image with higher resolution ===
    coords = wcs.wcs_world2pix(world_coords, 1)

    # === 4. Round to integer pixel values and filter valid pixels inside image with higher resolution ===
    coords_int = np.round(coords).astype(int)

    # verify we don't exceed the fits dimension
    h, w = data.shape
    valid = (coords_int[:, 0] >= 0) & (coords_int[:, 0] < w) & \
            (coords_int[:, 1] >= 0) & (coords_int[:, 1] < h)
    coords_valid = coords_int[valid]
    if coords_valid.shape[0] == 0:
        raise ValueError("No valid pixel coordinates found in the image bounds.")
    return coords_valid

def read_cat(catalog_name, center_coord):
    # Open the FITS-LDAC catalogue
    hdu = fits.open(catalog_name, memmap=True)

    # Usually, the data is in the second HDU
    data_table = Table(hdu[2].data)  # Sometimes it's in hdu[1], depending on the version

    # Print column names
    #print(data_table.colnames)
    output_file = "objects_" + catalog_name + "_summary.txt"
    with open(output_file, "w") as f:
        # Loop through the rows
        for row in data_table:
            number = row["NUMBER"]
            x = row["X_IMAGE"]
            y = row["Y_IMAGE"]
            flux = row["FLUX_APER"]
            star = row["CLASS_STAR"]
            f.write(f"Object {number}: (x={x:.1f}, y={y:.1f}), flux={flux:.2f}, star_ratio={star:.1f}\n")
    # Your target pixel coordinates (in image space)
    x_target, y_target = center_coord
    # Compute Euclidean distance to each object
    distances = np.sqrt((data_table['X_IMAGE'] - x_target)**2 + (data_table['Y_IMAGE'] - y_target)**2)
    # Find index of the closest object
    closest_index = np.argmin(distances)
    # Get the object data
    closest_object = data_table[closest_index]
    # Display information
    print(f"Closest object to (x={x_target}, y={y_target}):")
    print(f"  ID (NUMBER) : {closest_object['NUMBER']}")
    print(f"  Coordinates : x={closest_object['X_IMAGE']:.2f}, y={closest_object['Y_IMAGE']:.2f}")
    print(f"  Flux    : {closest_object['FLUX_APER']:.2f}")
    print(f"  Distance    : {distances[closest_index]:.2f} pixels")
    print(f"  Star_ratio    : {closest_object['CLASS_STAR']:.1f}")
    return closest_object['NUMBER'], (int(closest_object['X_IMAGE']), int(closest_object['Y_IMAGE']))

def coord_galaxy(catalog_name, object_id): # ID of the object you're interested in (e.g., from SExtractor catalog)
    """
    Extract pixel coordinates belonging to a given object ID from a segmentation map.
    Save them to a text file and return them as a NumPy array.
    """
    # Read the segmentation map
    seg_data = fits.getdata(catalog_name + "_segmentation.fits")

    # Get coordinates of all pixels belonging to this object
    coords = np.column_stack(np.where(seg_data == object_id))  # (y, x) coordinates
    
    output_file = catalog_name + "_galaxy_ref_coords.txt"
    with open(output_file, "w") as f:
        for y, x in coords:
            f.write(f"{x} {y}\n")  # Save as x y (more intuitive)
    # Example: display the number of pixels
    print(f"Object {object_id} contains {coords.shape[0]} pixels.")
    print(f"Coordinates saved to {output_file}")
    
    # Optional: display or use the pixel coordinates
    # for y, x in coords:
    #     print(f"Pixel at (x={x}, y={y}) belongs to object {object_id}")
    return coords

def plot_shape2(image_data, catalog_name, object_id, galaxy_coord): #  plot_shape(1338, (6737, 4059))
    """
    Read saved (x, y) coordinates.
    """
    coords_2 = galaxy_coord

    """
    # Optional : remove too far away pixels
    # Your target pixel coordinates from the catalog (in image space)
    x_target, y_target = center_coord_cat
    # Compute Euclidean distance to each object
    distances = np.sqrt((coords_2[:, 0] - x_target)**2 + (coords_2[:, 1] - y_target)**2)
    if 
    """
    # Load original image for display
    image_data = image_data[0].data

    # Plot image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_data, origin='lower', cmap='gray')
    ax.scatter(coords_2[:, 0], coords_2[:, 1], s=1, color='red')  # x, y
    ax.set_title(f"Object ID {object_id}")
    plt.show()
    
def match_coord3(center_coord, hdu_src):
    # Load the associated ref galaxy image
    hdu_jwst_coord = fits.open(fits_jwst_coord, memmap=True)
    
    # === 1. Get the pixel/world coordinates system (WCS) ===
    wcs_ref = WCS(hdu_jwst_coord[0].header)
    if type(hdu_src) == astropy.io.fits.hdu.hdulist.HDUList:
        wcs = WCS(hdu_src[0].header)
    else:
        wcs = WCS(hdu_src.header)   # type 'hduist'
    
    # === 2. Convert pixel coordinates in ref to world coordinates (RA, Dec) ===
    world_coords = wcs_ref.pixel_to_world(center_coord[0], center_coord[1])  # pixel_to_world okay for just one or two pixels

    # === 3. Convert world coordinates to pixel coordinates in image with higher resolution ===
    coords = wcs.world_to_pixel(world_coords)

    # === 4. Round to integer pixel values and filter valid pixels inside image with higher resolution ===
    coords_int = np.round(coords).astype(int)
    return coords_int

def update_checkimage_name(line, new_segmentation_name):
    """
    Updates the CHECKIMAGE_NAME line from a SExtractor .sex file to include three filenames:
    segmentation.fits, <new>_background.fits, <new>_background_rms.fits.
    Preserves spacing and inline comments.

    Parameters:
        line (str): The original CHECKIMAGE_NAME line.
        new_background_rms_name (str): Base name (with or without .fits) to generate background filenames.

    Returns:
        str: Updated line with three filenames and original comment, newline preserved.
    """
    # Extract comment if present
    parts = line.split('#', 1)
    main_part = parts[0].rstrip()
    comment = '#' + parts[1] if len(parts) > 1 else ''

    # Extract keyword and value
    tokens = main_part.split(None, 1)
    if len(tokens) != 2:
        raise ValueError("Line does not match expected CHECKIMAGE_NAME format")

    keyword, value = tokens
    if keyword != "CHECKIMAGE_NAME":
        raise ValueError("This is not a CHECKIMAGE_NAME line")

    # Parse filenames
    filenames = [f.strip() for f in value.split(',')]
    if len(filenames) < 1:
        raise ValueError("Expected at least one filename")

    # Extract the first filename (usually segmentation)
    background_file = filenames[1]
    background_rms_file = filenames[2]

    # Sanitize the base name
    base = new_segmentation_name.replace('.fits', '')

    # Build new filenames
    segmentation_file = f"{base}_segmentation.fits"

    # Reconstruct the line
    new_value = f"{segmentation_file}, {background_file}, {background_rms_file}"
    return f"{keyword}  {new_value:<65} {comment}".rstrip() + '\n'

def update_sex_config(sex_file_path, new_background_name):
    """
    Updates the CHECKIMAGE_NAME line in a SExtractor .sex configuration file,
    preserving file formatting.

    Parameters:
        sex_file_path (str): Path to the .sex file to update.
        new_background_name (str): The new background RMS filename (without automatic .fits prefix).
    """
    updated_lines = []
    catalog_name = new_background_name.replace('.fits', '.cat')

    with open(sex_file_path, "r") as f:
        for line in f:
            if line.strip().startswith('CATALOG_NAME'):
                updated_lines.append(f'CATALOG_NAME     {catalog_name}\n')
            elif line.strip().startswith("CHECKIMAGE_NAME"):
                new_line = update_checkimage_name(line, new_background_name)
                updated_lines.append(new_line)
            else:
                updated_lines.append(line)

    with open(sex_file_path, 'w') as f:
        f.writelines(updated_lines)

    print(f'[INFO] Updated CHECKIMAGE_NAME in {sex_file_path} → {new_background_name}_background.fits')

def run_sextractor(fits_image, sex_config_file):
    """Run SExtractor to generate the object catalog from a FITS image."""
    sex_cmd = f'sex {fits_image} -c {sex_config_file}'
    print(f"[INFO] Running: {sex_cmd}")
    subprocess.run(["wsl.exe", "sex", fits_image, "-c", sex_config_file], check=True)

def save_fits(hdu_final, image_path):
    # output directory for the convolved images (.fits format)
    output_dir = os.path.join(direc, "matched_images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"matched_{os.path.basename(image_path)}")

    # Save the resulting convolved image
    hdu_final.writeto(output_path, overwrite = True)
    print(f"Final image saved in: {output_path}")

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
    
def match_coord2(center_coord, hdu_src):
    # Load the associated ref galaxy image
    ref_path = psf_ref.replace('_psf.npy', '.fits')
    hdu_ref = fits.open(ref_path, memmap = True)
    
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

def cutout_fits2(center_coord, hdu_source, size): # region size (lentgh = height) in pixels
    """
    Performs a cutout around a given sky position and reprojects a high-resolution source image
    onto the WCS of a lower-resolution reference image.

    Parameters
    ----------

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
    - The resulting header includes WCS information and selected metadata (e.g., 'BUNIT', 'EXPTIME') from the source image if available.
    """
    
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
    cutout_source = Cutout2D(source_data, position=source_center_px, size = size, wcs = source_wcs)
    cutout_header = cutout_source.wcs.to_header()

    return cutout_source.data, cutout_header

def run_cigale(alma_frequencies_ghz=None, filter_name="ALMA_CUSTOM", filter_dir="filters/"):
    """
    Run CIGALE SED fitting. If ALMA frequency range is provided, generates a custom filter.

    Parameters
    ----------
    alma_frequencies_ghz : list of tuple of two floats, optional
        List of frequency ranges in GHz, e.g. [(220, 230), (230, 240)].
        For each tuple, a custom filter is generated.
    filter_name : str
        Base name of the filter (without extension). Will be suffixed with _1, _2, etc.
    filter_dir : str
        Directory where the filter file will be saved (default: "filters/").
    """

    if alma_frequencies_ghz is None or len(alma_frequencies_ghz) == 0:
        print("[INFO] No ALMA frequency ranges provided, skipping filter creation.")
        return

    c = 2.99792458e18  # speed of light in Å/s
    os.makedirs(filter_dir, exist_ok=True)

    for i, freq_range in enumerate(alma_frequencies_ghz):
        if not (isinstance(freq_range, (tuple, list)) and len(freq_range) == 2):
            raise ValueError(f"Frequency range at index {i} is not a tuple/list of length 2.")

        freq_min, freq_max = freq_range
        wl_min = c / (freq_max * 1e9)  # Convert GHz to Hz then to wavelength in Å
        wl_max = c / (freq_min * 1e9)

        wavelengths = np.linspace(wl_min, wl_max, 5)
        transmissions = np.ones_like(wavelengths)

        filter_lines = [
            f"# {filter_name}",
            "# photon",
            "# ALMA custom filter generated from frequency range"
        ]
        # Correction ici : écrire longueur d’onde et transmission sur la même ligne
        for wl, tr in zip(wavelengths, transmissions):
            filter_lines.append(f"{wl:.2f}  {tr:.4f}")

        filter_path = os.path.join(filter_dir, f"{filter_name}.filter")
        with open(filter_path, "w") as f:
            f.write("\n".join(filter_lines))

        print(f"[INFO] Custom ALMA filter created at {filter_path}")

        # Register the filter with CIGALE
        cmd_add_filter = ["pcigale-filters", "add", filter_path]
        print(f"[INFO] Running: {' '.join(cmd_add_filter)}")
        subprocess.run(cmd_add_filter, check=True)
"""
    # Run pcigale
    run_cmd = ["pcigale", "run"]
    print(f"[INFO] Running: {' '.join(run_cmd)}")
    try:
        subprocess.run(run_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("[ERROR] pcigale run failed.")
        print(f"[STDOUT]\n{e.stdout}")
        print(f"[STDERR]\n{e.stderr}")
        raise
    
    # Plot SEDs
    plot_cmd = ["pcigale-plots", "sed"]
    print(f"[INFO] Running: {' '.join(plot_cmd)}")
    try:
        subprocess.run(plot_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("[ERROR] pcigale-plots sed failed.")
        print(f"[STDOUT]\n{e.stdout}")
        print(f"[STDERR]\n{e.stderr}")
        raise
"""
def extract_filter_name(filename):
    """
    Extracts the JWST NIRCam filter name from a filename and returns it in CIGALE format.
    Example: 'rxcj0600-grizli-v5.0-f115w-clear_drc_sci' -> 'jwst.nircam.F115W'
    """
    match = re.search(r'f(\d{3})w', filename, re.IGNORECASE)
    match2 = re.search(r'f(\d{3})m', filename, re.IGNORECASE)
    if match:
        filter_code = match.group(1)
        return f'jwst.nircam.F{filter_code.upper()}W'
    elif match2:
        filter_code = match2.group(1)
        return f'jwst.nircam.F{filter_code.upper()}M'
    else:
        raise ValueError(f"Could not extract NIRCam filter from filename: {filename}")

def create_cigale_test_file_single_object(
        galaxy_name,
        fluxes_mjy, 
        errors_mjy, 
        filter_filenames, 
        redshift, 
        output_csv,
        alma_frequencies_ghz=None, 
        alma_fluxes_mjy=None, 
        alma_errors_mjy=None, 
        upper_limit = True):
    """
    Create a CIGALE-compatible input CSV file for a single object, including optional ALMA data.

    Parameters
    ----------
    fluxes_mjy : ndarray
        1D array of shape (N_filters,) with fluxes in mJy (e.g. JWST).
    errors_mjy : ndarray
        1D array of shape (N_filters,) with errors in mJy.
    filter_filenames : list of str
        List of JWST filter filenames (e.g., "jwst_nircam_f200w.filter").
    redshift : float
        Redshift of the object.
    output_csv : str
        Output path for the CSV file.
    alma_frequencies_ghz : list of tuple(float, float), optional
        List of tuples with (freq_min, freq_max) in GHz.
    alma_fluxes_mjy : list of float, optional
        Fluxes at those frequency ranges in mJy.
    alma_errors_mjy : list of float, optional
        Errors on those fluxes in mJy.
    upper_limit : bool
        Whether to apply upper limit logic when flux is below detection.
    """
    n_filters = fluxes_mjy.shape[0]
    assert errors_mjy.shape[0] == n_filters
    assert len(filter_filenames) == n_filters

    # Extract JWST-compatible filter names
    filter_names = [extract_filter_name(name) for name in filter_filenames]

    # Initialize dictionary
    data = {
        'id': [galaxy_name],
        'redshift': [redshift]
    }

    # Add JWST data
    for i, filt in enumerate(filter_names):
        data[filt] = [fluxes_mjy[i]]
        data[filt + '_err'] = [errors_mjy[i]]
        """
        # This is in case of upper limits considerations
        if upper_limit:
            if [fluxes_mjy[i]] <= [errors_mjy[i] * 7]:
                data[filt] = [errors_mjy[i] * 3]
                data[filt + '_err'] = [-errors_mjy[i] * 3]
            else:
                data[filt] = [fluxes_mjy[i]]
                data[filt + '_err'] = [errors_mjy[i]]
        else:
            if fluxes_mjy[i] > 0:
                data[filt] = [fluxes_mjy[i]]
                data[filt + '_err'] = [errors_mjy[i]]
            else:
                data[filt] = [errors_mjy[i] * 3]
                data[filt + '_err'] = [-errors_mjy[i] * 3]
        """

    # Add ALMA data (if any)
    if alma_fluxes_mjy is not None:
        for k in range(len(alma_fluxes_mjy)):
            alma_fluxes = alma_fluxes_mjy[k]
            alma_errors = alma_errors_mjy[k]
            (fmin, fmax) = alma_frequencies_ghz[k]
            
            filter_name = f'ALMA_{int(fmin)}-{int(fmax)}GHz_1'
            filt = filter_name
            data[filt] = [alma_fluxes]
            data[filt + '_err'] = [alma_errors]
    else:
        filter_name = None

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return filter_name

def create_cigale_test_file_multiple_objects(
    fluxes_list,  # list of 1D numpy arrays (JWST fluxes)
    errors_list,  # list of 1D numpy arrays (JWST errors)
    filter_filenames,  # list of JWST filter filenames
    redshift,  # redshift float or list of floats (one per object)
    output_multi,  # output CSV file path
    alma_frequencies_ghz=None,  # list of tuples like (min_freq, max_freq)
    alma_fluxes_list=None,  # list of 1D arrays (ALMA fluxes per object)
    alma_errors_list=None,  # list of 1D arrays (ALMA errors per object)
    upper_limit=True  # apply upper limit logic if True
):

    """
    Creates a CIGALE-compatible CSV file from multiple objects' photometric data.

    Parameters
    ----------
    fluxes_list : list of ndarray
        List of 1D arrays containing JWST fluxes in mJy for each object.
    errors_list : list of ndarray
        List of 1D arrays containing JWST errors in mJy for each object.
    filter_filenames : list of str
        List of JWST filter filenames (e.g., "jwst_nircam_f200w.filter").
    redshift : float
        redshift for each object.
    alma_fluxes_list : list of ndarray, optional
        List of 1D arrays with ALMA fluxes for each object.
    alma_errors_list : list of ndarray, optional
        List of 1D arrays with ALMA errors for each object.
    alma_frequencies_ghz : list of tuple, optional
        List of (min_freq, max_freq) tuples representing ALMA filter bands.
    output_csv : str
        Path to output CSV file.
    upper_limit : bool
        Whether to apply upper limit logic when flux is below detection.

    Returns
    -------
    pandas.DataFrame
        The resulting DataFrame written to the CSV file.
    """
    
    n_objects = len(fluxes_list[0])
    n_filters = len(filter_filenames)

    # Sanity checks
    assert all(len(arr) == n_objects for arr in fluxes_list), "All flux arrays must match the same number of pixels."
    assert all(len(arr) == n_objects for arr in errors_list), "All flux arrays must match the same number of pixels."

    # Redshift handling
    redshift_list = [float(redshift)] * n_objects


    if alma_fluxes_list is not None:
        assert alma_frequencies_ghz is not None
        assert all(len(arr) == n_objects for arr in alma_fluxes_list), "All flux arrays must match the same number of pixels."
        assert all(len(arr) == n_objects for arr in alma_errors_list), "All flux arrays must match the same number of pixels."

    # Extract filter names from filenames
    filter_names = [extract_filter_name(name) for name in filter_filenames]

    rows = []
    for i in range(n_objects):
        row = {'id': f'obj_{i+1}', 'redshift': redshift_list[i]}
        for j in range(n_filters):
            filt = filter_names[j]
            flux = fluxes_list[j][i]
            err = errors_list[j][i]
            if flux >= 0:
                row[filt] = flux
            else:
                row[filt] = 0
            row[filt + '_err'] = err
            """
            if upper_limit:
                if flux <= err * 7:
                    row[filt] = err * 3
                    row[filt + '_err'] = -err * 3
                else:
                    row[filt] = flux
                    row[filt + '_err'] = err
            else:
                if flux > 0:
                    row[filt] = flux
                    row[filt + '_err'] = err
                else:
                    row[filt] = err * 3
                    row[filt + '_err'] = -err * 3
            """
        # ALMA (optionnel)
        if alma_fluxes_list is not None:
            for k in range(len(alma_fluxes_list)):
                alma_flux = alma_fluxes_list[k][i]
                alma_error = alma_errors_list[k][i]
                (fmin, fmax) = alma_frequencies_ghz[k]
                
                alma_name = f'ALMA_{int(fmin)}-{int(fmax)}GHz_1'
                if alma_flux >= 0:
                    row[alma_name] = alma_flux
                else:
                    row[alma_name] = 0
                row[alma_name + '_err'] = alma_error
        else:
            alma_name = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_multi, index=False)
    return alma_name

def plot_sed_component(data, component_name):
    """
    Plot a specific SED component from a CIGALE best model FITS file.

    Parameters
    ----------
    data : astropy.io.fits.fitsrec.FITS_rec
        The table data from the second HDU of a CIGALE best model FITS file (e.g., fit[1].data).
    component_name : str
        The name of the SED component to plot. Must be one of the column names in the FITS file, 
        such as 'Fnu', 'dust', 'stellar.old', 'nebular.emission_young', etc.

    Returns
    -------
    None
        Displays a log-log plot of the SED component versus wavelength.
    """
    if component_name not in data.columns.names:
        raise ValueError(f"Component '{component_name}' not found in data columns.")

    λ = data['wavelength']
    flux = data[component_name]
    fig, ax = plt.subplots()
    plt.loglog(λ, flux, label=component_name.replace('.', ' ').capitalize())
    ax.set_ylabel('Flux', size = 20)
    ax.set_xlabel('Wavelength [nm]', size = 20)
    ax.tick_params(axis='x', labelsize = 20)
    ax.tick_params(axis='y', labelsize = 20)
    fig.canvas.manager.set_window_title(f"SED fitting of R0600-ID67 (JWST) Component: {component_name.replace('.', ' ').capitalize()}")
    #plt.subplots_adjust(left=0.07, bottom=0.2, right=0.96, top=0.8, wspace=0.4, hspace=0.4) # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    plt.show()

def convert_results_to_transposed_csv(input_file='results.txt', output_file='results_transposed.csv'):
    # Chemins
    input_file = os.path.join("out", input_file)
    output_dir = "sed_fittings"
    os.makedirs(output_dir, exist_ok=True)   # <-- crée le dossier si nécessaire
    output_file = os.path.join(output_dir, output_file)

    # Lire le fichier texte
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Séparer en-tête et données
    header = lines[0].strip().split()
    values = lines[1].strip().split()

    if len(header) != len(values):
        raise ValueError(
            f"Le nombre de colonnes ({len(header)}) ne correspond pas au nombre de valeurs ({len(values)})."
        )

    # Créer la structure transposée
    rows = [('parameter', 'value')]
    for h, v in zip(header, values):
        rows.append((h, v))

    # Écrire dans un fichier CSV transposé
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"✅ Données sauvegardées dans {output_file}")

def convert_results_to_transposed_csv_2(input_file='results.txt', output_file='results.csv'):
    # Chemins
    input_file = os.path.join("out", input_file)
    output_dir = "sed_fittings"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file)

    # Charger le fichier avec pandas (séparateur = espaces)
    df = pd.read_csv(input_file, delim_whitespace=True)

    # Sauvegarder en CSV tabulaire
    df.to_csv(output_file, index=False)

    print(f"✅ Données sauvegardées dans {output_file}")
    
def read_spat_resol(image_path, hdu, hduref, coords_ref, pxsl):
    # === 1. Load FITS image ===
    wcs_ref = WCS(hduref[0].header)

    if type(hdu) == astropy.io.fits.hdu.hdulist.HDUList:
        header = hdu[0].header
        data = hdu[0].data.astype(np.float32)
    else:
        header = hdu.header
        data = hdu.data.astype(np.float32)   # type 'hdulist'
    wcs = WCS(header)
    # === 2. Convert pixel coordinates in ref to world coordinates (RA, Dec) ===
    world_coords = wcs_ref.wcs_pix2world(coords_ref, 1)

    # === 3. Convert world coordinates to pixel coordinates in image with higher resolution ===
    coords = wcs.wcs_world2pix(world_coords, 1)

    # === 4. Round to integer pixel values and filter valid pixels inside image with higher resolution ===
    coords_int = np.round(coords).astype(int)

    # verify we don't exceed the fits dimension
    h, w = data.shape
    valid = (coords_int[:, 0] >= 0) & (coords_int[:, 0] < w) & \
            (coords_int[:, 1] >= 0) & (coords_int[:, 1] < h)
    coords_valid = coords_int[valid]
    if coords_valid.shape[0] == 0:
        raise ValueError("No valid pixel coordinates found in the image bounds.")
    # === 5. Extract pixel values from image with higher resolution at the corresponding positions ===
    # pixels_values = data
    new_image = np.zeros((h, w))
    new_image[:,:] = np.nan
    new_image[coords_valid[:, 1], coords_valid[:, 0]] = pxsl
    """
    # Plot image of the galaxy highlighted
    image_data = new_image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_data, origin='lower', cmap='gray', vmin=np.percentile(
        image_data, 5), vmax=np.percentile(image_data, 99))
    ax.scatter(
        coords_valid[:, 0], coords_valid[:, 1], s=1, color='red')  # x, y
    ax.set_title(f"{image_path}")
    fig.canvas.manager.set_window_title(f"Image traitée de {image_path}")
    plt.show()
    """
    return new_image, header

def get_maps_pxls(name_file):
    Rv = 3.1 # from cigale
    # M_sun = 1.9885e30  # [kg]
    L_sun = 3.828e26 # [W]
    # Replace this path with the path to your file (csv or tsv)
    df = pd.read_csv(name_file, sep=r'\s+|,', engine='python')
    # Extracting columns
    sfr       = df['bayes.sfh.sfr'].to_numpy()
    sfr10M   = df['bayes.sfh.sfr10Myrs'].to_numpy()
    sfr100M   = df['bayes.sfh.sfr100Myrs'].to_numpy()
    sm = np.log10(df['bayes.stellar.m_star'].to_numpy())
    e_bv      = df['bayes.attenuation.E_BV_lines'].to_numpy() * Rv
    chi2_red  = df['best.reduced_chi_square'].to_numpy()
    dust_lum = np.log10(df['bayes.dust.luminosity'].to_numpy() / L_sun)
    age_burst = df['bayes.sfh.age_burst'].to_numpy()
    f_burst = df['bayes.sfh.f_burst'].to_numpy()
    tau_main_sfh = df['bayes.sfh.tau_main'].to_numpy()
    UV_slope = df['bayes.attenuation.powerlaw_slope'].to_numpy()
    beta      = df['bayes.param.beta_calz94'].to_numpy()
    irx       = df['bayes.param.IRX'].to_numpy()
    
    sfr_err       = df['bayes.sfh.sfr_err'].to_numpy()
    sfr10M_err   = df['bayes.sfh.sfr10Myrs_err'].to_numpy()
    sfr100M_err   = df['bayes.sfh.sfr100Myrs_err'].to_numpy()
    sm_err = df['bayes.stellar.m_star_err'].to_numpy() / (df['bayes.stellar.m_star'].to_numpy() * np.log(10))
    e_bv_err      = df['bayes.attenuation.E_BV_lines_err'].to_numpy() * Rv
    dust_lum_err = df['bayes.dust.luminosity_err'].to_numpy() / (df['bayes.dust.luminosity'].to_numpy() / np.log(10))
    age_burst_err = df['bayes.sfh.age_burst_err'].to_numpy()
    f_burst_err = df['bayes.sfh.f_burst_err'].to_numpy()
    tau_main_sfh_err = df['bayes.sfh.tau_main_err'].to_numpy()
    UV_slope_err = df['bayes.attenuation.powerlaw_slope_err'].to_numpy()
    beta_err       = df['bayes.param.beta_calz94_err'].to_numpy()
    irx_err       = df['bayes.param.IRX_err'].to_numpy()

    # Store in a list in the desired order
    results = [sfr, sfr10M, sfr100M, sm, e_bv, chi2_red, dust_lum,
               age_burst, f_burst, tau_main_sfh, UV_slope, irx, beta]
    err_results = [sfr_err, sfr10M_err, sfr100M_err, sm_err, e_bv_err, np.zeros(1), dust_lum_err,
                   age_burst_err, f_burst_err, tau_main_sfh_err, UV_slope_err, irx_err, beta_err]
    names = [r'SFR [$M_\odot \mathrm{yr}^{-1}$]',
    r'SFR_10M [$M_\odot \mathrm{yr}^{-1}$]',
    r'SFR_100M [$M_\odot \mathrm{yr}^{-1}$]',
    r'Stellar Mass [$\log_{10} \left( \frac{M_*}{M_\odot} \right)$]', 
    r'Av [mag]',
    r'red_chi²',
    r'Dust luminosity [$\log_{10} \left( \frac{L}{L_\odot} \right)$]',
    r'Age burst [Myr]',
    r'f burst [0 : 1]',
    r'Tau_main [Myr]',
    r'UV_slope',
    r'IRX',
    r'β']
    
    """
    # Exemple d’utilisation
    for i, arr in enumerate(results):
        print(f'Tableau {i+1}, premiers 5 éléments :', arr[:5])
    """
    return results, err_results, names

def get_beta_IRX(name_file):
    df = pd.read_csv(name_file, sep=r'\s+|,', engine='python')
    # Extracting columns
    beta      = df['bayes.param.beta_calz94'].to_numpy()
    irx       = df['bayes.param.IRX'].to_numpy()
    
    beta_err       = df['bayes.param.beta_calz94_err'].to_numpy()
    irx_err       = df['bayes.param.IRX_err'].to_numpy()

    # Store in a list in the desired order
    results = [beta, irx]
    err_results = [beta_err, irx_err]
    names = [r'β', r'IRX']
    
    """
    # Exemple d’utilisation
    for i, arr in enumerate(results):
        print(f'Tableau {i+1}, premiers 5 éléments :', arr[:5])
    """
    return results, err_results, names

def add_scalebar(ax, length_pixels, label, color, location = (0.02, 0.05)):
    """
    Add a horizontal scale bar of a given length in pixels to a matplotlib WCS Axes.
    
    Parameters
    ----------
    ax : matplotlib Axes
        The axis to draw the scale bar on.
    length_pixels : float
        The length of the scale bar in pixels.
    label : str
        The label to show next to the scale bar.
    location : tuple
        The (x, y) location of the left end of the scale bar in Axes coordinates (0–1).
    color : str
        Color of the scale bar and label.
    """
    # Convert axes coordinates to pixel coordinates
    ax.transAxes
    x0, y0 = location

    # Get axis limits to compute data coordinates from Axes coordinates
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Convert Axes coords to data coords
    x_start = xlim[0] + x0 * (xlim[1] - xlim[0])
    y_start = ylim[0] + y0 * (ylim[1] - ylim[0])
    
    # End point of scale bar
    x_end = x_start + length_pixels

    # Draw scale bar
    ax.plot([x_start, x_end], [y_start, y_start], color = color, lw = 2, transform = ax.transData, zorder = 10)

    # Add label
    ax.text(x_start + length_pixels / 2, y_start + 0.01 * (ylim[1] - ylim[0]), label,
            ha = 'center', va = 'bottom', fontsize = 12, color = color,
            bbox = dict(facecolor = 'white', alpha = 0, edgecolor = 'none'))
    
def compute_pixel_distances(pixel_coords, pixel_values):
    """
    Computes the distance from each pixel to the pixel with the maximum value.

    Parameters:
    - pixel_coords: list or array of (y, x) tuples representing pixel coordinates.
    - pixel_values: list or array of pixel values, same length as pixel_coords.

    Returns:
    - distances: array of same length as inputs, containing the Euclidean distance
                 in pixels to the pixel with the maximum value.
    """

    if len(pixel_coords) != len(pixel_values):
        raise ValueError("pixel_coords and pixel_values must have the same length.")

    # Find the index of the pixel with the maximum value
    max_index = np.argmax(pixel_values)
    max_coord = pixel_coords[max_index]

    # Compute Euclidean distances to the max pixel
    distances = np.linalg.norm(pixel_coords - max_coord, axis = 1)
    return distances

def donnees_ALPINE():
    names = np.array([
    "CANDELS_GOODSS_32", "DEIMOS_COSMOS_396844", "DEIMOS_COSMOS_422677", "DEIMOS_COSMOS_539609", "DEIMOS_COSMOS_683613",
    "DEIMOS_COSMOS_818760", "DEIMOS_COSMOS_848185", "DEIMOS_COSMOS_873756", "DEIMOS_COSMOS_881725", "vuds_cosmos_5100969402",
    "vuds_cosmos_5101209780", "vuds_cosmos_5101218326", "vuds_cosmos_5180966608", "vuds_efdcs_530029038",
    "DEIMOS_COSMOS_434239", "DEIMOS_COSMOS_454608", "DEIMOS_COSMOS_627939", "DEIMOS_COSMOS_630594", "DEIMOS_COSMOS_845652",
    "vuds_cosmos_5100541407", "vuds_cosmos_5100559223", "vuds_cosmos_510786441", "vuds_cosmos_5110377875"])
    
    z = np.array([
    4.41, 4.54, 4.44, 5.18, 5.54, 4.56, 5.29, 4.55, 4.58, 4.58, 4.57, 4.57,
    4.53, 4.43, 4.49, 4.58, 4.53, 4.44, 5.31, 4.56, 4.56, 4.46, 4.55])
    
    irx = np.array([
    (0.57, 0.14), (0.57, 0.14), (0.63, 0.11), (0.07, 0.22), (0.51, 0.17), (0.81, 0.14),
    (0.32, 0.09), (1.36, 0.22), (0.66, 0.13), (0.54, 0.19), (0.17, 0.26), (0.51, 0.14),
    (0.76, 0.13), (0.00, 0.16), (0.61, 0.08), (0.33, 0.08), (0.48, 0.08), (0.55, 0.09),
    (-0.20, 0.10), (1.04, 0.09), (0.40, 0.18), (-0.10, 0.10), (0.65, 0.05)])
    
    beta = np.array([
    (-1.20, 0.08), (-1.44, 0.24), (-1.29, 0.20), (-2.42, 0.11), (-1.90, 0.22), (-0.74, 0.17),
    (-1.20, 0.17), (-1.31, 0.31), (-1.20, 0.22), (-1.85, 0.25), (-2.16, 0.15), (-0.98, 0.18),
    (-0.80, 0.24), (-1.92, 0.07), (-1.25, 0.26), (-1.45, 0.20), (-1.46, 0.24), (-1.64, 0.25),
    (-1.73, 0.10), (-1.94, 0.26), (-2.00, 0.22), (-2.00, 0.10), (-1.38, 0.18)])
    """
    delta = np.array([
    (-0.84, 0.31), (-0.62, 0.42), (-0.67, 0.32), None, (0.02, 0.33), (-0.72, 0.26),
    (-1.44, 0.36), None, (-0.67, 0.30), (-0.12, 0.40), None, (-0.85, 0.30),
    (-0.68, 0.29), (-0.64, 0.45), (-0.71, 0.33), (-1.23, 0.43), (-0.75, 0.44), (-0.41, 0.36),
    (-1.84, 0.17), None, (-0.08, 0.45), (-1.19, 0.45), (-0.55, 0.23)])
    
    afuv = np.array([
    (1.46, 0.24), (1.42, 0.23), (1.54, 0.20), (0.67, 0.21), (1.28, 0.26), (1.94, 0.26),
    (1.05, 0.14), (3.03, 0.49), (1.60, 0.23), (1.36, 0.30), (0.81, 0.28), (1.34, 0.24),
    (1.79, 0.26), (0.62, 0.17), (1.49, 0.16), (1.02, 0.13), (1.25, 0.13), (1.37, 0.15),
    (0.43, 0.09), (2.25, 0.19), (1.11, 0.26), (0.48, 0.09), (1.56, 0.09)])
    """
    Av = np.array([
    (0.24, 0.10), (0.32, 0.18), (0.31, 0.13), (0.39, 0.20), (0.58, 0.23), (0.35, 0.13),
    (0.08, 0.05), (1.40, 0.47), (0.31, 0.13), (0.55, 0.27), (0.40, 0.23), (0.20, 0.09),
    (0.33, 0.13), (0.14, 0.09), (0.28, 0.12), (0.11, 0.07), (0.24, 0.14), (0.38, 0.17),
    (0.02, 0.01), (1.25, 0.26), (0.49, 0.27), (0.05, 0.04), (0.34, 0.11)])
    
    SFR = np.array([
    (50.86, 15.89), (76.88, 18.75), (91.97, 18.88), (66.52, 13.38), (54.64, 14.19), (159.53, 53.88),
    (120.35, 30.07), (142.62, 66.46), (87.12, 23.58), (59.00, 21.63), (54.90, 18.01), (82.33, 20.71),
    (70.99, 17.94), (32.50, 6.47), (103.13, 15.75), (62.19, 8.09), (65.79, 5.93), (57.27, 5.52),
    (72.76, 7.19), (99.83, 11.58), (34.92, 8.01), (61.60, 6.62), (161.06, 9.09)])
    
    SM = np.array([
    (7.52e9, 1.83e9), (7.84e9, 2.27e9), (8.43e9, 1.92e9), (5.41e9, 1.46e9), (1.95e10, 7.7e9),
    (5.07e10, 9.63e9), (2.42e10, 5.32e9), (4.26e10, 1.88e10), (1.04e10, 2.5e9), (1.24e10, 4.58e9),
    (1.84e10, 4.73e9), (1.22e11, 1.8e10), (7.17e10, 1.32e10), (1.51e10, 2.35e9), (2.74e10, 7.85e9),
    (6.04e9, 1.55e9), (1.03e10, 3.42e9), (6.28e9, 2.09e9), (5.08e10, 7.7e9), (3.30e10, 1.56e10),
    (9.51e9, 3.16e9), (1.18e10, 2.18e9), (1.41e10, 3.62e9)])
    
    if not len(SM) == len(SFR) == len(Av):
        print("Les donnéees d'ALPINE ne sont pas à la bonne taille")
    
    return [z, irx, beta, Av, SFR, SM, names]

def extract_mstar_av(file_path = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//Smail+21_Mstar_Av.csv'):
    """
    Extrait les colonnes Mstar et Av du fichier Smail+21_Mstar_Av.csv.

    Paramètres :
        file_path (str) : chemin du fichier CSV.

    Retourne :
        tuple : deux listes (Mstar, Av)
    """
    # Lire le fichier en ignorant la première ligne
    df = pd.read_csv(file_path, delim_whitespace = True, skiprows=1, names=["Mstar", "Av", "Flag"])
    
    # Extraire les colonnes
    mstar_list = np.log10(df["Mstar"].tolist())
    av_list = df["Av"].tolist()
    
    return mstar_list, av_list

def plot_SFR_M(SFR, SFR_err, M, M_err, radius, galaxy_name):
    SFR_alpine, SFR_alpine_err = np.log10(donnees_ALPINE()[4][:, 0]), donnees_ALPINE()[4][:, 1] / (donnees_ALPINE()[4][:, 0] * np.log(10))
    M_alpine, M_alpine_err = np.log10(donnees_ALPINE()[5][:, 0]), donnees_ALPINE()[5][:, 1] / (donnees_ALPINE()[5][:, 0] * np.log(10))
    log_SFR_err = SFR_err / (SFR * np.log(10))
    log_SFR = np.log10(SFR)
    scale = cosmo.kpc_proper_per_arcmin(z_spec)  # Résultat avec unité, ex : 486.95 kpc/arcmin
    scale_arcsec = (scale / 60).value  # Pour avoir kpc / arcsec
    radius_kpc = radius * 0.04 * scale_arcsec
    fig, ax = plt.subplots(figsize = (15, 10))
    
    M_min, M_max = min(min(M), min(M_alpine)), max(max(M), max(M_alpine))
    # SFR_min, SFR_max = min(min(SFR), min(SFR_alpine)), max(max(SFR), max(SFR_alpine))
    X = np.linspace(M_min, M_max, 5000)
    r = np.log10(1 + z_spec)
    m0 = 0.5
    m0_err = 0.07
    a0 = 1.5
    a0_err = 0.15
    a1 = 0.3
    a1_err = 0.08
    m1 = 0.36
    m1_err = 0.3
    a2 = 2.5
    a2_err = 0.6
    y = [m - 9 - m0 + a0*r - a1 * max(0, m - 9 - m1 - a2 * r)**2 for m in X]
    y_min = np.array([m - 9 - (m0 + m0_err) + (a0 - a0_err)*r - (a1 + a1_err) * max(0, m - 9 - (m1 - m1_err) - (a2 - a2_err) * r)**2 for m in X])
    y_max = np.array([m - 9 - (m0 - m0_err) + (a0 + a0_err)*r - (a1 - a1_err) * max(0, m - 9 - (m1 + m1_err) - (a2 + a2_err) * r)**2 for m in X])
    
    # ax.scatter(M_alpine, SFR_alpine, color = 'r', marker = 's', label = 'ALPINE')
    ax.errorbar(M_alpine, SFR_alpine, xerr = M_alpine_err, yerr = SFR_alpine_err, fmt = 'gs', capsize = 3, alpha = 0.6, linestyle = 'none', label = 'ALPINE')
    sc = ax.scatter(M, log_SFR, marker = 'o', c = radius_kpc, cmap = 'viridis', label = galaxy_name)
    plt.colorbar(sc, label = 'Radius [kpc]')
    ax.set_ylabel(r'SFR [$ \log_{10} \left( M_\odot \mathrm{yr}^{-1} \right)$]', size = 20)
    ax.set_xlabel(r'Stellar Mass [$\log_{10} \left( \frac{M_*}{M_\odot} \right)$]', size = 20)
    """
    indices = np.argsort(M)
    M_sorted = M[indices]
    y_min_sorted = y_min[indices]
    y_max_sorted = y_max[indices]
    """
    
    ax.plot(X, y, 'r', label = "Main Sequence")
    # ax.errorbar(M, SFR, xerr = M_err, yerr = log_SFR_err, fmt = 'bo', capsize = 3, alpha = 0.4, linestyle = 'none')
    # Valeurs des barres d'erreur à afficher
    M_mean_err = np.mean(M_err)
    SFR_mean_err = np.mean(log_SFR_err)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # On convertit l'erreur (par ex. 0.2 en unités réelles) en % relatif
    xerr_rel = M_mean_err / (xlim[1] - xlim[0])
    yerr_rel = SFR_mean_err / (ylim[1] - ylim[0])
    # Ajouter un point d’erreur fictif en coord. Axes
    ax.errorbar(0.75, 0.25, xerr = xerr_rel, yerr = yerr_rel, fmt = 'bo', capsize = 4, transform = ax.transAxes)
    # Optionnel : afficher le texte à côté
    # ax.text(0.97, 0.05, f'±{M_mean_err}, ±{SFR_mean_err}', transform = ax.transAxes, fontsize = 9, va = 'center', color = 'b')
    
    ax.fill_between(X, y_min, y_max, color = 'r', alpha = 0.1)
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    plt.legend()
    fig.canvas.manager.set_window_title(f"SFR vs M* for galaxy {galaxy_name}")
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "SFR_vs_M*_" + galaxy_name + '.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def plot_Av_M(Av, Av_err, M, M_err, radius, galaxy_name):
    Av_alpine, Av_alpine_err = donnees_ALPINE()[3][:, 0], donnees_ALPINE()[3][:, 1]
    M_alpine, M_alpine_err = np.log10(donnees_ALPINE()[5][:, 0]), donnees_ALPINE()[5][:, 1] / (donnees_ALPINE()[5][:, 0] * np.log(10))
    M_Smail, Av_Smail = extract_mstar_av(file_path = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//Smail+21_Mstar_Av.csv')
    scale = cosmo.kpc_proper_per_arcmin(z_spec)  # Résultat avec unité, ex : 486.95 kpc/arcmin
    scale_arcsec = (scale / 60).value  # Pour avoir kpc / arcsec
    radius_kpc = radius * 0.04 * scale_arcsec
    fig, ax = plt.subplots(figsize = (15, 10))
    
    # ax.scatter(M_alpine, Av_alpine, color = 'r', marker = 's', label = 'ALPINE')
    ax.errorbar(M_alpine, Av_alpine, xerr = M_alpine_err, yerr = Av_alpine_err, fmt = 'gs', capsize = 3, alpha = 0.6, linestyle = 'none', label = 'ALPINE')
    ax.scatter(M_Smail, Av_Smail, color = 'r', marker = '^', label = 'Smail and al.', alpha = 0.6)
    sc = ax.scatter(M, Av, marker = 'o', c = radius_kpc, cmap = 'viridis', label = galaxy_name)
    plt.colorbar(sc, label = 'Radius [kpc]')
    ax.set_ylabel(r'Av [mag]', size = 20)
    ax.set_xlabel(r'Stellar Mass [$\log_{10} \left( \frac{M_*}{M_\odot} \right)$]', size = 20)
    
    # ax.errorbar(M, Av, xerr = M_err, yerr = Av_err, fmt = 'bo', capsize = 3, alpha = 0.4, linestyle = 'none')
    # Valeurs des barres d'erreur à afficher
    M_mean_err = np.mean(M_err)
    Av_mean_err = np.mean(Av_err)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # On convertit l'erreur (par ex. 0.2 en unités réelles) en % relatif
    xerr_rel = M_mean_err / (xlim[1] - xlim[0])
    yerr_rel = Av_mean_err / (ylim[1] - ylim[0])
    # Ajouter un point d’erreur fictif en coord. Axes
    ax.errorbar(0.75, 0.85, xerr = xerr_rel, yerr = yerr_rel, fmt = 'bo', capsize = 4, transform = ax.transAxes)
    # Optionnel : afficher le texte à côté
    # ax.text(0.97, 0.05, f'±{M_mean_err}, ±{Av_mean_err}', transform = ax.transAxes, fontsize = 9, va = 'center', color = 'b')

    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    plt.legend()
    fig.canvas.manager.set_window_title(f"Av vs M* for galaxy {galaxy_name}")
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "Av_vs_M*_" + galaxy_name + '.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def plot_IRX_β(IRX, IRX_err, β, β_err, radius, galaxy_name):
    IRX_alpine, IRX_alpine_err = np.log10(donnees_ALPINE()[1][:, 0]), donnees_ALPINE()[1][:, 1] / (donnees_ALPINE()[1][:, 0] * np.log(10))
    β_alpine, β_alpine_err = donnees_ALPINE()[2][:, 0], donnees_ALPINE()[2][:, 1]
    log_IRX_err = IRX_err / (IRX * np.log(10))
    log_IRX = np.log10(IRX)
    scale = cosmo.kpc_proper_per_arcmin(z_spec)  # Résultat avec unité, ex : 486.95 kpc/arcmin
    scale_arcsec = (scale / 60).value  # Pour avoir kpc / arcsec
    radius_kpc = radius * 0.04 * scale_arcsec
    fig, ax = plt.subplots(figsize = (15, 10))
    
    β_min, β_max = min(min(β), min(β_alpine)), max(max(β), max(β_alpine))
    # IRX_min, IRX_max = min(min(IRX), min(IRX_alpine)), max(max(IRX), max(IRX_alpine))
    X = np.linspace(β_min, β_max, 5000)
    y_starbust = [np.log10(1.67 * (10**(0.4 * (2.13 * m + 5.57)) - 1)) for m in X]
    y_SMC = [np.log10(1.79 * (10**(0.4 * (1.07 * m + 2.79)) - 1)) for m in X]

    # ax.scatter(M_alpine, IRX_alpine, color = 'r', marker = 's', label = 'ALPINE')
    ax.errorbar(β_alpine, IRX_alpine, xerr = β_alpine_err, yerr = abs(IRX_alpine_err), fmt = 'ms', capsize = 3, alpha = 0.6, linestyle = 'none', label = 'ALPINE')
    sc = ax.scatter(β, log_IRX, marker = 'o', c = radius_kpc, cmap = 'viridis', label = galaxy_name)
    plt.colorbar(sc, label = 'Radius [kpc]')
    ax.set_ylabel(r'$\log_{10} \left( IRX \right)$', size = 20)
    ax.set_xlabel(r'β', size = 20)
    
    ax.plot(X, y_starbust, 'r', label = 'Starbust')
    ax.plot(X, y_SMC, 'black', label = 'SMC')
    
    # ax.errorbar(M, IRX, xerr = M_err, yerr = log_IRX_err, fmt = 'bo', capsize = 3, alpha = 0.4, linestyle = 'none')
    # Valeurs des barres d'erreur à afficher
    β_mean_err = np.mean(β_err)
    IRX_mean_err = np.mean(log_IRX_err)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # On convertit l'erreur (par ex. 0.2 en unités réelles) en % relatif
    xerr_rel = β_mean_err / (xlim[1] - xlim[0])
    yerr_rel = IRX_mean_err / (ylim[1] - ylim[0])
    # Ajouter un point d’erreur fictif en coord. Axes
    ax.errorbar(0.85, 0.15, xerr = xerr_rel, yerr = yerr_rel, fmt = 'bo', capsize = 4, transform = ax.transAxes)
    # Optionnel : afficher le texte à côté
    # ax.text(0.97, 0.05, f'±{β_mean_err}, ±{IRX_mean_err}', transform = ax.transAxes, fontsize = 9, va = 'center', color = 'b')
    
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    plt.legend()
    fig.canvas.manager.set_window_title(f"log10(IRX) vs β for galaxy {galaxy_name}")
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "log10(IRX)_vs_β_" + galaxy_name + '.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def plot_IRX_β2(IRX, IRX_err, β, β_err, radius, galaxy_name):
    IRX_alpine, IRX_alpine_err = np.log10(donnees_ALPINE()[1][:, 0]), donnees_ALPINE()[1][:, 1] / (donnees_ALPINE()[1][:, 0] * np.log(10))
    β_alpine, β_alpine_err = donnees_ALPINE()[2][:, 0], donnees_ALPINE()[2][:, 1]
    log_IRX_err = IRX_err / (IRX * np.log(10))
    log_IRX = np.log10(IRX)
    scale = cosmo.kpc_proper_per_arcmin(z_spec)  # Résultat avec unité, ex : 486.95 kpc/arcmin
    scale_arcsec = (scale / 60).value  # Pour avoir kpc / arcsec
    radius_kpc = radius * 0.04 * scale_arcsec
    fig, ax = plt.subplots(figsize = (15, 10))
    
    β_min, β_max = min(min(β), min(β_alpine)), max(max(β), max(β_alpine))
    # IRX_min, IRX_max = min(min(IRX), min(IRX_alpine)), max(max(IRX), max(IRX_alpine))
    X = np.linspace(β_min, β_max, 5000)
    y_starbust = [np.log10(1.67 * (10**(0.4 * (2.13 * m + 5.57)) - 1)) for m in X]
    y_SMC = [np.log10(1.79 * (10**(0.4 * (1.07 * m + 2.79)) - 1)) for m in X]

    # ax.scatter(M_alpine, IRX_alpine, color = 'r', marker = 's', label = 'ALPINE')
    ax.errorbar(β_alpine, IRX_alpine, xerr = β_alpine_err, yerr = abs(IRX_alpine_err), fmt = 'ms', capsize = 3, alpha = 0.6, linestyle = 'none', label = 'ALPINE')
    sc = ax.scatter(β, log_IRX, marker = 'o', c = radius_kpc, cmap = 'viridis', label = galaxy_name)
    plt.colorbar(sc, label = 'Radius [kpc]')
    ax.set_ylabel(r'$\log_{10} \left( IRX \right)$', size = 20)
    ax.set_xlabel(r'β', size = 20)
    
    ax.plot(X, y_starbust, 'r', label = 'Starbust')
    ax.plot(X, y_SMC, 'black', label = 'SMC')
    
    # ax.errorbar(M, IRX, xerr = M_err, yerr = log_IRX_err, fmt = 'bo', capsize = 3, alpha = 0.4, linestyle = 'none')
    # Valeurs des barres d'erreur à afficher
    β_mean_err = np.mean(β_err)
    IRX_mean_err = np.mean(log_IRX_err)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # On convertit l'erreur (par ex. 0.2 en unités réelles) en % relatif
    xerr_rel = β_mean_err / (xlim[1] - xlim[0])
    yerr_rel = IRX_mean_err / (ylim[1] - ylim[0])
    # Ajouter un point d’erreur fictif en coord. Axes
    ax.errorbar(0.85, 0.15, xerr = xerr_rel, yerr = yerr_rel, fmt = 'bo', capsize = 4, transform = ax.transAxes)
    # Optionnel : afficher le texte à côté
    # ax.text(0.97, 0.05, f'±{β_mean_err}, ±{IRX_mean_err}', transform = ax.transAxes, fontsize = 9, va = 'center', color = 'b')
    
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    ax.set_xlim(-3, 0)   # Pour les abscisses (x)
    ax.set_ylim(-1, 3)   # Pour les ordonnées (y)
    plt.legend()
    fig.canvas.manager.set_window_title(f"log10(IRX) vs β for galaxy {galaxy_name} zoom")
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "log10(IRX)_vs_β_" + galaxy_name + '_zoom.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def complete_maps(list_maps, maps_names):
    IRX_name = r'$\log_{10} \left( IRX \right)$'
    IRX = np.copy(list_maps[-2])
    beta = list_maps[-1]
    log_SFR = np.log10(list_maps[0])
    M = list_maps[3]
    
    r = np.log10(1 + z_spec)
    m0 = 0.5
    a0 = 1.5
    a1 = 0.3
    m1 = 0.36
    a2 = 2.5
    log_MS_SFR = M - 9 - m0 + a0*r - a1 * np.maximum(0, M - 9 - m1 - a2 * r)**2
    Δ_MS = log_SFR - log_MS_SFR
    Δ_MS_name = r" ΔMS = $\log_{10} \left( \frac{\mathrm{SFR}\ [M_\odot\, \mathrm{yr}^{-1}]}{\mathrm{SFR}_{\mathrm{MS}}\ [M_\odot\, \mathrm{yr}^{-1}]} \right)$"
    
    IRX_SMC = 1.79 * (10**(0.4 * (1.07 * beta + 2.79)) - 1)
    Δ_SMC = np.log10(IRX / IRX_SMC)
    Δ_SMC_name = r'ΔSMC = $ \log_{10} \left( \frac{IRX}{IRX_{MS}} \right) $'
    
    maps_names[-2] = IRX_name
    maps_names[-1] = Δ_SMC_name
    maps_names.append(Δ_MS_name)
    
    list_maps[-2] = np.log10(IRX)
    list_maps[-1] = Δ_SMC
    list_maps.append(Δ_MS)
    
    return list_maps, maps_names

def cercle_resol(ax, relative_pos):
    # === Paramètres ===
    radius_px = 2               # rayon en pixels image

    # === Convertir la position de ax.transAxes vers ax.transData ===
    # --> De coordonnées normalisées à coordonnées image
    display_coord = ax.transAxes.transform(relative_pos)      # (x, y) en pixels écran
    data_coord = ax.transData.inverted().transform(display_coord)  # (x, y) en pixels image

    # === Créer le cercle en coordonnées image ===
    circle = Circle(data_coord, radius_px, edgecolor = 'red', facecolor='none', lw=1.5)
    ax.add_patch(circle)
    
def draw_pixel_shape_outline_xy(pixel_coords_xy, ax = None, color = 'grey', linewidth = 1.5):
    """
    Dessine les contours d'une forme définie par une liste de pixels (x, y).

    Paramètres
    ----------
    pixel_coords_xy : array-like de shape (N, 2)
        Liste de tuples ou array numpy contenant les coordonnées (x, y) des pixels de la forme.
    ax : matplotlib.axes._axes.Axes ou None
        Axe matplotlib sur lequel dessiner. Si None, crée une nouvelle figure.
    color : str
        Couleur du contour.
    linewidth : float
        Épaisseur de la ligne de contour.
    """
    pixel_coords_xy = np.array(pixel_coords_xy)
    if pixel_coords_xy.shape[1] != 2:
        raise ValueError("pixel_coords_xy must be of shape (N, 2) with (x, y) coordinates.")
    
    # Inverser (x, y) -> (y, x) pour la logique image
    pixel_coords = pixel_coords_xy[:, [1, 0]]  # devient (y, x)

    # Créer une image binaire
    y_max, x_max = pixel_coords.max(axis=0) + 2
    binary_image = np.zeros((y_max + 1, x_max + 1), dtype=np.uint8)
    binary_image[pixel_coords[:, 0], pixel_coords[:, 1]] = 1

    # Détection des contours
    contours = measure.find_contours(binary_image, level=0.5)

    # Créer l'axe si nécessaire
    if ax is None:
        fig, ax = plt.subplots()
        ax.imshow(binary_image, origin='lower', cmap='gray')

    # Tracer les contours (dans l'ordre x, y)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)

    ax.set_aspect('equal')
    
def print_maps(list_maps1, maps_names1, map_header, galaxy_name, rgb_img, pixel_coords_xy):
    """Read and plot the galaxy maps."""
    list_maps, maps_names = complete_maps(list_maps1, maps_names1)
    list_maps.append(rgb_img)
    maps_names.append("RGB image")
    N_maps = len(list_maps)
    colormaps = ["plasma", "plasma", "plasma", "inferno", "magma", "viridis", "cividis", "coolwarm", "Wistia", "seismic", "seismic", "hot", "hot", "inferno", "seismic"]
    fig, axs = plt.subplots(3, 5, figsize = (15, 10))
    axs = axs.flatten()
    for i in range(N_maps):
        ax = axs[i]
        """
        ax.coords['ra'].set_axislabel('')
        ax.coords['dec'].set_axislabel('')
        
        # Étiquettes RA/Dec
        if i//8 == 1:
            ax.coords['ra'].set_axislabel('RA (J2000)', fontsize = 6)
        ax.coords['dec'].set_axislabel('Dec (J2000)', fontsize = 6)
        """
        im = ax.imshow(list_maps[i], origin='lower', cmap = colormaps[i])
        # ax.axis('off')
        # im = ax.imshow(list_maps[i], origin='lower', cmap = colormaps[i], vmin=np.percentile(list_maps[i], 5), vmax=np.percentile(list_maps[i], 99), label = maps_names[i])
        # plt.colorbar(im, ax = ax, orientation='vertical')

        ax_divider = make_axes_locatable(ax)
        ax.tick_params(axis = 'both', which = 'both', labelsize = 2, length = 2, pad = 1)
        ax.text(0.98, 0.95, maps_names[i],transform=ax.transAxes,
                fontsize = 8, color='black', ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5, pad=2, edgecolor='none'))
        scale = cosmo.kpc_proper_per_arcmin(z_spec)  # Résultat avec unité, ex : 486.95 kpc/arcmin
        scale_arcsec = (scale / 60).value  # Pour avoir kpc / arcsec
        kpc = np.round(scale_arcsec * 0.2 * scl, 1)
        pxl_len = 5 * scl
        if i == N_maps - 1:
            add_scalebar(ax, length_pixels = pxl_len, label = f'{kpc} kpc', color = "white", location = (0.05, 0.05))
        else:
            add_scalebar(ax, length_pixels = pxl_len, label = f'{kpc} kpc', color = "black", location = (0.05, 0.05))
            # Add an Axes to the right of the main Axes.
            cax = ax_divider.append_axes("top", size = "2%", pad = "2%")
            cbar = plt.colorbar(im, cax = cax, orientation = 'horizontal')
            cbar.ax.tick_params(axis='x', labelsize = 6, labelbottom = False, labeltop = True, top = True)
            cbar.ax.tick_params(axis = 'y', left = False, right = False, labelleft = False, labelright = False) # on cible l'axe Y 
        cercle_resol(ax, relative_pos = (0.05, 0.25))  # en coordonnées normalisées (Axes))
        draw_pixel_shape_outline_xy(pixel_coords_xy, ax = ax, color = 'grey', linewidth = 1.5)
    fig.canvas.manager.set_window_title("Maps de galaxy : " + galaxy_name)
    plt.subplots_adjust(left = 0.02, bottom = 0.05, right = 0.98, top = 0.95, wspace = 0.15, hspace = 0.15) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "Maps_" + galaxy_name + '.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def plot_isocontours(
    ax, image, cmap,
    levels=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    label_level=50, label_text=None,
    label_position='topright',
    single_color=None  # option pour une couleur unique
):
    max_val = np.nanmax(image)
    norm_levels = [max_val * lvl / 100. for lvl in levels]

    if single_color is None:
        # palette dégradée si pas de couleur unique spécifiée
        colors = plt.get_cmap(cmap)(np.linspace(0.2, 1, len(norm_levels)))
    else:
        # une seule couleur pour tous les niveaux
        colors = [single_color] * len(norm_levels)

    contours = ax.contour(image, levels=norm_levels, colors=colors, linewidths=1.5)

    ax.clabel(contours, fmt=lambda x: f'{(x / max_val * 100):.0f}%', fontsize=12, inline=1)

    if label_text is None:
        label_text = f'{label_level} %'

    try:
        idx = levels.index(label_level)
    except ValueError:
        print(f"Niveau {label_level}% non trouvé dans la liste des niveaux.")
        idx = None

    if idx is not None:
        color = colors[idx]
        if label_position == 'topright':
            x_pos, y_pos, ha, va = 0.98, 0.98, 'right', 'top'
        elif label_position == 'topleft':
            x_pos, y_pos, ha, va = 0.02, 0.98, 'left', 'top'
        else:
            x_pos, y_pos, ha, va = 0.98, 0.98, 'right', 'top'

        ax.text(
            x_pos, y_pos, label_text,
            color=color, fontsize=10, fontweight='bold',
            ha=ha, va=va,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)
        )

    return contours
    
def print_maps2(list_maps1, maps_names1, map_header, galaxy_name, pixel_coords_xy):
    """Read and plot the galaxy maps."""
    list_maps, maps_names = list_maps1, maps_names1
    maps = [list_maps[3], list_maps[4], list_maps[3], list_maps[6]]
    maps_name = [maps_names[3], maps_names[4], maps_names[3], maps_names[6]]
    colormaps = ["viridis", "magma", "Blues", "Oranges"]
    # Colormaps contrastantes
    contrast_cmaps = ["plasma", "winter", "Wistia", "Purples"]
    fig, axs = plt.subplots(1, 2, figsize = (15, 10))
    for i in range(2):
        ax = axs[i]
        """
        ax.coords['ra'].set_axislabel('')
        ax.coords['dec'].set_axislabel('')
        
        # Étiquettes RA/Dec
        if i//8 == 1:
            ax.coords['ra'].set_axislabel('RA (J2000)', fontsize = 6)
        ax.coords['dec'].set_axislabel('Dec (J2000)', fontsize = 6)
        """
        ax_divider = make_axes_locatable(ax)
        ax.tick_params(axis = 'both', which = 'both', labelsize = 8, length = 2, pad = 1)
        """
        ax.text(0.98, 0.95, maps_name[2*i] +' + '+ maps_name[2*i + 1],transform=ax.transAxes,
                fontsize = 8, color='black', ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5, pad=2, edgecolor='none'))
        """
        scale = cosmo.kpc_proper_per_arcmin(z_spec)  # Résultat avec unité, ex : 486.95 kpc/arcmin
        scale_arcsec = (scale / 60).value  # Pour avoir kpc / arcsec
        kpc = np.round(scale_arcsec * 0.2 * scl, 1)
        pxl_len = 5 * scl
        
        plot_isocontours(ax = ax, image = maps[2*i], cmap = contrast_cmaps[2*i], levels=[70, 90], label_level = 90, label_text = maps_name[2*i],
        label_position = 'topleft', single_color = 'red')
        # ax.axis('off')
        # im = ax.imshow(list_maps[i], origin='lower', cmap = colormaps[i], vmin=np.percentile(list_maps[i], 5), vmax=np.percentile(list_maps[i], 99), label = maps_names[i])
        # plt.colorbar(im, ax = ax, orientation='vertical')
        """
        im1 = ax.imshow(maps[2*i], origin='lower', cmap = colormaps[2*i], alpha = 0.8, label = maps_name[2*i])
        add_scalebar(ax, length_pixels = pxl_len, label = f'{kpc} kpc', color = "black", location = (0.05, 0.05))
        # Add an Axes to the right of the main Axes.
        cax1 = ax_divider.append_axes("top", size="3%", pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
        cbar1.ax.tick_params(axis='x', labelsize=6, labelbottom=False, labeltop=True, top=True)
        cbar1.ax.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
        """
        im2 = ax.imshow(maps[2*i + 1], origin = 'lower', cmap = colormaps[2*i + 1], alpha = 0.8, label = maps_name[2*i + 1])
        plot_isocontours(ax = ax, image = maps[2*i + 1], cmap = contrast_cmaps[2*i + 1], levels=[70, 90], label_level = 90, label_text = maps_name[2*i + 1],
        label_position = 'topright', single_color = 'blue')
        # ax.axis('off')
        # im = ax.imshow(list_maps[i], origin='lower', cmap = colormaps[i], vmin=np.percentile(list_maps[i], 5), vmax=np.percentile(list_maps[i], 99), label = maps_names[i])
        # plt.colorbar(im, ax = ax, orientation='vertical')

        add_scalebar(ax, length_pixels = pxl_len, label = f'{kpc} kpc', color = "black", location = (0.05, 0.05))
        # Add an Axes to the right of the main Axes.
        # --- Deuxième map superposée : colorbar à droite ---
        cax2 = ax_divider.append_axes("right", size="3%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2, orientation='vertical')
        cbar2.ax.tick_params(axis='y', labelsize = 8)
        
        cercle_resol(ax, relative_pos = (0.05, 0.25))  # en coordonnées normalisées (Axes))
        draw_pixel_shape_outline_xy(pixel_coords_xy, ax = ax, color = 'grey', linewidth = 1.5)
    fig.canvas.manager.set_window_title("Maps doubles de galaxy : " + galaxy_name)
    plt.subplots_adjust(left = 0.02, bottom = 0.05, right = 0.98, top = 0.95, wspace = 0.15, hspace = 0.15) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "Maps_doubles" + galaxy_name + '.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def print_rgb(map_header, galaxy_name, rgb_img, pixel_coords_xy):
    """Read and plot the galaxy maps."""
    wcs = WCS(map_header)
    fig, ax = plt.subplots(figsize = (15, 10), subplot_kw = {'projection': wcs})
    ax.coords['ra'].set_axislabel('')
    ax.coords['dec'].set_axislabel('')
    
    # Étiquettes RA/Dec
    ax.coords['ra'].set_axislabel('RA (J2000)', fontsize = 12)
    ax.coords['dec'].set_axislabel('Dec (J2000)', fontsize = 12)
    
    ax.imshow(rgb_img, origin='lower')
    # ax.coords.grid(True, color = 'white', ls = 'dotted')
    # im = ax.imshow(list_maps[i], origin='lower', cmap = colormaps[i], vmin=np.percentile(list_maps[i], 5), vmax=np.percentile(list_maps[i], 99), label = maps_names[i])
    # plt.colorbar(im, ax = ax, orientation='vertical')
    """
    ax_divider = make_axes_locatable(ax)
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    # Add an Axes to the right of the main Axes.
    cax = ax_divider.append_axes("top", size="4%", pad="2%")
    cbar = plt.colorbar(im, cax = cax, orientation = 'horizontal')
    cbar.ax.tick_params(axis='x', labelsize = 20, labelbottom = False, labeltop = True, top = True)
    cbar.ax.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False) # on cible l'axe Y 
    """
    # Titre incrusté dans l'image (en haut à gauche)
    ax.text(0.98, 0.95, "blue : 115W, green : 277W, red : 444W",transform=ax.transAxes,
            fontsize = 10, color='black', ha='right', va='top', bbox=dict(facecolor='white', alpha = 0.7, pad=2, edgecolor='none'))
    scale = cosmo.kpc_proper_per_arcmin(z_spec)  # Résultat avec unité, ex : 486.95 kpc/arcmin
    scale_arcsec = (scale / 60).value  # Pour avoir kpc / arcsec
    kpc = np.round(scale_arcsec * 0.2 * scl, 1)
    pxl_len = 5 * scl
    add_scalebar(ax, length_pixels = pxl_len, label = f'{kpc} kpc', color = "white", location = (0.05, 0.05))
    draw_pixel_shape_outline_xy(pixel_coords_xy, ax = ax, color = 'grey', linewidth = 1.5)
    fig.canvas.manager.set_window_title("RGB de galaxy : " + galaxy_name)
    # plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "RGB_" + galaxy_name + '.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    
def print_cutout_imgs(map_header, galaxy_name, list_imgs, names_imgs):
    """Read and plot the galaxy maps."""
    wcs = WCS(map_header)
    names_imgs.append("ALMA : band 7")
    N_imgs = len(list_imgs)
    lgn = N_imgs//3 + N_imgs%3
    row = 3
    fig, axs = plt.subplots(lgn, row, figsize = (15, 10), subplot_kw = {'projection': wcs})
    axs = axs.flatten()
    for i in range(N_imgs):
        ax = axs[i]
        ax.coords['ra'].set_axislabel('')
        ax.coords['dec'].set_axislabel('')
        
        # Étiquettes RA/Dec
        if i//row == row-1:
            ax.coords['ra'].set_axislabel('RA (J2000)', fontsize = 6)
        if i%row == 0:
            ax.coords['dec'].set_axislabel('Dec (J2000)', fontsize = 6)
            
        vmin = np.percentile(list_imgs[i], 5)
        vmax = np.percentile(list_imgs[i], 99)
        ax.imshow(list_imgs[i], origin = 'lower', cmap = 'grey', vmin = vmin, vmax = vmax)
        # ax.coords.grid(True, color = 'white', ls = 'dotted')
        # im = ax.imshow(list_maps[i], origin='lower', cmap = colormaps[i], vmin=np.percentile(list_maps[i], 5), vmax=np.percentile(list_maps[i], 99), label = maps_names[i])
        # plt.colorbar(im, ax = ax, orientation='vertical')
        ax.tick_params(axis = 'x', labelsize = 6)
        ax.tick_params(axis = 'y', labelsize = 6)
        """
        # Add an Axes to the right of the main Axes.
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="4%", pad="2%")
        cbar = plt.colorbar(im, cax = cax, orientation = 'horizontal')
        cbar.ax.tick_params(axis='x', labelsize = 20, labelbottom = False, labeltop = True, top = True)
        cbar.ax.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False) # on cible l'axe Y 
        """
        # Titre incrusté dans l'image (en haut à gauche)
        ax.text(0.98, 0.95, names_imgs[i], transform = ax.transAxes,
                fontsize = 10, color='black', ha='right', va='top', bbox=dict(facecolor='white', alpha = 0.7, pad=2, edgecolor='none'))
        scale = cosmo.kpc_proper_per_arcmin(z_spec)  # Résultat avec unité, ex : 486.95 kpc/arcmin
        scale_arcsec = (scale / 60).value  # Pour avoir kpc / arcsec
        kpc = np.round(scale_arcsec * 0.2 * scl, 1)
        pxl_len = 5 * scl
        add_scalebar(ax, length_pixels = pxl_len, label = f'{kpc} kpc', color = "white", location = (0.05, 0.05))
        cercle_resol(ax, relative_pos = (0.05, 0.25))  # en coordonnées normalisées (Axes))
    fig.canvas.manager.set_window_title("cutout_imgs de galaxy : " + galaxy_name)
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.96, top = 0.9, wspace = 0.2, hspace = 0.2) # plt.tight_layout(pad = 0.1, w_pad = 0.1, h_pad = 0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    fig.savefig(os.path.join("final_images", "cutout_imgs_" + galaxy_name + '.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()

def convert_latex_to_excel_header(header):
    replacements = {
        r'$': '',  # supprime les dollars LaTeX
        r'\,': '',  # supprime les espaces LaTeX
        r'\mathrm{yr}': 'yr',
        r'\log_{10}': 'log10',
        r'\left(': '(',
        r'\right)': ')',
        r'\frac{M}{M_\odot}': 'M/Msun',
        r'\frac{M_dust}{M_\odot}': 'Mdust/Msun',
        r'\frac{L}{L_\odot}': 'L/Lsun',
        r'_': ' ',  # underscore devient espace
        r'\odot': 'sun',
        r'²': '^2',
        r'chi²': 'chi^2'
    }

    for pattern, repl in replacements.items():
        header = header.replace(pattern, repl)

    # Nettoyage final : espace avant les [ et retrait espaces multiples
    header = header.replace('[', ' [').replace('  ', ' ').strip()
    return header

def excel_results():
    os.makedirs("out", exist_ok = True)
    Int_analys, Int_analys_err, name_maps = get_maps_pxls(name_file=os.path.join("out", "results.txt"))
    name_maps_excel = [convert_latex_to_excel_header(h) for h in name_maps]

    # Conversion des valeurs et erreurs depuis les ndarray
    values = [v.item() for v in Int_analys]
    errors = [e.item() for e in Int_analys_err]

    output_path = "Integrated_analysis_results.csv"
    with open(output_path, "w") as f:
        # Write headers
        f.write(",".join(name_maps_excel) + "\n")
        # Write values with uncertainties in 'value ± error' format
        f.write(",".join(f"{v:.3f} +/- {e:.3f}" for v, e in zip(values, errors)) + "\n")
    

def main():
    """Main function to run CIGALE and process images' results."""
    
    # === CONFIGURATION ===
    output_csv = 'cigale_integrated_sed.csv'
    output_multi='cigale_spatially_resolved.csv'
    center_coord = (x0, y0)
    npy_direc = "psf_cubes"
    psf_files, psf_imgs = get_psf_from_npy(npy_direc)
    hdu_ref = hdu_jwst_coord

    # Reference galaxy pixel coordinates in reference image (as (x, y) = (col, row))
    filename = fits_jwst_coord.replace('.fits', '.cat') + "_galaxy_ref_coords.txt"
    """
    Read saved (x, y) coordinates from a file.
    Returns a list of (x, y) tuples.
    """
    
    # Step 0 : get back the coordinates of the galaxy for JWST
    coords = []
    with open(filename, "r") as f:
        for line in f:
            x_str, y_str = line.strip().split()
            coords.append([int(x_str), int(y_str)])
            coords_ref = np.array(np.copy(coords))
            
    # Step 0.5 : Get the maps of the spatially resolved SED Analysis
    maps, maps_err, name_maps = get_maps_pxls(name_file = "results.txt")
    # Get an excel file of the intersting values of the SED fitting
    excel_results()
    
    maps_list = []
    cutout_list = []
    cutout_names_list = []
    maps_header = 0
    # Loop over all PSF images
    os.makedirs("final_images", exist_ok = True)
    for j, psf_src in enumerate(psf_imgs):
        # Step 1 : Load the associated galaxy image
        image_path = psf_files[j].replace('_psf.npy', '.fits')
        matched_img_path = os.path.join("matched_images", 'matched_' + image_path)
        hdu = fits.open(image_path, memmap = True)
        matched_hdu = fits.open(matched_img_path, memmap = True)
        
        # Step 2 : create cutout images to print them and save them
        cutout_image, cutout_header = cutout_fits2(center_coord, hdu, size)
        # Create a new HDU with the modified data and original header
        cutout_hdu = fits.PrimaryHDU(data = cutout_image, header = cutout_header)
        # save fits
        cutout_hdu.writeto(os.path.join("final_images", "final_" + image_path), overwrite = True)
        cutout_list.append(cutout_image)
        cutout_names_list.append(extract_filter_name(image_path))
        
        cutout_matched_image, cutout_matched_header = cutout_fits2(center_coord, matched_hdu, size)
        # Create a new HDU with the modified data and original header
        cutout_matched_hdu = fits.PrimaryHDU(data = cutout_matched_image, header = cutout_matched_header)
        # save fits
        cutout_matched_hdu.writeto(os.path.join("final_images", "final_matched_" + image_path), overwrite = True)
        
        # Step 2.5 : create RBG image
        if extract_filter_name(image_path) == 'jwst.nircam.F115W':
            blue = cutout_image
        elif extract_filter_name(image_path) == 'jwst.nircam.F277W':
            green = cutout_image
        elif extract_filter_name(image_path) == 'jwst.nircam.F444W':
            red = cutout_image
        
        # Step 3 : Print the maps of the spatially resolved SED Analysis
        if j == 0:
            for i, (mp, name_mp) in enumerate(zip(maps, name_maps)):
                    
                    final_map, map_header = read_spat_resol(name_mp, cutout_matched_hdu, hdu_ref, coords_ref, mp)
                    map_fits = fits.PrimaryHDU(data = final_map, header = map_header)
                    maps_list.append(final_map)
                    maps_header = map_header
                    # save fits
                    map_fits.writeto(os.path.join("final_images", "final_map_" + name_mp + '.fits'), overwrite = True)
    
    # For ALMA data
    
    # Step 1 : Load the associated galaxy image
    go_to_parent_and_into("ALMA") 
    fits_files_path = [f for f in os.listdir(os.getcwd()) if f.endswith(end_alma_name)]
    fits_files = [fits.open(f, memmap = True) for f in fits_files_path]
    os.chdir(direc) # Change to working directory
    print(f"now into directory: {os.getcwd()}")
    
    if not fits_files:
        print("[WARN] No FITS files found.")
    for i, fits_image in enumerate(fits_files):
        image_path = fits_files_path[i]
        matched_img_path = os.path.join("matched_images", 'matched_' + image_path)
        hdu = fits_image
        matched_hdu = fits.open(matched_img_path, memmap = True)
        
        # Step 2 : create cutout images to print them and save them
        cutout_image, cutout_header = cutout_fits2(center_coord, hdu, size)
        # Create a new HDU with the modified data and original header
        cutout_hdu = fits.PrimaryHDU(data = cutout_image, header = cutout_header)
        # save fits
        cutout_hdu.writeto(os.path.join("final_images", "final_" + image_path), overwrite = True)
        cutout_list.append(cutout_image)
        
        cutout_matched_image, cutout_matched_header = cutout_fits2(center_coord, matched_hdu, size)
        # Create a new HDU with the modified data and original header
        cutout_matched_hdu = fits.PrimaryHDU(data = cutout_matched_image, header = cutout_matched_header)
        # save fits
        cutout_matched_hdu.writeto(os.path.join("final_images", "final_matched_" + image_path), overwrite = True)
        
        fits_path = os.path.join("matched_images", "matched_" + image_path)
        
        hdu = fits.open(fits_path, memmap=True)
        # Step 2 : Get the coordinates of the galaxy's pixels
        
        center_coord_alma = match_coord3(center_coord, hdu)
        # Derive catalog name from FITS filename
        catalog_name = image_path.replace('.fits', '.cat')
        
        # Step 3: Update default.sex
        update_sex_config(sex_config_file, catalog_name)
        
        # Step 4: Run SExtractor
        try:
            run_sextractor(fits_path, sex_config_file)
        except subprocess.CalledProcessError:
            print(f"[ERROR] SExtractor failed for {fits_image}")
            
        # Step 5: Get the pixel datas of the studied galaxy (the brightest), plot those and save the pixels coordinates
        galaxy_id, center_coord_cat = read_cat(catalog_name, center_coord_alma)
        galaxy_pixels = coord_galaxy(catalog_name, galaxy_id)
        
        galaxy_pixels2 = np.copy(galaxy_pixels)
        galaxy_pixels2[:, 0] = galaxy_pixels[:, 1]
        galaxy_pixels2[:, 1] = galaxy_pixels[:, 0]
        # plot_shape2(hdu, catalog_name, galaxy_id, galaxy_pixels2)
        galaxy_pixels3 = check_gal_coord(cutout_matched_hdu, hdu, galaxy_pixels2)
    print_cutout_imgs(maps_header, Galaxy_name, cutout_list, cutout_names_list)
    
    # === Print RGB Image ===
    # for Q in [5, 7, 9, 11, 13, 15, 17, 19]:
    # for stretch in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    rgb_image1 = make_lupton_rgb(red, green, blue, stretch = stretch1, Q = Q1)
    rgb_image2 = make_lupton_rgb(red, green, blue, stretch = stretch2, Q = Q2)
    print_rgb(maps_header, Galaxy_name, rgb_image1, galaxy_pixels3)
    
    # === Print maps with RGB image ===
    print_maps(maps_list, name_maps, maps_header, Galaxy_name, rgb_image2, galaxy_pixels3)
    print_maps2(maps_list, name_maps, maps_header, Galaxy_name, galaxy_pixels3)
    
    # === Print SFR vs Stellar Mass plot ===
    SFR, SFR_err = maps[0], maps_err[0]
    M, M_err = maps[3], maps_err[3]
    Av, Av_err = maps[4], maps_err[4]
    IRX, IRX_err = maps[-2], maps_err[-2]
    β, β_err = maps[-1], maps_err[-1]
    radius = compute_pixel_distances(coords_ref, M)
    plot_SFR_M(SFR, SFR_err, M, M_err, radius, Galaxy_name)
    plot_Av_M(Av, Av_err, M, M_err, radius, Galaxy_name)
    plot_IRX_β(IRX, IRX_err, β, β_err, radius, Galaxy_name)
    plot_IRX_β2(IRX, IRX_err, β, β_err, radius, Galaxy_name)
        
    
    # For CIGALE
    ALMA_filter_name = create_cigale_test_file_single_object(Galaxy_name, flux_jwst, err_jwst, names_jwst, z_spec, output_csv,
                                                             alma_frequencies_ghz, flux_alma, err_alma, upper_limit = True)
    ALMA_filter_name2 = create_cigale_test_file_multiple_objects(pxls_flux_jwst, pxls_err_flux_jwst, names_jwst, z_spec,
                                                                 output_multi, alma_frequencies_ghz, pxls_flux_alma,
                                                                 pxls_err_flux_alma, upper_limit = True)
    
    # run_cigale(alma_frequencies_ghz, ALMA_filter_name, filter_dir="filters/")
    
    convert_results_to_transposed_csv(input_file='results.txt', output_file='R0600-ID67_2_filtres_alma.csv')
    convert_results_to_transposed_csv_2(input_file='results.txt', output_file='R0600-ID67_2_filtres_alma.csv')
    return ALMA_filter_name


if __name__ == '__main__':
    
    ALMA_filter_name = main()
    # ALMA_filter_name = 
