# -*- coding: utf-8 -*-
"""
Created on Tue May 20 08:52:25 2025

@author: zacha
"""

import os
import subprocess
import numpy as np
import astropy.io
from astropy.table import Table
from astropy import stats as st
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable # Utile pour placer la colorbar

sex_config_file = 'default.sex' # Configuration file for Sextractor

"""
# === INPUT VARIABLES R0600-ID67 ===
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//R0600-ID67//jwst_test//' # Directory in which are all .fits files + all Sextractor and PSFextractor files
# direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//R0600-ID67//jwst_test_2//' # Directory in which are all .fits files + all Sextractor and PSFextractor files
fits_jwst_coord = 'rxcj0600-grizli-v5.0-f444w-clear_drc_sci.fits'   # reference fits relative path for the PSF matching
x0, y0 = 6734, 4057                                                 # center coordinates
end_alma_name = 'image.pbcor.mJyppix.fits'
lambda_jwst = np.array([1.15, 1.50, 2.77, 3.56, 4.44])
"""

# === INPUT VARIABLES A0102-ID224 ===
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//A0102-ID224//JWST' # Directory in which are all .fits files + all Sextractor and PSFextractor files
fits_jwst_coord = "elgordo-grizli-v7.0-f444w-clear_drc_sci.fits" # reference fits relative path for the PSF matching
x0, y0 = 3300, 3600  # center coordinates
end_alma_name = 'image.pbcor.mJyppix.fits'
lambda_jwst = np.array([0.90, 1.15, 1.50, 2.00, 2.77, 3.56, 4.10, 4.44])

"""
# === INPUT VARIABLES M0417-ID46 ===
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//M0417-ID46//JWST' # Directory in which are all .fits files + all Sextractor and PSFextractor files
fits_jwst_coord = "hlsp_canucs_jwst_macs0417-clu-40mas_f444w_v1_sci.fits" # reference fits relative path for the PSF matching
x0, y0 = 6030, 4435                                                    # center coordinates                                                           # region size (lentgh = height) in pixels for the new images
end_alma_name = 'image.pbcor.mJyppix.fits'
lambda_jwst = np.array([0.90, 1.15, 1.50, 2.00, 2.77, 4.44])
"""

# === CONFIGURATION ===
os.chdir(direc) # Change to working directory
# path_jwst_coord = os.path.join("matched_images", 'matched_' + fits_jwst_coord) # reference fits relative path for the PSF matching (well less relative than the precedent)
hdu_jwst_coord = fits.open(fits_jwst_coord, memmap=True) # reference fits for the PSF matching
# catalog_target = fits_jwst_coord.replace('.cat', '.fits')

    
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
    
def get_psf_from_npy(direc_init):
    # Load the PSF .npy files (each containing PSF components)
    psf_files = sorted(os.listdir(f'{direc_init}/'))
    psf_imgs = [np.load(f'{direc_init}/' + f)[0]
                for f in psf_files]  # take first component of each PSF
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

def update_checkimage_name(line, new_background_rms_name):
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
    segmentation_file = filenames[0]

    # Sanitize the base name
    base = new_background_rms_name.replace('.fits', '')

    # Build new filenames
    background_file = f"{base}_background.fits"
    background_rms_file = f"{base}_background_rms.fits"

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
    
def create_backgrounds(image_path):
    # === 1. Load FITS image ===
    # Load the associated science image
    fits_path = os.path.join("matched_images", "matched_" + image_path)
    update_sex_config(sex_config_file, image_path)
    try:
        run_sextractor(fits_path, sex_config_file)
    except subprocess.CalledProcessError:
        print(f"[ERROR] SExtractor failed for {fits_path}")

def convert_in_mJy(hdu, pixels_values): # Convert pixel values into jy_per_pixel, then sum it to obtain flux ===
    header = hdu[0].header
    image_jy_per_pixel = None
    # if "PHOTMJSR" in header:
        # pixels_values = pixels_values * header["PHOTMJSR"]

        # if "PIXAR_SR" in header:
        #     pixel_area_sr = header["PIXAR_SR"] * u.sr
        # else:
        #     pixel_area_sr = proj_plane_pixel_area(header).to(u.sr)

        # image_jy_per_pixel = (flux_density * pixel_area_sr).to(u.Jy)
    if "BUNIT" in header:
        if header["BUNIT"] == "10.0*nanoJansky":
            image_jy_per_pixel = pixels_values * 1e-5 # [mJy/pix]
        elif header["BUNIT"] == "Jy/beam":
            # --- Beam parameters ---
            # Beam major and minor axes [radians]
            bmaj = header['BMAJ'] # [rad]
            bmin = header['BMIN'] # [rad]

            # Beam area [steradian] using Gaussian beam formula
            beam_area = (np.pi * bmaj * bmin) / (4 * np.log(2)) # [sr/beam]

            # --- Pixel scale ---
            # CDELT1 and CDELT2 are usually in degrees/pixel
            cdelt1 = abs(header['CDELT1']) # [deg/pix]
            cdelt2 = abs(header.get('CDELT2', header['CDELT1'])) # [deg/pix]  , fallback if CDELT2 is missing

            # Convert pixel size to radians
            pixel_scale_x = cdelt1*np.pi/180 # [rad/pix]
            pixel_scale_y = cdelt2*np.pi/180 # [rad/pix]

            # Pixel area [steradian]
            pixel_area = pixel_scale_x * pixel_scale_y # [sr/pix²]

            # --- Conversion ---
            # Convert each pixel from Jy/beam to Jy/pixel
            conversion_factor = (pixel_area / beam_area) # [beam/pix²]

            # Convert data: input is in Jy/beam, result will be in Jy
            image_jy_per_pixel = pixels_values * conversion_factor * 1e3 # [mJy/pix]
        elif header["BUNIT"] == "nJy":
            image_jy_per_pixel = pixels_values * 1e-6 # [mJy/pix]
        else: # ALMA or background data already processed
            image_jy_per_pixel = pixels_values
    return image_jy_per_pixel

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

def orig_img(image_path, hduref, coords_ref, ALMA):
    # === 1. Load FITS image ===
    # Load the associated science image
    if image_path.strip().endswith("_background.fits"):
        image_path = image_path.replace('.fits', '.fits')
        origin_fits_path = image_path
        fits_path = image_path
        hdu_origin = fits.open(origin_fits_path, memmap=True)
    elif image_path.strip().endswith("_background_rms.fits"):
        image_path = image_path.replace('.fits', '.fits')
        origin_fits_path = image_path
        fits_path = image_path
        hdu_origin = fits.open(origin_fits_path, memmap=True)
    else :
        fits_path = os.path.join("matched_images", "matched_" + image_path)
        # Because ALMA original data are not in the same directory
        if not ALMA:
            origin_fits_path = image_path
            hdu_origin = fits.open(origin_fits_path, memmap=True)
        else:
            go_to_parent_and_into('ALMA')
            hdu_origin = fits.open(image_path, memmap=True)
            os.chdir(direc) # Change to working directory
    hdu = fits.open(fits_path, memmap=True)
    data = hdu[0].data.astype(np.float32)
    """
    # Because ALMA original data are not in the same directory
    if not ALMA:
        hdu_origin = fits.open(origin_fits_path, memmap=True)
    else:
        go_to_parent_and_into('ALMA')
        hdu_origin = fits.open(origin_fits_path, memmap=True)
        os.chdir(direc) # Change to working directory
    """
    return hdu, hdu_origin, data

def flux_per_pix(image_path, hduref, coords_ref, ALMA):
    # === 1. Load FITS image ===
    hdu, hdu_origin, data = orig_img(image_path, hduref, coords_ref, ALMA)
    
    # === 2. Convert galaxy coordinates to the studied filter image ===
    coords_valid = check_gal_coord(hdu, hduref, coords_ref)
    
    """
    # Plot image of the galaxy highlighted
    image_data = data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_data, origin='lower', cmap='gray', vmin=np.percentile(
        image_data, 5), vmax=np.percentile(image_data, 99))
    ax.scatter(
        coords_valid[:, 0], coords_valid[:, 1], s=1, color='red')  # x, y
    ax.set_title(f"Filtre {image_path}")
    fig.canvas.manager.set_window_title(f"Image traitée de {image_path} (JWST)")
    plt.show()
    """
    # === 3. Extract pixel values from image with higher resolution at the corresponding positions ===
    # pixels_values = data
    pixels_values = data[coords_valid[:, 1], coords_valid[:, 0]]
    print("Les pixels sont au nombre de {}".format(np.shape(pixels_values)))
    # === 4. Convert pixel values into jy_per_pixel, then sum it to obtain flux ===
    image_jy_per_pixel = convert_in_mJy(hdu_origin, pixels_values)
    return image_path, image_jy_per_pixel

def plot_sed(flux_sed, err_values):
    flux = flux_sed*1e6
    error = err_values*1e6

    fig, ax = plt.subplots()
    ax.errorbar(lambda_jwst, flux, yerr = error, fmt = 'o', capsize = 3)
    ax.loglog(lambda_jwst, flux, 'r')
    ax.set_ylabel('Flux (nJy)', size = 20)
    ax.set_xlabel('λ(µm)', size = 20)
    ax.tick_params(axis='x', labelsize = 20)
    ax.tick_params(axis='y', labelsize = 20)
    fig.canvas.manager.set_window_title("overview SED (JWST)")
    #plt.subplots_adjust(left=0.07, bottom=0.2, right=0.96, top=0.8, wspace=0.4, hspace=0.4) # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    plt.show()
    
def proc_sed(flux, err_flux):
    sed = np.array(flux, dtype = object)
    err_sed = np.array(err_flux, dtype = object)
    flux_values = np.array([row[1] for row in sed])
    err_values = np.array([abs(row[1]) for row in err_sed])
    S_N = flux_values/err_values
    print(f"Signal to Noise ratio S/N is: {S_N}")
    return flux_values, err_values

def RAM_error(image_path, coords_ref, hdu_bkgd, hdu, hdu_ref, n_random, seed):
    """
    Perform photometry on custom pixel masks (list of lists of (y, x) indices),
    and on randomly shifted masks for background estimation, avoiding source contamination.

    Parameters
    ----------
    hdu : HDUList
        Contains science image 2D ndarray (e.g., background-subtracted flux map), header corresponding to Data
    hdu_ref : HDUList
        HDU of th reference image, from which come from the galaxy coordinates
    background_error : 2D ndarray
        2D array of per-pixel background RMS values.
    masks : list of list of tuple
        Each entry is a list of (y, x) pixel indices belonging to one object's aperture. Not used here
    n_random : int
        Minimum number of valid random masks to generate per object.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    flux : float
        Measured flux for the galaxy. Not used here
    error : float
        Corresponding uncertainties (from background error map). Not used here
    rand_ap : array of all random apertures
        Array of background fluxes from valid random positions (length galaxy_coord each).
    """
    # Load data from fits files
    data = hdu.data.astype(np.float32)
    # if type(hdu_bkgd) == astropy.io.fits.hdu.hdulist.HDUList:
    #     hdu_bkgd_data = hdu_bkgd[0].data.astype(np.float32)
    # else:
    #     hdu_bkgd_data = hdu_bkgd.data.astype(np.float32)   # type 'hduist'
        
    sigma = 4
    rng = np.random.default_rng(seed)
    ny, nx = data.shape
    galaxy_coord = check_gal_coord(hdu, hdu_ref, coords_ref)
    N = np.shape(galaxy_coord)[0]
    """
    # Make a clean copy of the image and mask out all source pixels without sigma_clip
    data_masked = data.copy() 
    for mask in masks:
        xs, ys = zip(*mask)
        data_masked[xs, ys] = np.nan
    """
    # We create a mask hiding objects with luminosity > 4 * sigma, and copy this mask for the background
    data_masked = st.sigma_clip(data, sigma = sigma, axis = (0, 1), masked = False)
    # mask = data_masked.mask  # tableau booléen
    # hdu_bkgd_data[mask] = np.nan
    
    # Plot image of the galaxy highlighted
    image_data = data_masked
    wcs = WCS(hdu.header)
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw = {'projection': wcs})
    
    ax.coords['ra'].set_axislabel('')
    ax.coords['dec'].set_axislabel('')
    
    # Étiquettes RA/Dec
    ax.coords['ra'].set_axislabel('RA (J2000)', fontsize = 12)
    ax.coords['dec'].set_axislabel('Dec (J2000)', fontsize = 12)
    
    ax.imshow(image_data, origin='lower', cmap  =  'grey')
    ax.scatter(galaxy_coord[:, 0], galaxy_coord[:, 1], s=1, color='red')  # x, y
    ax.set_title(f"Random Aperture Method of {image_path} with sigma = {sigma}")
    fig.canvas.manager.set_window_title(f"{image_path}, sigma = {sigma}")
    plt.show()
    
    rand_ap = np.ndarray((n_random, N), dtype = np.float32)
    xs, ys = galaxy_coord[:, 0], galaxy_coord[:, 1]
    ys = np.array(ys)
    xs = np.array(xs)
    # Random background estimates
    attempts = 0
    i_rand = 0
    max_attempts = n_random * 50  # prevent infinite loop

    while i_rand < n_random and attempts < max_attempts:
        y_offset = rng.integers(-ny // 3, ny // 3)
        x_offset = rng.integers(-nx // 3, nx // 3)
        ys_shifted = ys + y_offset
        xs_shifted = xs + x_offset

        # Skip if out of bounds
        if (np.any(ys_shifted < 0) or np.any(ys_shifted >= ny) or
            np.any(xs_shifted < 0) or np.any(xs_shifted >= nx)):
            attempts +=1
            continue
        
        shifted_values = data_masked[ys_shifted, xs_shifted]
        # Only accept if no NaNs in the shifted mask
        if np.all(np.isfinite(shifted_values)):
            rand_ap[i_rand, :] = shifted_values
            i_rand += 1
        attempts += 1
        
    if attempts == max_attempts:
        print(f"The random apertures could not achieve a {n_random} goal of samples number for {image_path}")
    else:
        print(f"The random apertures succeded for {image_path} with {n_random} samples and seed: {seed}")
    return rand_ap

def get_sub_bckg_img(hdu, hdu_bkgd):
    if type(hdu) == astropy.io.fits.hdu.hdulist.HDUList:
        hdu_header = hdu[0].header
        hdu_data = hdu[0].data.astype(np.float32)
    else:
        hdu_header = hdu.header   # type 'hduist'
        hdu_data = hdu.data.astype(np.float32)
    
    if type(hdu_bkgd) == astropy.io.fits.hdu.hdulist.HDUList:
        hdu_bkgd_data = hdu_bkgd[0].data.astype(np.float32)
    else:
        hdu_bkgd_data = hdu_bkgd.data.astype(np.float32)   # type 'hduist'
    sub_bckg_img = hdu_data - hdu_bkgd_data
    return sub_bckg_img, hdu_bkgd_data, hdu_header

"""
def get_alma_noise():
    # Step 1 : Load the associated galaxy image noise
    go_to_parent_and_into("ALMA") 
    fits_files_path = [f for f in os.listdir(os.getcwd()) if f.endswith(end_alma_noise_name)]
    fits_files = [fits.open(f, memmap=True) for f in fits_files_path]
    os.chdir(direc) # Change to working directory
    print(f"now into directory: {os.getcwd()}")
    return fits_files[0]
"""

def final_flux(pxls_flux, pxls_err_flux, flux, err_flux, files_names, coords_ref, first_image_path, hdu_ref, ALMA):
    # Run Sexctractor for the new images and create background and background_rms files
    create_backgrounds(first_image_path)
    first_image_path_background = first_image_path.replace('.fits', '_background.fits')
    backgrn_path, bckgrd_flux_mjy = flux_per_pix(first_image_path_background, hdu_ref, coords_ref, ALMA)
    
    # Get the galaxy flux
    image_path, image_mjy_per_pixel = flux_per_pix(first_image_path, hdu_ref, coords_ref, ALMA)
    img_minus_bckgrd = image_mjy_per_pixel - bckgrd_flux_mjy
    flux_mjy = np.sum(img_minus_bckgrd)
    flux.append([image_path, flux_mjy])
    files_names.append(image_path)
    pxls_flux.append(img_minus_bckgrd)
    
    # Get the rms by tha Random Aperture Method
    hdu, hdu_origin, data = orig_img(first_image_path, hdu_ref, coords_ref, ALMA)
    hdu_bkgd, hdu_origin, data = orig_img(first_image_path_background, hdu_ref, coords_ref, ALMA)
    img_minus_bckgrd2, bckgrd, hdu_header = get_sub_bckg_img(hdu, hdu_bkgd)
    hdu_img_minus_bckgrd2 = fits.PrimaryHDU(data = img_minus_bckgrd2, header = hdu_header)
    rand_ap = RAM_error(first_image_path, coords_ref, hdu_bkgd, hdu_img_minus_bckgrd2, hdu_ref, n_random = 1000, seed = 1002)
    err_flux_integ = np.sum(rand_ap, axis = 1)
    err_flux_mjy = convert_in_mJy(hdu_origin, err_flux_integ)
    err_flux_quad = np.std(err_flux_mjy)
    # for pixel by pixel analysis
    sparesol_err_flux = np.std(rand_ap, axis = 0)
    sparesol_err_flux = convert_in_mJy(hdu_origin, sparesol_err_flux)
    """
    # my own old method that's different
    err_flux_nounit = np.mean(rand_ap, axis = 0)
    err_flux_mjy2 = err_flux_mjy * err_flux_mjy
    err_flux_quad = np.sqrt(np.sum(err_flux_mjy2))
    """
    err_flux.append([image_path, err_flux_quad])
    pxls_err_flux.append(sparesol_err_flux)
    """
    # Get the flux error with RMS from SExtractor
    first_image_path_background_rms = first_image_path.replace('.fits', '_background_rms.fits')
    backgrn_rms_path, err_flux_mjy = flux_per_pix(first_image_path_background_rms, hdu_ref, coords_ref, ALMA)
    err_flux_mjy2 = err_flux_mjy * err_flux_mjy
    err_flux_quad = np.sqrt(np.sum(err_flux_mjy2))
    err_flux.append([image_path, err_flux_quad])
    pxls_err_flux.append(err_flux_mjy)
    """
    
    # for checking images on carta
    S_N = img_minus_bckgrd / sparesol_err_flux
    header1 = hdu[0].header
    S_N2 = fits.PrimaryHDU(data = S_N, header = header1)
    img_minus_bckgrd2 = fits.PrimaryHDU(data = img_minus_bckgrd, header = header1)
    return pxls_flux, pxls_err_flux, flux, err_flux, files_names, S_N2, img_minus_bckgrd2


def main():
    """Main function to process all FITS files in the directory."""
    
    # === CONFIGURATION ===
    center_coord = (x0, y0)
    npy_direc = "psf_cubes"
    psf_files, psf_imgs = get_psf_from_npy(npy_direc)

    # Reference galaxy pixel coordinates in reference image (as (x, y) = (col, row))
    filename = fits_jwst_coord.replace('.fits', '.cat') + "_galaxy_ref_coords.txt"
    """
    Read saved (x, y) coordinates from a file.
    Returns a list of (x, y) tuples.
    """
    # Step 1 : get back the coordinates of the galaxy for JWST
    coords = []
    with open(filename, "r") as f:
        for line in f:
            x_str, y_str = line.strip().split()
            coords.append([int(x_str), int(y_str)])
            coords_ref = np.array(np.copy(coords))
            
    # Step 2 : Get the appropriate flux values
    flux_jwst = []
    err_flux_jwst = []
    pxls_flux_jwst = []
    pxls_err_flux_jwst = []
    files_names_jwst = []
    # Loop over all PSF images and apply matching for JWST images
    for i, psf_src in enumerate(psf_imgs):
        first_image_path = psf_files[i].replace('_psf.npy', '.fits')
        """
        if first_image_path == "rxcj0600-grizli-v5.0-f356w-clear_drc_sci.fits":
            pxls_flux_jwst, pxls_err_flux_jwst, flux_jwst, err_flux_jwst, files_names_jwst, S_N, subs_img = final_flux(pxls_flux_jwst, pxls_err_flux_jwst, flux_jwst, err_flux_jwst, files_names_jwst, coords_ref, first_image_path, hdu_jwst_coord, ALMA = False)
        """
        pxls_flux_jwst, pxls_err_flux_jwst, flux_jwst, err_flux_jwst, files_names_jwst, S_N, subs_img = final_flux(
        pxls_flux_jwst, pxls_err_flux_jwst, flux_jwst, err_flux_jwst, files_names_jwst, coords_ref, first_image_path,
        hdu_jwst_coord, ALMA = False)
        """
        # Step 3 : Check the S_N, bck sub image  (must change last line of flux_per_pix function)
        save_fits(S_N, "S_N_" + first_image_path)
        save_fits(subs_img, "subs_img_" + first_image_path)
        """
    flux_values_jwst, err_values_jwst = proc_sed(flux_jwst, err_flux_jwst)
    
    # Step 3 : Check the SED
    plot_sed(flux_values_jwst, err_values_jwst)
    
    # For ALMA data
    
    # Step 1 : Load the associated galaxy image
    go_to_parent_and_into("ALMA") 
    fits_files_path = [f for f in os.listdir(os.getcwd()) if f.endswith(end_alma_name)]
    fits_files = [fits.open(f, memmap=True) for f in fits_files_path]
    os.chdir(direc) # Change to working directory
    print(f"now into directory: {os.getcwd()}")
    
    flux_alma = []
    err_flux_alma = []
    pxls_flux_alma = []
    pxls_err_flux_alma = []
    files_names_alma = []
    if not fits_files:
        print("[WARN] No FITS files found.")
    for i, fits_image in enumerate(fits_files):
        image_path = fits_files_path[i]
        """
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
        plot_shape2(hdu, catalog_name, galaxy_id, galaxy_pixels2)
        """
        # Step 6: Get the appropriate flux values
        pxls_flux_alma, pxls_err_flux_alma, flux_alma, err_flux_alma, files_names_alma, S_N, subs_img  = final_flux(
            pxls_flux_alma, pxls_err_flux_alma, flux_alma, err_flux_alma, files_names_alma, coords_ref, image_path, 
            hdu_jwst_coord, ALMA = True)
    flux_values_alma, err_values_alma = proc_sed(flux_alma, err_flux_alma)
    
    return pxls_flux_jwst, pxls_err_flux_jwst, flux_values_jwst, err_values_jwst, files_names_jwst, pxls_flux_alma, pxls_err_flux_alma, flux_values_alma, err_values_alma, files_names_alma


if __name__ == '__main__':
    pxls_flux_jwst, pxls_err_flux_jwst, flux_jwst, err_jwst, names_jwst, pxls_flux_alma, pxls_err_flux_alma, flux_alma, err_alma, names_alma = main()
