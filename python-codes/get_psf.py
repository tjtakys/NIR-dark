# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:31:42 2025

@author: zacha
"""


import os
import re
import subprocess
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable # Utile pour placer la colorbar


# === INPUT VARIABLES R0600-ID67 ===
# direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//R0600-ID67//jwst_test//' # Directory in which are all .fits files + all Sextractor and PSFextractor files
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//R0600-ID67//jwst_test_2//' # Directory in which are all .fits files + all Sextractor and PSFextractor files
# fits_image = 'rxcj0600-grizli-v5.0-f444w-clear_drc_sci.fits'
catalog_target = "rxcj0600-grizli-v5.0-f444w-clear_drc_sci.cat"
center_coord = (6734, 4057)
file_issues = None

"""
# === INPUT VARIABLES A0102-ID224 ===
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//A0102-ID224//JWST' # Directory in which are all .fits files + all Sextractor and PSFextractor files
# fits_image = 'rxcj0600-grizli-v5.0-f444w-clear_drc_sci.fits'
catalog_target = "elgordo-grizli-v7.0-f444w-clear_drc_sci.cat"
center_coord = (3300, 3600)
file_issues = ["elgordo-grizli-v7.0-f200w-clear_drc_sci.cat", "elgordo-grizli-v7.0-f410m-clear_drc_sci.cat"] # If one image has artefacts or other that needs a modification of the sex or psfex files
"""
"""
# === INPUT VARIABLES M0417-ID46 ===
direc = '//mnt//c//zachman//Cours//Cours_ENS//M1.1//stage//Données//M0417-ID46//JWST' # Directory in which are all .fits files + all Sextractor and PSFextractor files
catalog_target = "hlsp_canucs_jwst_macs0417-clu-40mas_f444w_v1_sci.cat"
center_coord = (6030, 4435)
file_issues = ["hlsp_canucs_jwst_macs0417-clu-40mas_f090w_v1_sci.cat", "hlsp_canucs_jwst_macs0417-clu-40mas_f115w_v1_sci.cat", "hlsp_canucs_jwst_macs0417-clu-40mas_f356w_v1_sci.cat", "hlsp_canucs_jwst_macs0417-clu-40mas_f410m_v1_sci.cat"]
"""
# === CONFIGURATION ===
sex_config_file = 'default.sex' # Configuration file for Sextractor
psfex_config_file = 'default.psfex' # Configuration file for PSFextractor
os.chdir(direc) # Change to working directory

# === STEP 0: Update the default.sex configuration file ===

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

def update_sex_config(sex_file_path, new_catalog_name):
    """Update the CATALOG_NAME field in the SExtractor configuration file."""
    lines = []
    with open(sex_file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('CATALOG_NAME'):
                lines.append(f'CATALOG_NAME     {new_catalog_name}\n')
            elif line.strip().startswith("CHECKIMAGE_NAME"):
                new_line = update_checkimage_name(line, new_catalog_name)
                lines.append(new_line)
            else:
                lines.append(line)
    with open(sex_file_path, 'w') as f:
        f.writelines(lines)
    print(f'[INFO] Updated {sex_file_path} with CATALOG_NAME = {new_catalog_name} and segmentation file')
    
def update_psfex_config(psfex_file_path, S_N): # change_scale
    """Update the PSF_SAMPLING field in the PSFEx configuration file."""
    lines = []
    with open(psfex_file_path, 'r') as f:
        for line in f:
            """
            if line.strip().startswith('PSF_SAMPLING'):
                if change_scale:
                    lines.append('PSF_SAMPLING    0.0             # Sampling step in pixel units (0.0 = auto)\n')
                else:
                    lines.append('PSF_SAMPLING    0.5             # Sampling step in pixel units (0.0 = auto)\n')
            """
            if line.strip().startswith('SAMPLE_MINSN'):
                if S_N != None:
                    lines.append(f'SAMPLE_MINSN       {S_N}           # Minimum S/N for a source to be used\n')
                else:
                    lines.append(f'SAMPLE_MINSN       {200}           # Minimum S/N for a source to be used\n')
            else:
                lines.append(line)
    with open(psfex_file_path, 'w') as f:
        f.writelines(lines)
    print(f'[INFO] Updated {psfex_file_path}')

def run_sextractor(fits_image, sex_config_file):
    """Run SExtractor to generate the object catalog from a FITS image."""
    sex_cmd = f'sex {fits_image} -c {sex_config_file}'
    print(f"[INFO] Running: {sex_cmd}")
    subprocess.run(["wsl.exe", "sex", fits_image, "-c", sex_config_file], check=True)

def run_psfex(catalog_name, psfex_config_file):
    """Run PSFEx on the generated catalog to create the PSF model."""
    psfex_cmd = f'psfex {catalog_name} -c {psfex_config_file}'
    print(f"[INFO] Running: {psfex_cmd}")
    subprocess.run(["wsl.exe", "psfex", catalog_name, "-c", psfex_config_file], check=True)
    
def read_psf_file(psf_filename):
    """Read and reshape the PSF cube from a PSFEx .psf FITS file."""
    print(f"[INFO] Reading PSF file: {psf_filename}")
    hdu_list = fits.open(psf_filename, memmap=True)

    # Find the HDU containing the PSF dimensions
    psf_hdu_index = None
    for i, hdu in enumerate(hdu_list):
        if 'PSFAXIS1' in hdu.header:
            psf_hdu_index = i
            break

    if psf_hdu_index is None:
        raise RuntimeError("No HDU with PSFAXIS1 keyword found in PSF file.")
    
    psf_hdu = hdu_list[psf_hdu_index]
    print(psf_hdu)  # Affiche les infos de la table PSF

    psf_data = psf_hdu.data['PSF_MASK'][0]
    hdr = psf_hdu.header

    # Reconstruct the 3D PSF cube from header dimensions
    psf_shape = (hdr['PSFAXIS3'], hdr['PSFAXIS2'], hdr['PSFAXIS1'])
    psf_cube = np.reshape(psf_data, psf_shape)

    print(f"[INFO] PSF cube reconstructed with shape {psf_cube.shape}")
    return psf_cube

def plt_psf(psf_cube, fits_image):
    """Read and plot the PSF cube."""
    psf_shape = np.shape(psf_cube)
    col = int(psf_shape[0]/2)
    vmax = 0
    vmin = 0
    for j in range (int(2*col)):
        if vmax < np.max(psf_cube[j]):
            vmax = np.max(psf_cube[j])
        if vmin > np.min(psf_cube[j]):
            vmin = np.min(psf_cube[j])
    # Pour conserver la colorbar centrée sur 0 :
    if abs(vmin) < abs(vmax) and vmin !=0 :
        vmin = abs(vmax)*(vmin/abs(vmin)) # Conserver le signe de vmin
    else: # On suppose que vmax ne peut être nul
        if vmax != 0:
            vmax = abs(vmin)*(vmax/abs(vmax)) # Conserver le signe de vmax
    # cmap = 'seismic'

    fig, axs = plt.subplots(2, col, figsize=(10, 6))
    for i in range(int(2*col)):
        ax = axs[i // col, i % col]
        ax.imshow(psf_cube[i], origin='lower', cmap='viridis')
        ax.set_title(f"PSF #{i}")
        if i % col == col-1:
            im = ax.imshow(psf_cube[i], vmin = vmin, vmax = vmax)
            ax_divider = make_axes_locatable(ax)
            # Add an Axes to the right of the main Axes.
            cax = ax_divider.append_axes("right", size="4%", pad="2%")
            cbar = fig.colorbar(im, cax = cax)
            cbar.ax.tick_params(labelsize = 15)
    fig.canvas.manager.set_window_title("PSF de " + fits_image)
    #plt.subplots_adjust(left=0.07, bottom=0.2, right=0.96, top=0.8, wspace=0.4, hspace=0.4) # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    plt.show()

def edge_values(img, tol, return_values):
    """
    Get the list of pixels at a given radial distance from the image center.
    
    Parameters
    ----------
    img : 2D numpy array
        Input image.
    rad : float
        Radius (in pixels) from the center.
    tol : float
        Tolerance around the radius (default is 0.5 pixel).
    return_values : bool
        If True, return pixel values instead of coordinates.

    Returns
    -------
    List of (i, j) pixel coordinates OR values at distance `rad` ± `tol`.
    """
    ny, nx = img.shape
    cx, cy = nx // 2, ny // 2  # image center
    rad = cx
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    mask = (r >= rad - tol) & (r <= rad + tol)
    
    if return_values:
        return img[mask].tolist()
    else:
        return list(zip(*np.where(mask)))
    
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

def plt_psf_prez(psf_cube, fits_names, header):
    """Read and plot the PSF cube."""
    psf_cube = np.array(psf_cube)
    psf_shape = np.shape(psf_cube)
    col = psf_shape[0]
    vmax = 0
    vmin = 0
    edge_ratio = []
    os.makedirs("psf_edges", exist_ok=True)
    for j in range (col):
        if vmax < np.max(psf_cube[j]):
            vmax = np.max(psf_cube[j])
        if vmin > np.min(psf_cube[j]):
            vmin = np.min(psf_cube[j])
    # Pour conserver la colorbar centrée sur 0 :
        """
    if abs(vmin) < abs(vmax) and vmin !=0 :
        vmin = abs(vmax)*(vmin/abs(vmin)) # Conserver le signe de vmin
    else: # On suppose que vmax ne peut être nul
        if vmax != 0:
            vmax = abs(vmin)*(vmax/abs(vmax)) # Conserver le signe de vmax
    # cmap = 'seismic'
        """
    col2 = col//2 + col%2
    if col == 5:
        fig, axs = plt.subplots(1, col, figsize = (15, 10))
    else:
        fig, axs = plt.subplots(2, col2, figsize = (15, 10))
    axs = axs.flatten()
    for i in range(col):
        ax = axs[i]
        edge_img = psf_cube[i] / np.max(psf_cube[i])
        ax.imshow(psf_cube[i], origin='lower', vmin = vmin, vmax = vmax)
        edge_value = edge_values(edge_img, tol = 0.5, return_values = True)
        # edge_value /= np.max(psf_cube[i])
        edge_ratio.append(edge_value)
        filter_name = extract_filter_name(fits_names[i])
        ax.set_title(f"PSF {filter_name}")
        if i == col:
            im = ax.imshow(psf_cube[i], vmin = vmin, vmax = vmax)
            ax_divider = make_axes_locatable(ax)
            # Add an Axes to the right of the main Axes.
            cax = ax_divider.append_axes("right", size="4%", pad="2%")
            cbar = fig.colorbar(im, cax = cax)
            cbar.ax.tick_params(labelsize = 15)
        hdu_final = fits.PrimaryHDU(data = edge_img, header = header)
        hdu_final.writeto(os.path.join("psf_edges", "edges_values_" + filter_name + '.fits'), overwrite=True)
    fig.canvas.manager.set_window_title("PSF de " + filter_name)
    plt.subplots_adjust(left=0.07, bottom=0.2, right=0.96, top=0.8, wspace=0.4, hspace=0.4) # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.2) 
                                                                                                 # et plt.subplot_tool() marche aussi
    os.makedirs("final_images", exist_ok = True)
    fig.savefig(os.path.join("final_images", 'PSF_JWST.png'), dpi = 300, bbox_inches = 'tight')
    plt.show()
    return edge_ratio
    
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

from scipy.ndimage import binary_fill_holes

def fill_missing_pixels(pixels, shape=None):
    """
    Fills missing interior pixels in a shape represented by a list of (y, x) coordinates.

    Parameters
    ----------
    pixels : list of tuple
        List of (y, x) pixel coordinates representing the shape, possibly with holes.
    shape : tuple, optional
        Shape of the image as (height, width). If None, it is inferred automatically.

    Returns
    -------
    list of tuple
        List of (y, x) pixel coordinates with missing interior pixels filled.
    """
    pixels = np.array(pixels)
    if shape is None:
        max_y = pixels[:, 0].max() + 1
        max_x = pixels[:, 1].max() + 1
        shape = (max_y, max_x)

    # Create a binary mask of the shape
    mask = np.zeros(shape, dtype=bool)
    mask[pixels[:, 0], pixels[:, 1]] = True

    # Fill interior holes
    filled = binary_fill_holes(mask)

    # Return filled pixels as a list of (y, x)
    y_filled, x_filled = np.where(filled)
    return list(zip(y_filled, x_filled))

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

def plot_shape(catalog_name, object_id): #  plot_shape(1338, (6737, 4059))
    filename = catalog_name + "_galaxy_ref_coords.txt"
    """
    Read saved (x, y) coordinates from a file.
    Returns a list of (x, y) tuples.
    """
    coords = []
    with open(filename, "r") as f:
        for line in f:
            x_str, y_str = line.strip().split()
            coords.append([int(x_str), int(y_str)])
            coords_2 = np.array(np.copy(coords))
    # Load original image for display
    image_data = fits.open(catalog_name.replace('.cat', '.fits'), memmap=True)
    image_data = image_data[0].data
    # Plot image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_data, origin='lower', cmap='gray', vmin=np.percentile(image_data, 5), vmax=np.percentile(image_data, 99))
    ax.scatter(coords_2[:, 0], coords_2[:, 1], s=1, color='red')  # x, y
    ax.set_title(f"Object ID {object_id}")
    plt.show()
    """
    if type(pxls_coord) == None:
        
    else:
        coords_2 = pxls_coord
        image_data = hdu[0].data
        # Load original image for display
        image_data = fits.open(catalog_name.replace('.cat', '.fits'), memmap=True)
        image_data = image_data[0].data
        # Plot image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_data, origin='lower', cmap='gray', vmin=np.percentile(image_data, 5), vmax=np.percentile(image_data, 99))
        ax.scatter(coords_2[:, 0], coords_2[:, 1], s=1, color='red')  # x, y
        ax.set_title(f"Object ID {object_id}")
        plt.savefig(f"object_{object_id}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)  # Optional but recommended
        """
    
def match_pixels_coord(wcs_ref, hdu):
    
    filename = catalog_target + "_galaxy_ref_coords.txt"
    """
    Read saved (x, y) coordinates from a file.
    Returns a list of (x, y) tuples.
    """
    coords = []
    with open(filename, "r") as f:
        for line in f:
            x_str, y_str = line.strip().split()
            coords.append([int(x_str), int(y_str)])
            coords_2 = np.array(np.copy(coords))
    
    wcs = WCS(hdu[0].header)
    
    # === 2. Convert pixel coordinates in ref to world coordinates (RA, Dec) ===
    world_coords = wcs_ref.wcs_pix2world(coords_2, 1)

    # === 3. Convert world coordinates to pixel coordinates in image with higher resolution ===
    coords = wcs.wcs_world2pix(world_coords, 1)

    # === 4. Round to integer pixel values and filter valid pixels inside image with higher resolution ===
    coords_int = np.round(coords).astype(int)
    
    return coords_int

def main():
    L=[]
    """Main function to process all FITS files in the directory."""
    
    fits_files = [f for f in os.listdir(direc) if f.endswith('sci.fits') if not f.startswith(('match', 'background', 'samp', 'chi', 'proto', 'resi')) ]
    fits_target = catalog_target.replace('.cat', '.fits')
    hdu_target = fits.open(fits_target, memmap=True)
    fits_names = []
    if not fits_files:
        print("[WARN] No FITS files found.")
    for fits_image in fits_files:
        # if fits_image.replace('.fits', '.fits') == "rxcj0600-grizli-v5.0-f444w-clear_drc_sci.fits" or fits_image.replace('.fits', '.fits') == "rxcj0600-grizli-v5.0-f356w-clear_drc_sci.fits":
        # if fits_image.replace('.fits', '.fits') == "elgordo-grizli-v7.0-f444w-clear_drc_sci.fits" or fits_image.replace('.fits', '.fits') == "elgordo-grizli-v7.0-f356w-clear_drc_sci.fits":
        fits_names.append(fits_image.replace('.fits', '.fits'))
        print(f"\n[INFO] Processing: {fits_image}")
        # Derive catalog name from FITS filename
        catalog_name = fits_image.replace('.fits', '.cat') 
        
        
        # Step 0: Update default.sex
        update_sex_config(sex_config_file, catalog_name)
        
        # Step 1: Run SExtractor
        try:
            run_sextractor(fits_image, sex_config_file)
        except subprocess.CalledProcessError:
            print(f"[ERROR] SExtractor failed for {fits_image}")
            continue
        
        # Step 2: Run PSFEx
        try:
            run_psfex(catalog_name, psfex_config_file)
        except subprocess.CalledProcessError:
            print(f"[ERROR] PSFEx failed for {catalog_name}")
            continue
        """
        # Step 2.5: Change PSFex file
        change_scale = True
        #if catalog_name == "rxcj0600-grizli-v5.0-f115w-clear_drc_sci.cat" or catalog_name == "rxcj0600-grizli-v5.0-f150w-clear_drc_sci.cat":
        if catalog_name == "rxcj0600-grizli-v5.0-f444w-clear_drc_sci.cat":
            change_scale = False
        update_psfex_config(psfex_config_file, change_scale)
        """
        # Step 2.5: Change PSFex file
        S_N = None
        
        if file_issues != None:
            if catalog_name == "hlsp_canucs_jwst_macs0417-clu-40mas_f410m_v1_sci.cat":
                S_N = 2000
            elif catalog_name == "hlsp_canucs_jwst_macs0417-clu-40mas_f115w_v1_sci.cat":
                S_N = 500
            elif catalog_name == "hlsp_canucs_jwst_macs0417-clu-40mas_f090w_v1_sci.cat":
                S_N = 1000
            elif catalog_name == "hlsp_canucs_jwst_macs0417-clu-40mas_f356w_v1_sci.cat":
                S_N = 2000
                    
        
        update_psfex_config(psfex_config_file, S_N)
        
        # Step 3: Read PSF file
        psf_filename = catalog_name.replace('.cat', '.psf')
        try:
            psf_cube = read_psf_file(psf_filename)
        except Exception as e:
            print(f"[ERROR] Failed to read PSF file {psf_filename}: {e}")
            continue
        L.append(psf_cube[0])
        # plt_psf(psf_cube, fits_image)

        # Step 4: Save images as python (.npy) files, for futur process (test_matching.py)
        output_dir = os.path.join(direc, "psf_cubes")
        os.makedirs(output_dir, exist_ok=True)

        # Name of the file based on the FITS file's name
        psf_base = os.path.splitext(fits_image)[0]  # delete end of fits file .fits
        psf_output_path = os.path.join(output_dir, f"{psf_base}_psf.npy")

        # save of psf components
        np.save(psf_output_path, psf_cube)
        print(f"[INFO] PSF cube saved to: {psf_output_path}")
        # psf_cube = np.load("psf_cubes/rxcj0600-grizli-v5.0-f444w-clear_drc_sci_psf.npy")
        
        # Step 5: Get the pixel datas of the studied galaxy for the reference image (the brightest), plot those and save the pixels coordinates
        if catalog_name == catalog_target:
            galaxy_id, center_coord_cat = read_cat(catalog_name, center_coord)
            galaxy_pixels = coord_galaxy(catalog_name, galaxy_id)
            plot_shape(catalog_target, 'galaxy_id')
        
        

                
    edge_ratios = plt_psf_prez(L, fits_names, hdu_target[0].header)
    for l, rat in enumerate(edge_ratios):
        print ("La dimension du tableau est " + "{}".format(np.shape(rat)))
        print ("Les valeurs maximale et minimale des objets dans le tableau sont " + "{} et {}, et la moyenne est {}".format(np.max(rat), np.min(rat), np.mean(rat)))
        print()
        os.makedirs("edges_values", exist_ok=True)
        np.savetxt(os.path.join("edges_values", fits_names[l] + '.txt'), rat, fmt="%.3e")  # ou fmt="%.3e" pour notation scientifique
    
        
       
if __name__ == '__main__':
    main()
