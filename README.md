# NIR-dark

# JWST Galaxy Analysis Pipeline

This repository contains a set of Python scripts designed to perform **PSF extraction, PSF matching, SED measurement, and plotting** for JWST galaxy data.  
The workflow is intended for studying **one galaxy at a time**, followed by comparative analysis across multiple galaxies.  

---

## Workflow Overview

1. **`get_psf.py`**  
   - Extracts the PSF from JWST images located in the working directory.  
   - The working directory and additional parameters (e.g., the reference file used for PSF matching) must be specified at the beginning of the script.  
   - **This script must be run first.**

2. **`psf_matching.py`**  
   - Performs PSF matching for the galaxies.  
   - Returns cutouts centered on each galaxy with matched PSFs.  
   - Requires configuration of the working directory and relevant file paths at the beginning of the script.

3. **`get_sed.py`**  
   - Computes the flux values (with uncertainties) for the matched galaxies.  
   - The working directory and galaxy-specific parameters must be set at the beginning of the script.

4. **`fits_sed.py`**  
   - Produces diagnostic and scientific plots of the SED for the studied galaxy.  
   - Parameters such as directories and file names must be specified at the beginning of the script.  

5. **`plots_all_galaxies.py`**  
   - Generates comparison plots across all galaxies studied.  
   - Unlike the other scripts, the working directories for each individual galaxy must be specified within the script.  
   - Should be executed only after completing the analysis of all galaxies of interest.

---

## Usage Order

The scripts must be executed in the following order:

1 - get_psf.py
2 - psf_matching.py
3 - get_sed.py
4 - fits_sed.py
(repeat steps 1–4 for each galaxy)
5 - plots_all_galaxies.py
