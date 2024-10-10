import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy import units as u
from astropy.table import Table

# Load Stokes I and Stokes V FITS files
stokes_i_file = input('Enter stokesI image: ')
stokes_v_file = input('Enter stokesV image: ')
Obs = input('Enter observation ID :')
catalog_file = 'bootes_final_cross_match_catalogue-v1.0_classifications.fits'

# Open FITS files for Stokes I and V
with fits.open(stokes_i_file) as hdul_i:
    stokes_i_data = hdul_i[0].data
    stokes_i_header = hdul_i[0].header
    wcs = WCS(stokes_i_header).sub(2)

with fits.open(stokes_v_file) as hdul_v:
    stokes_v_data = hdul_v[0].data
    stokes_v_header = hdul_v[0].header

bmaj = 6/3600
bmin = 6/3600

#Get the ebam area in pixels
beammaj = bmaj / (2.0 * (2 * np.log(2)) ** 0.5)  # Convert to sigma
beammin = bmin / (2.0 * (2 * np.log(2)) ** 0.5)  # Convert to sigma
pixarea = np.abs(stokes_v_header['CDELT1'] * stokes_v_header['CDELT2'])

beamarea = 2 * np.pi * 1.0 * beammaj * beammin  # Note that the volume of a two dimensional gauss
beamarea_pix = beamarea / pixarea

# Load the catalog from the FITS file
with fits.open(catalog_file) as hdul_cat:
    catalog_data = hdul_cat[1].data

# Extract tables from the catalog
ra = catalog_data['RA']
dec = catalog_data['DEC']
stokes_i_flux = catalog_data['Total_flux']

# Convert the catalog RA/DEC to pixel coordinates using W
sky_coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
x_positions, y_positions = wcs.world_to_pixel(sky_coords)

# Print number of sources
print(f"Number of sources in catalog: {len(x_positions)}")

# Define the box size in arcseconds
width_arcsec = int(input('width of box in arcsec: '))
height_arcsec =int(input('height of box in arcsec: '))
width_deg = width_arcsec / 3600
height_deg = height_arcsec / 3600

# Calculate pixel scale (degrees per pixel)
pixel_scale_deg = wcs.proj_plane_pixel_scales()[0]

# Convert the box dimensions to pixels
width_pix = int((width_deg / pixel_scale_deg).decompose().value)
height_pix = int((height_deg / pixel_scale_deg).decompose().value)

stokes_v_abs_list = []

# Iterate over the sources in the catalog
for x, y in zip(x_positions, y_positions):
    # Define the rectangular box aperture around each source
    x_min = int(x - width_pix // 2)
    x_max = int(x + width_pix // 2)
    y_min = int(y - height_pix // 2)
    y_max = int(y + height_pix // 2)
    # Ensure the box is within the image boundaries
    if x_min < 0 or y_min < 0 or x_max >= stokes_i_data.shape[1] or y_max >= stokes_i_data.shape[0]:
        continue
    # Stokes V: find the maximum pixel value within the rectangular region
    stokes_v_abs_pixel = np.max(np.abs(stokes_v_data[y_min:y_max, x_min:x_max]))

    #Get stokes v absolute flux
    stokes_v_abs =  stokes_v_abs_pixel/beamarea_pix

    # Append the value to the list
    stokes_v_abs_list.append(stokes_v_abs)


# Get leakage ratios
leakage_ratios = stokes_v_abs_list/stokes_i_flux

#SNR for stokes I
snr_i = stokes_i_flux/catalog_data['Isl_rms']

# Summarize leakage ratios
print(f"Number of valid leakage ratios: {len(leakage_ratios)}")
print(f"Minimum leakage ratio: {np.min(leakage_ratios)}")
print(f"Maximum leakage ratio: {np.max(leakage_ratios)}")
print(f"Mean leakage ratio: {np.mean(leakage_ratios)}")
print(f"Median leakage ratio: {np.median(leakage_ratios)}")

print(len(stokes_v_abs_list))
print(ra,len(ra))
print(dec,len(dec))
# Create a table with two columns: Leakage Ratio and SNR_I
table = Table([leakage_ratios, snr_i], names=('Leakage_Ratio', 'SNR_I'))

# Create a table with 3 columns: RA, DEC, Stokes_v_flux
table1 = Table([ra, dec, stokes_v_abs_list ], names=('ra', 'dec', 'stokes_v_flux'))

#create table snr table
table2 = Table([ra, dec, snr_i], names=('ra', 'dec', 'snr_i'))

# Save the tables to a CSV file
filename = f'leakage_snr_table({Obs}).csv'
table.write(filename, format='csv', overwrite=True)

filename1 = f'coords_stokesV_flux_table({Obs}).csv'
table1.write(filename1, format='csv', overwrite=True)

filename2 = f'snr_i table.csv'
table2.write(filename2, format='csv', overwrite=True)

print(f"Data saved to '{filename}' and '{filename1}'")
