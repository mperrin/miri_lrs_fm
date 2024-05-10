import pysiaf

# Read in constants from SIAF that we will use below
# Determine center of LRS slit, in pixels in the science frame

miri_siaf = pysiaf.Siaf('MIRI')
ap_slit = miri_siaf.apertures['MIRIM_SLIT']  # SLIT aperture type, does not have coordinates in det frame defined
ap_full = miri_siaf.apertures['MIRIM_FULL']

# Toggle parameter - do we rely on WCS or on Pysiaf for inferring slit center and so on?
USE_WCS_FOR_SLIT_COORDS =  False   # I would expect WCS to be more accurate, due to including filter shifts, but empirically this is not the case.

# retrieve/derive some values from the SIAF:
# Slit center, in pixels. **TODO** Update for slight filter dependence?
slit_center = ap_full.tel_to_sci(ap_slit.V2Ref, ap_slit.V3Ref )

# corners of the slit, as seen on the detector
slit_closed_poly_points = ap_full.tel_to_sci(*ap_slit.closed_polygon_points('tel', rederive=False))

# Slit dimensions, in arcsec
slit_width = (ap_slit.XIdlVert4+ap_slit.XIdlVert3)/2 - (ap_slit.XIdlVert1+ap_slit.XIdlVert2)/2
slit_height = (ap_slit.YIdlVert2+ap_slit.YIdlVert3)/2 - (ap_slit.YIdlVert4+ap_slit.YIdlVert1)/2

# Where do the spectral traces end up in the two dithers? X coords in pixels
trace_center_dith1 = 316
trace_center_dith2 = 333



