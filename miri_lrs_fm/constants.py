import pysiaf

# Read in constants from SIAF that we will use below
# Determine center of LRS slit, in pixels in the science frame

miri_siaf = pysiaf.Siaf('MIRI')
ap_slit = miri_siaf.apertures['MIRIM_SLIT']  # SLIT aperture type, does not have coordinates in det frame defined
ap_full = miri_siaf.apertures['MIRIM_FULL']

# retrieve/derive some values from the SIAF:

# Slit center, in pixels. **TODO** Update for slight filter dependence? 
slit_center = ap_full.tel_to_sci(ap_slit.V2Ref, ap_slit.V3Ref )

# corners of the slit, as seen on the detector
slit_closed_poly_points = ap_full.tel_to_sci(*ap_slit.closed_polygon_points('tel', rederive=False))

# Slit dimensions, in pixels
slit_width = (ap_slit.XIdlVert4+ap_slit.XIdlVert3)/2 - (ap_slit.XIdlVert1+ap_slit.XIdlVert2)/2
slit_height = (ap_slit.YIdlVert2+ap_slit.YIdlVert3)/2 - (ap_slit.YIdlVert4+ap_slit.YIdlVert1)/2




