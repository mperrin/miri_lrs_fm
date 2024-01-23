import os
import copy
import functools

import numpy as np
import scipy
import scipy.interpolate
import matplotlib, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import astropy.io.fits as fits
import astropy.units as u
import astropy.coordinates

import jwst.datamodels
import jwst
import webbpsf
import poppy

from . import lrs_target_acq
from . import constants
from . import utils



def sim_offaxis_star(model, miri, star_coords, wcs_offset=(0,0), npix=80, verbose=False,
                     tweak_offset=None,
                     cube_waves=None,
                     **kwargs):
    """Simulate an off-axis star seen through the LRS slit
    """

    def vprint(*args):
        if verbose:
            print(*args)

    vprint(f"Host star coordinates: {star_coords.to_string('hmsdms')}")
    star_coords_pix = np.asarray(model.meta.wcs.world_to_pixel(star_coords))

    star_coords_pix -= np.asarray(wcs_offset)
    vprint(f" Using WCS offset = {wcs_offset}; got star_coords_pix = {star_coords_pix}")
    if tweak_offset is not None:
        star_coords_pix -= np.asarray(tweak_offset)
        vprint(f" Using extra offset tweak = {tweak_offset}; got star_coords_pix = {star_coords_pix}")
    delta_x = star_coords_pix[0] - constants.slit_center[0]
    delta_y = star_coords_pix[1] - constants.slit_center[1]

    miri.options['source_offset_x'] = delta_x * miri.pixelscale
    miri.options['source_offset_y'] = delta_y * miri.pixelscale
    vprint(f"Setup off-axis star at {miri.options['source_offset_x']}, {miri.options['source_offset_y']} arcsec relative to slit center")

    # TODO: Add wavelength-dependent MIRI cruciform model...

    if cube_waves is not None:
        vprint("data cube mode")
        psf_star_cube = miri.calc_datacube(cube_waves, progressbar=True,
                                           fov_pixels=npix, add_distortion=False, **kwargs)
        return psf_star_cube
    else:
        psf_star = miri.calc_psf(fov_pixels=npix, add_distortion=False, **kwargs)
        vprint("PSF calculation complete")
        return psf_star




def plot_taconfirm_psf_comparison(model, miri, star_coords, wcs_offset=(0,0),  tweak_offset=None,
                               box_size=80, vmax=20, saveplot=True):
    """ Generate and plot simulation of the PSF scene entering the LRS slit in a TACONFIRM image.
    Intended to help verify the model is setup correctly before the more time-consuming calculations of disperesed PSFs.
    """

    # Get arrays from the provided image:
    cutout = astropy.nddata.Cutout2D(model.data, constants.slit_center, box_size)
    cutout_dq = astropy.nddata.Cutout2D(model.dq, constants.slit_center, box_size)
    cutout_err = astropy.nddata.Cutout2D(model.err, constants.slit_center, box_size)


    print("Generating PSF sim for off-axis star:")
    psf_star = sim_offaxis_star(model, miri, star_coords, wcs_offset,
                                tweak_offset=tweak_offset,
                                verbose=True)

    imcopy = cutout.data.copy()

    # Generate a combined sim including PSF scaled and background model
    mask = np.isfinite(model.data[200:400, 450:800])
    stats = astropy.stats.sigma_clipped_stats(model.data[200:400, 450:800][mask])
    background_estimate = stats[1]

    # use left and right parts of slit to estimate flux ratio
    sampled_slit = get_slit_model(npix=box_size, pixelscale=miri.pixelscale)
    y, x = np.indices((box_size, box_size))
    mask = (sampled_slit>0.1) & ((x< box_size/2-12 ) | (x >box_size/2)) & np.isfinite(cutout.data) # mask to get pixels IN the slit, but EXCLUDING the target at its first dither pos

    # Find the scale factor and offset that best fit that PSF to the data, via a simple least-squares fit
    scalefactor, background_estimate = utils.find_scale_and_offset(psf_star[1].data[mask], cutout.data[mask],  cutout_err.data[mask])

    psfmodel = psf_star[1].data*scalefactor + sampled_slit*background_estimate


    # Determine off-axis star coordinates in pixels (for use in plotting)
    # Note, the actual values used in the calculation happen in sim_offaxis_star
    # this duplicates that calculation for plotting & labeling
    pix_coords = np.asarray(model.meta.wcs.world_to_pixel(star_coords))
    pix_coords -= np.asarray(wcs_offset)
    if tweak_offset is not None:
        pix_coords -= np.asarray(tweak_offset)

    # Plotting
    fig, axes = plt.subplots(figsize=(16,6), ncols=3)
    extent = [cutout.xmin_original-0.5, cutout.xmax_original+0.5,
              cutout.ymin_original-0.5, cutout.ymax_original+0.5,]

    norm = matplotlib.colors.AsinhNorm(vmin=0, vmax=vmax)
    cmap = matplotlib.cm.viridis
    cmap.set_bad('0.4')

    # Plot observed data
    axes[0].imshow(imcopy, norm=norm, cmap=cmap, extent=extent)
    axes[0].set_title("Cutout from \n"+ os.path.basename(model.meta.filename))
    axes[0].text(0.05, 0.05, f"Host star: {pix_coords[0]:.3f}, {pix_coords[1]:.3f} pix\n"+
                 f"using WCS offset={wcs_offset[0]:.3f}, {wcs_offset[1]:.3f}"+
                 (f"\ntweak offset={tweak_offset[0]:.3f}, {tweak_offset[1]:.3f}" if tweak_offset is not None else "\n "),
                color='white', transform=axes[0].transAxes)


    # Plot model
    axes[1].imshow(psfmodel, extent=extent,
                   norm=norm, cmap=cmap)
    axes[1].set_title("WebbPSF Simulation for \noff-axis host star PSF + background")

    # Plot Residuals of observed data - model
    axes[2].imshow(imcopy - psfmodel, norm=norm, cmap=cmap, extent=extent)
    axes[2].set_title("Residual \n Data - Model")

    for ax in axes:
        ax.plot(pix_coords[0],
               pix_coords[1],
               marker='+', markersize=20, color='red', label='Host star position (blocked)')
        ax.plot(constants.slit_center[0],
                constants.slit_center[1],
                marker='+', markersize=20, color='cyan', label='LRS slit center')
        ax.plot(constants.slit_closed_poly_points[0],
                constants.slit_closed_poly_points[1],
                color='white', ls=':', label='LRS slit aperture')

        ax.legend(loc='upper right')
        ax.set_xlabel("X pixels")
        ax.set_ylabel("Y pixels")

    fig.suptitle("Stellar PSF model for "+model.meta.target.catalog_name +
                 f" and host star seen in LRS Slit in {model.meta.instrument.filter}", fontweight='bold')
    fig.tight_layout()


    # Save output to FITS and PDF
    # TODO copy/add header metadata
    outname = f'lrs_taconfirm_fm_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}.fits'
    fits.writeto(outname,
                 np.stack((imcopy, psfmodel, imcopy-psfmodel)), overwrite=True)
    print(f" => Saved to {outname}")
    if saveplot:
        outname = f'lrs_taconfirm_fm_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}.pdf'
        plt.savefig(outname)
        print(f" => Saved to {outname}")


def setup_sim_to_match_file(filename):
    """ MIRI-LRS specific wrapper to webbpsf.setup_sim_to_match_file()

        Returns a WebbPSF instrument instance configured to match setup in a given file
    """

    miri = webbpsf.setup_sim_to_match_file(filename)

    # TODO verify it's actually an LRS exposure in that file?

    miri.image_mask = 'LRS slit'  # not automatically set in the webbpsf function, need to set this manually

    miri.pupil_mask = 'P750L'     # Ditto! Should make this automatic as well, for the LRS image, though not sure if also for the filters
                                  # After reviewing MIRI OBA doc, looks like 3.8% oversized stop is included on all
                                  # imaging filters, so same stop is relevant for regular filters as for LRS.
                                  # This is not quite correctly implemented in webbpsf right now, so should fix that.
                                  # To be confirmed with MIRI team. Also to check re pupil alignemtns
    return miri


@functools.cache
def get_slit_model(npix=80, pixelscale=0.10995457, oversample=25):
    """ Return a simple model of the LRS slit shape, sampled onto detector pixels """

    # TODO update this per filter, using miricoord?? For subtle per-filter offsets

    lrs_slit = poppy.RectangularFieldStop(width=constants.slit_width, height=constants.slit_height)

    sampled_slit = lrs_slit.sample(npix=npix*oversample, grid_size=npix*pixelscale)
    sampled_slit = poppy.utils.krebin(sampled_slit, (npix, npix)) / oversample**2

    return sampled_slit



def measure_dither_offset(model_dith1, model_dith2, plot=False, saveplot=False):
    """ Measure dither offset between two LRS exposures, from the WCS """

    # For dither 1, we know the pointing is exactly the same as the TA image,
    # so can reuse that WCS from the TA image for the 2D alignment.

    # For dither 2, since there is not any pointing confirmation image,
    # we infer an offset between the two dithered LRS exposure's WCSes

    targ_coords= lrs_target_acq.get_target_coords(model_dith1)

    ref_wave = astropy.coordinates.spectral_coordinate.SpectralCoord(5.1*u.micron)
    xy_dith1 = model_dith1.meta.wcs.world_to_pixel(targ_coords, ref_wave)
    xy_dith2 = model_dith2.meta.wcs.world_to_pixel(targ_coords, ref_wave)

    dither_offset = np.asarray(xy_dith2) - np.asarray(xy_dith1)
    print("Dither offset: ", dither_offset)

    if plot:
        fig, axes = plt.subplots(figsize=(9,16), ncols=2, gridspec_kw={'right':0.66})

        fig.suptitle(model_dith1.meta.filename)

        axes[0].imshow(model_dith1.data, norm=matplotlib.colors.AsinhNorm(vmin=0, vmax=np.nanmax(model_dith1.data)))
        axes[1].imshow(model_dith2.data, norm=matplotlib.colors.AsinhNorm(vmin=0, vmax=np.nanmax(model_dith1.data)))

        axes[0].set_title("Dither 1")
        axes[1].set_title("Dither 2")
        fig.text(0.7, 0.82, f"Dither offset:\n(inferred from WCS)\n\n\n$\Delta$x, $\Delta$y = {dither_offset[0]:.3f}, {dither_offset[1]:.3f} pix")

        for ax in axes:
            ax.set_xlim(300,350)
            ax.set_ylim(0,400)
            ax.axhline(385, color='orange')

        if saveplot:
            outname = f'lrs_dithers_jw{model_dith1.meta.observation.program_number}obs{model_dith1.meta.observation.observation_number}.pdf'
            plt.savefig(outname)
            print(f" => Saved to {outname}")



    return dither_offset




def generate_lrs_psf_cube(model_taconfirm, model_dispersed, miri,
                          star_coords, wcsoffset,
                          tweak_offset=None,
                          nlambda=None,
                          force_recalc=False,
                          plot=True):
    """ Make a PSF cube of LRS PSFs over many wavelengths, corresponding to some image.
   
    Parameters
    ----------
   
    nlambda : int
        number of wavelengths to simulate.
        Leave as None to simulate one PSF per wavelength row
    """
    # This needs both the TA confirmation image, for WCS inference of the host star position in 2D relative to the slit
    # and also the dispersed image for inference about the wavelength axes.

    # Infer the wavelength range and sampling from the FITS WCS
    ypos = np.arange(400)
    wavelen = model_dispersed.wavelength[ypos, int(constants.slit_center[0])]
   
    # Set up interpolators for converting back and forth between coordinates
    y_to_wave = scipy.interpolate.interp1d(ypos, wavelen)
    wave_to_y = scipy.interpolate.interp1d(wavelen, ypos)
   
    # delta wavelength per each bin
    y_to_deltawave = scipy.interpolate.interp1d(ypos, (np.roll(wavelen,1)-np.roll(wavelen,-1))/2)   
   
    converters = {"y_to_wave": y_to_wave,
                  "wave_to_y": wave_to_y,
                  "y_to_deltawave": y_to_deltawave}
   
   
    # Set up sampled wavelengths for simulation
    ymin=20
    ymax=385
    if nlambda is None:
        nlambda = ymax-ymin+1
    # TODO this hard-coded range is semi-arbitrary, though 385 is around the blue wave cutoff
    y_samp = np.linspace(ymax, ymin, nlambda, dtype=int)
    wave_samp = y_to_wave(y_samp)

   
    if plot:
        plt.figure()
        plt.plot( wavelen, ypos, label='wavelengths from WCS')
        plt.plot( wave_samp, y_samp, label='wavelengths for PSF sim', marker='o', ls='none')

        plt.xlabel("Wavelength")
        plt.ylabel("Y Pixel")
        plt.title("Wavelengths inferred from WCS")
        plt.legend()

    print(f"Generating monochromatic PSFS for {nlambda} wavelengths")


    model=model_dispersed
    outname = f'psfcube_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}exp{model.meta.observation.exposure_number}_nlam{nlambda}.fits'
    if os.path.exists(outname) and not force_recalc:
        print(f'Reloading PSF from {outname}')
        psfs_cube = fits.open(outname)
    else:
        psfs_cube = sim_offaxis_star(model_taconfirm, miri, star_coords, wcsoffset,
                                     cube_waves=wave_samp*1e-6, tweak_offset=tweak_offset,
                                     verbose=True)
        psfs_cube.writeto(outname, overwrite=True)
        print(f"Saved PSF to {outname} for reuse.")

    return psfs_cube, y_samp, wave_samp, converters



def generate_dispersed_lrs_model(psfs_cube, miri, wave_samp, converters, powerlaw=2, plot=True,
                                 cropwidth=22, xshift=1, add_cruciform=False):
    """Generate a dispersed model of an LRS spectrum, for some target in or near the slit

    Parameters
    ----------
    psfs_cube : datacube of monochromatic PSFs

    """

    n = 400

    dispersed_model = np.zeros((n,n), float)

    if add_cruciform:
        # Cruciform amplitude vs wavelength.
        # Values taken from webbpsf.detectors.apply_miri_scattering

        cruciform_ref_waves =     [5.6,     7.7,     10.0,    11.3,    12.8,  15.0]
        cruciform_ref_amplitude = [0.00445, 0.00285, 0.00065, 0.00009, 0.00014, 0.0]
        cruciform_amp_interp = scipy.interpolate.interp1d(cruciform_ref_waves,
                                                          cruciform_ref_amplitude,
                                                          fill_value='extrapolate')


    for iw, wave in enumerate(wave_samp):
        p = psfs_cube[1].data[iw]
        # Put the PSF into an oversized padded array
        padded = poppy.utils.pad_or_crop_to_shape(p, (n,n))

        # TODO: add wavelength-dependent cruciform, if that wasn't already done in the PSF generation

        if add_cruciform:
            kernel_amp = cruciform_amp_interp(wave)
            #print(wave, kernel_amp)
            kernel_x = webbpsf.detectors._make_miri_scattering_kernel(padded, kernel_amp, 1)
            im_conv_both = webbpsf.detectors._apply_miri_scattering_kernel(padded, kernel_x, 1)
            tot_int = padded.sum()
            padded += im_conv_both
            padded *= tot_int/padded.sum()

        # For each monochromatic PSF, we want to scale it in the calibrated image relative to the stellar spectrum
        # For this revised version of the code, we assume here the varying delta-lambda per each wavelength step
        # is already taken care of in the pipeline calibrations.
        # This should be the case since the image is calibrated into MJy/sr units.
        y = converters['wave_to_y'](wave)
        dy = (y-n//2)

        #dlambda = converters['y_to_deltawave'](y)

        # THIS IS A PLACEHOLDER, REPLACE WITH REAL STELLAR SPECTRUM RATHER THAN POWER LAW? Or spline fit?
        #scalefactor = dlambda * 1/wave**powerlaw
        scalefactor = 1/wave**powerlaw

        #print(wave, dy, scalefactor)

        # Shift the padded array, apply scale factor, add into acculumator
        dispersed_model += np.roll(np.roll(padded, int(dy), axis=0), xshift, axis=1) *scalefactor


    # Add detector IPC (This is pretty negligible for MIRI)
    sigma = webbpsf.constants.INSTRUMENT_DETECTOR_CHARGE_DIFFUSION_DEFAULT_PARAMETERS['MIRI']
    dispersed_model = scipy.ndimage.gaussian_filter(dispersed_model, sigma / miri.pixelscale)

    # Add detector cruciform. This requires shoving the data into a FITS HDUList as expected by webbpsf
    #hdu = fits.PrimaryHDU(dispersed_model)
    #hdu['INSTRUME'] = "MIRI"
    #hdu['FILTER'] = 'F770W'
    # TODO leave this for later...

    if plot:
        plt.figure(figsize=(12,12))

        dispersed_model_cropped = dispersed_model[:, n//2-cropwidth:n//2+cropwidth]
        print(n//2-cropwidth, n//2+cropwidth)

        vmx = 1e-5 # dispersed_model_cropped.max()
        print("Vmax", vmx)
        plt.imshow(dispersed_model_cropped,
                   norm=matplotlib.colors.AsinhNorm(vmin=0, vmax=vmx,
                  linear_width=dispersed_model_cropped.max()/5))

        ytickvals = np.linspace(0,400,9).clip(20,385)
        plt.yticks(ytickvals, [f'{int(y)}\n{converters["y_to_wave"](y):.02f} $\mu$m' for y in ytickvals])


    return dispersed_model_cropped
