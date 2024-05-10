import os
import copy

import numpy as np
import scipy
import matplotlib, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import astropy.io.fits as fits
import astropy.units as u
import astropy.coordinates

import stdatamodels
import jwst.datamodels
import jwst
import pysiaf
import webbpsf

import misc_jwst
import spaceKLIP


from . import constants
from . import lrs_fm
from . import utils

def get_target_coords(model):
    """Get target coordinates from FITS metadata"""
    return astropy.coordinates.SkyCoord(model.meta.target.ra,
                                        model.meta.target.dec,
                                        frame='icrs', unit=u.deg)


def plot_full_image(filename_or_datamodel, ax=None, vmax = 10, colorbar=True, colorbar_orientation='vertical'):
    """Plot a full-frame LRS image, with annotations"""

    if isinstance(filename_or_datamodel, stdatamodels.jwst.datamodels.JwstDataModel):
        model = filename_or_datamodel
    else:
        model = jwst.datamodels.open(filename_or_datamodel)

    norm = matplotlib.colors.Normalize(vmin=0.1, vmax=vmax)
    cmap = matplotlib.cm.viridis
    cmap.set_bad('orange')

    if ax is None:
        fig = plt.figure(figsize=(16,9))
        ax = plt.gca()

    imcopy = model.data.copy()
    imcopy[(model.dq &1)==1]  = np.nan
    im = ax.imshow(model.data, norm=norm, cmap=cmap, origin='lower')
    ax.set_title(model.meta.filename, fontweight='bold')


    # Metadata annotations

    annotation_text = f"{model.meta.target.proposer_name}\n{model.meta.instrument.filter}, {model.meta.exposure.readpatt}:{model.meta.exposure.ngroups}:{model.meta.exposure.nints}\n{model.meta.exposure.effective_exposure_time:.2f} s"

    try:
        wcs = model.meta.wcs
        # I don't know how to deal with the slightly different API of the GWCS class
        # so, this is crude, just cast it to a regular WCS and drop the high order distortion stuff
        # This suffices for our purposes in plotting compass annotations etc.
        # (There is almost certainly a better way to do this...)
        simple_wcs = astropy.wcs.WCS(model.meta.wcs.to_fits()[0])
    except:
        wcs = model.get_fits_wcs()
        if cube_ints:
            wcs = wcs.dropaxis(2)  # drop the nints axis

    if colorbar:
        # Colorbar

        cb = plt.gcf().colorbar(im, pad=0.1, aspect=60, label=model.meta.bunit_data,
                               orientation=colorbar_orientation)
        cbaxfn = cb.ax.set_xscale if colorbar_orientation=='horizontal' else cb.ax.set_yscale
        cbaxfn('asinh')

    if model.meta.exposure.type =='MIR_TACQ':
        labelstr="Target Acquisition Image"
    elif model.meta.exposure.type =='MIR_TACONFIRM':
        labelstr="Target Acquisition Verification Image"
    else:
        labelstr=""



    ax.set_xlabel("Pixels", fontsize='small')
    ax.set_ylabel("Pixels", fontsize='small')
    ax.tick_params(labelsize='small')
    ax.text(0.01, 0.99, annotation_text,
        transform=ax.transAxes, color='white', verticalalignment='top', fontsize=10)

    ax.text(0.5, 0.99, labelstr,
            style='italic', fontsize=10, color='white',
            horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    spaceKLIP.plotting.annotate_scale_bar(ax, model.data, simple_wcs, yf=0.07)

    # Leave off for now, this is not working ideally due to some imviz issue
    #spaceKLIP.plotting.annotate_compass(ax, model.data, wcs, yf=0.07, length_fraction=30)

def simple_position_fit(model, cutout_center_coords, box_size = 40, plot=False,
                        initial_estimate_stddev=3,
                       use_dq = True):

    cutout = astropy.nddata.Cutout2D(model.data, cutout_center_coords, box_size)
    cutout_dq = astropy.nddata.Cutout2D(model.dq, cutout_center_coords, box_size)
    cutout_err = astropy.nddata.Cutout2D(model.err, cutout_center_coords, box_size)
    print(cutout.xmin_original)

    if use_dq:
        good  = (cutout_dq.data & 1)==0
        weights = 1. / cutout_err.data[good]
    else:
        if np.isfinite(cutout_err.data).sum() >0 :
            good = np.isfinite(cutout.data) & np.isfinite(cutout_err.data) & (cutout_err.data != 0)
            weights = 1. / cutout_err.data[good]
        else:
            good = np.isfinite(cutout.data)
            weights = np.sqrt(np.abs(cutout.data))[good]  # treat as Poisson-dominated
    print(f"N good pixels to fit: {good.sum()}")

    y, x = np.indices(cutout.data.shape)
    x += cutout.xmin_original
    y += cutout.ymin_original

    med_bg = np.nanmedian(cutout.data)

    g_init = astropy.modeling.models.Gaussian2D(amplitude = np.nanmax(cutout.data),
                                                x_stddev=initial_estimate_stddev, y_stddev=initial_estimate_stddev,
                                      x_mean = cutout_center_coords[0], y_mean=cutout_center_coords[1])
    # Do an initial fit with larger bounds to get center
    g_init.bounds['x_stddev'] = [0.1, 3]
    g_init.bounds['y_stddev'] = [0.1, 3]

    fitter = astropy.modeling.fitting.LevMarLSQFitter()
    result0 = fitter(g_init, x[good], y[good], cutout.data[good]-med_bg,
                    weights=weights)

    # Do a refit that more precisely constrains the FWHM to something reasonable
    result0.bounds['x_stddev'] = [0.5, 1.5]
    result0.bounds['y_stddev'] = [0.5, 1.5]

    result = fitter(result0, x[good], y[good], cutout.data[good]-med_bg,
                    weights = weights)


    covariance = fitter.fit_info['param_cov']

    fitted_params = dict(zip(result.param_names, zip(result.parameters, np.diag(covariance)**0.5)))
    for k in fitted_params:
        print(f"{k} :  \t{fitted_params[k][0]:7.3f} +- {fitted_params[k][1]:.3f} ")


    if plot:
        fig, (ax1,ax2,ax3) = plt.subplots(figsize=(16,9), ncols=3)
        extent = [cutout.xmin_original-0.5, cutout.xmax_original+0.5,
                  cutout.ymin_original-0.5, cutout.ymax_original+0.5, ]

        norm = matplotlib.colors.AsinhNorm(vmin=0, vmax=np.nanmax(cutout.data),
                                           linear_width=np.nanmax(cutout.data)*0.01)
        ax1.imshow(cutout.data-med_bg,
                   norm=norm,
                   extent=extent, origin='lower')
        ax2.imshow(result(x,y),
                   norm=norm,
                   extent=extent, origin='lower')
        ax3.imshow(cutout.data-med_bg-result(x,y),
                   norm=norm,
                   extent=extent, origin='lower')
        ax1.set_title("Data Cutout")
        ax2.set_title("Model & Centroids")
        ax3.set_title("Residual vs Gauss2D")
        for ax in (ax1,ax2,ax3):
            ax.plot(result.x_mean, result.y_mean, color='white', marker='x', markersize=20, label='best fit')
            ax.plot(cutout_center_coords[0], cutout_center_coords[1], color='orange', marker='x', markersize=20, label='initial guess')
        ax1.legend()


    return result, covariance


def ta_position_fit_plot(model, box_size = 40, plot=True, saveplot=True, vmax=10, outname_extra=""):

    target_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit=u.deg)
    # target coordinates at epoch, as computed from APT
    print("Target RA, Dec at epoch:")
    print(target_coords)
    target_coords_pix = list(model.meta.wcs.world_to_pixel(target_coords))
    print(target_coords_pix)

    result, covariance = simple_position_fit(model, target_coords_pix, box_size=box_size)


    # Retrieve the OSS onboard centroids for comparison
    osslog = misc_jwst.engdb.get_ictm_event_log(startdate=model.meta.visit.start_time,
                                    enddate=model.meta.guidestar.visit_end_time)

    oss_centroid = misc_jwst.engdb.extract_oss_TA_centroids(osslog, "V"+model.meta.observation.visit_id, )
    print(f"Onboard OSS TA centroid (1-based): {oss_centroid}")
    print(f"    Agrees with Gaussian fit within: {oss_centroid[0]-1-result.x_mean:.4f}\t {oss_centroid[1]-1-result.y_mean:.4f} pix")

    # offset WCS - oSS
    wcs_offset = [target_coords_pix[i] - (oss_centroid[i] - 1) for i in range(2)]
    print(f"WCS offset relative to OSS: {wcs_offset}")

    if plot:

        # re-create some values we will need for the plot
        # this partially duplicates code from simple_position_fit for modularity
        cutout = astropy.nddata.Cutout2D(model.data, target_coords_pix, box_size)
        med_bg = np.nanmedian(cutout.data)
        y, x = np.indices(cutout.data.shape)
        x += cutout.xmin_original
        y += cutout.ymin_original



        # Now do some plots
        fig = plt.figure(figsize=(16, 9))

        gs = GridSpec(3, 4, figure=fig,
                      left=0.04, right=0.75, bottom=0.05, top=0.95,
                     hspace=0.4, wspace=0.15)
        ax0 = fig.add_subplot(gs[:, 0:3])
        ax1 = fig.add_subplot(gs[0, 3])
        ax2 = fig.add_subplot(gs[1, 3])
        ax3 = fig.add_subplot(gs[2, 3])

        ax0.add_artist(matplotlib.patches.Rectangle((cutout.xmin_original, cutout.ymin_original), box_size, box_size,
                            edgecolor='yellow', facecolor='none'))

        plot_full_image(model, ax=ax0, colorbar=False, vmax=vmax)

        extent = [cutout.xmin_original-0.5, cutout.xmax_original+0.5,
                  cutout.ymin_original-0.5, cutout.ymax_original+0.5, ]

        norm = matplotlib.colors.AsinhNorm(vmin=0, vmax=np.nanmax(cutout.data),
                                           linear_width=np.nanmax(cutout.data)*0.01)
        ax1.imshow(cutout.data-med_bg,
                   norm=norm,
                   extent=extent, origin='lower')
        ax2.imshow(result(x,y),
                   norm=norm,
                   extent=extent, origin='lower')
        ax3.imshow(cutout.data-med_bg-result(x,y),
                   norm=norm,
                   extent=extent, origin='lower')
        ax1.set_title("Data Cutout")
        ax2.set_title("Model & Centroids")
        ax3.set_title("Residual vs Gauss2D")

        for ax in (ax2,): #(ax1, ax2, ax3):
            ax.plot(oss_centroid[0]-1, oss_centroid[1]-1,
                     marker='x', color='pink', markersize=30, ls='none',
                     label='OSS centroid')
        ax2.plot(result.x_mean, result.y_mean,
                 marker='+', color='orange', markersize=30, ls='none',
                 label='Photutils Gaussian2D')
        ax2.plot(target_coords_pix[0], target_coords_pix[1],
                 marker='o', color='white', markersize=20, markerfacecolor='none', ls='none',
                 label='from WCS coords')

        ax2.legend(facecolor='none', frameon=True, labelcolor='white', fontsize='10',
                  markerscale=0.33)

        cprint = lambda xy: f"{xy[0]:.3f}, {xy[1]:.3f}"

        yt = 0.88
        line=0.03
        for i, (label, val) in enumerate((('OSS - 1', [o-1 for o in oss_centroid]),
                           ('Gaussian2D', (result.x_mean.value, result.y_mean.value)),
                           ('WCS', target_coords_pix))):
            fig.text(0.75, yt-line*i, f"{label+':':12s}{cprint(val)}", fontfamily='monospace')
        fig.text(0.75, yt+2*line, "TA Target Coords (pixels):", fontsize=14, fontweight='bold')

        fig.text(0.75, yt-line*5, f"{'Offset WCS-OSS:':12s}{cprint(wcs_offset)}", fontfamily='monospace')


        if saveplot:
            outname = f'lrs_ta_wcs_offset_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}{outname_extra}.pdf'
            plt.savefig(outname)
            print(f" => Saved to {outname}")

    return result, covariance, wcs_offset




def plot_ta_verification_image(model, wcs_offset=(0, 0), host_star_coords=None, tweak_offset=None,
                               plot=True, vmax=10, box_size=80, saveplot=True, outname_extra=''):

    target_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec, frame='icrs', unit=u.deg)
    # target coordinates at epoch, as computed from APT
    target_coords_pix = list(model.meta.wcs.world_to_pixel(target_coords))
    print(target_coords_pix)

    # Find the slit center in this particular image (note, this is slightly filter-dependent)
    # Separate into integer part (used for extracting a subarray) and subpixel part (for offsetting the slit model)

    if constants.USE_WCS_FOR_SLIT_COORDS:
        slit_center = model.meta.wcs.transform('v2v3', 'detector', constants.ap_slit.V2Ref,constants.ap_slit.V3Ref )
        slit_closed_poly_points = model.meta.wcs.transform('v2v3', 'detector', *constants.ap_slit.closed_polygon_points('tel', rederive=False))
    else:
        slit_center = constants.slit_center
        slit_closed_poly_points = constants.slit_closed_poly_points

    slit_center_rounded = np.round(slit_center)
    slit_center_subpix_offset = tuple(slit_center-slit_center_rounded)
    print(f'LRS Slit center in that image: {slit_center} pix; rounds to {slit_center_rounded} pix + {slit_center_subpix_offset}')

    cutout = cutout = astropy.nddata.Cutout2D(model.data, slit_center_rounded, box_size)

    # Get target coordinates at epoch, as computed from APT
    target_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec,
                                                 frame='icrs', unit=u.deg)
    target_coords_pix = np.asarray(model.meta.wcs.world_to_pixel(target_coords))
    print(f"Target coords from WCS: {target_coords_pix}")
    print(f"Using WCS offset from TA image: {wcs_offset}")
    target_coords_pix -= np.asarray(wcs_offset)
    if tweak_offset is not None:
        print(f"Using additional tweak offset: {tweak_offset}")
        target_coords_pix -= np.asarray(tweak_offset)
    print(f"Target coords adjusted: {target_coords_pix}")


    tfit_result, tfit_cov = simple_position_fit(model, target_coords_pix,
                                                use_dq=False, plot=False, initial_estimate_stddev=0.5)
    tfit_coords = np.asarray((tfit_result.x_mean.value, tfit_result.y_mean.value))

    if plot:

        med_bg = np.nanmedian(model.data)

        # Now do some plots
        fig = plt.figure(figsize=(16, 9))

        gs = GridSpec(3, 5, figure=fig,
                      left=0.04, right=0.95, bottom=0.05, top=0.95,
                     hspace=0.4, wspace=0.15)
        ax0 = fig.add_subplot(gs[:, 0:3])
        ax1 = fig.add_subplot(gs[0:2, 3:5])


        # Show main full image
        plot_full_image(model, ax=ax0, colorbar=False, vmax=vmax)
        ax0.add_artist(matplotlib.patches.Rectangle((cutout.xmin_original, cutout.ymin_original), box_size, box_size,
                            edgecolor='yellow', facecolor='none'))

        # Show cutout crop image
        extent = [cutout.xmin_original-0.5, cutout.xmax_original+0.5,
                  cutout.ymin_original-0.5, cutout.ymax_original+0.5, ]

        vmx = vmax=np.nanmax(cutout.data)
        norm = matplotlib.colors.AsinhNorm(vmin=-vmx*0.01, vmax=vmx ,
                                           linear_width=vmx*0.01)
        cmap = matplotlib.cm.viridis
        cmap.set_bad('orange')
        ax1.imshow(cutout.data-med_bg,
                   norm=norm, cmap=cmap,
                   extent=extent, origin='lower')

        ax1.plot(slit_center[0],
                 slit_center[1],
                 marker='+', markersize=30, color='white', ls='none', label='LRS slit center')

        # Slit corners, in this particular filter
        ax1.plot(slit_closed_poly_points[0],
                 slit_closed_poly_points[1],
                 color='white', ls=':', label='LRS slit aperture')
        ax1.plot(target_coords_pix[0],
                 target_coords_pix[1],
                 marker='x', markersize=30, color='gray', ls='none',
                 label='Science target pos., from WCS+offset')
        ax1.plot(tfit_coords[0],
                 tfit_coords[1],
                 marker='+', markersize=30, color='orange', ls='none',
                 label='Science target pos., from Gaussian2D')

        # Plot host star on the full frame image too
        ax0.plot(target_coords_pix[0],
                 target_coords_pix[1],
                 marker='x', markersize=30, color='gray', ls='none',
                 label='Science target pos., from WCS+offset')


        if host_star_coords is not None:
            host_star_coords_pix = np.asarray(model.meta.wcs.world_to_pixel(host_star_coords))
            #print("HOST STAR:")
            #print("DEBUG - PIX COORDS FROM WCS:", host_star_coords_pix)
            #print("DEBUG - WCS OFFSET: ", wcs_offset)
            host_star_coords_pix -= wcs_offset
            if tweak_offset is not None:
                host_star_coords_pix -= np.asarray(tweak_offset)
            #print("DEBUG - PIX COORDS W OFFSET:", host_star_coords_pix)

            ax1.plot(host_star_coords_pix[0],
                     host_star_coords_pix[1],
                     marker='+', markersize=30, color='red', ls='none',
                     label='Host star pos., from WCS+offset')


        ax1.legend(loc='lower right', facecolor=cmap(0), frameon=True, labelcolor='white', fontsize='12',
                   markerscale=0.5, framealpha=0.5)
        ax1.set_xlabel("X pixels")
        ax1.set_ylabel("Y pixels")

        ax1.set_title("Cutout around LRS slit")

        # Write text annotations
        cprint = lambda xy: f"{xy[0]:.3f}, {xy[1]:.3f}"

        yt = 0.2
        xt = 0.6
        line=0.03
        for i, (label, val) in enumerate( [
                           ('WCS+offset', target_coords_pix),
                           ('Gaussian2D', tfit_coords),
                            ]):
            fig.text(xt, yt-line*i, f"{label+':':14s}{cprint(val)}", fontfamily='monospace')
        fig.text(xt, yt+3*line, "Sci Target Coords (pixels):", fontsize=14, fontweight='bold')
        fig.text(xt, yt+2*line, f"{'Using WCS Offset = ':14s}{cprint(wcs_offset)}", fontfamily='monospace')
        if tweak_offset is not None:
            fig.text(xt, yt+1*line, f"{'Plus tweak Offset = ':14s}{cprint(tweak_offset)}", fontfamily='monospace')

        fig.text(xt, yt-line*3, f"{'WCS-Gauss2D:':14s}{cprint(target_coords_pix-tfit_coords)}", fontfamily='monospace')

        fig.text(xt, yt-5*line, "Host Star Coords (pixels):", fontsize=14, fontweight='bold')
        fig.text(xt, yt-6*line, f"{'WCS w offset:':14s}{cprint(host_star_coords_pix)}", fontsize=14, )


        if saveplot:
            outname = f'lrs_taconfirm_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}{outname_extra}.pdf'
            plt.savefig(outname)
            print(f" => Saved to {outname}")


def __OLD_VER_plot_taconfirm_psf_comparison(model, star_coords, wcs_offset=(0,0),  tweak_offset=None,
                               box_size=80):

    print("Generating PSF sim")
    psf_star = lrs_fm.sim_offset_source(model, miri, star_coords, wcs_offset,
                                tweak_offset=tweak_offset,
                                verbose=True)



    cutout = astropy.nddata.Cutout2D(model.data, constants.slit_center, box_size)
    cutout_dq = astropy.nddata.Cutout2D(model.dq, constants.slit_center, box_size)
    cutout_err = astropy.nddata.Cutout2D(model.err, constants.slit_center, box_size)

    pix_coords = np.asarray(model.meta.wcs.world_to_pixel(star_coords))
    #print("DEBUG - PIX COORDS FROM WCS:", pix_coords)
    #print("DEBUG - WCS OFFSET: ", wcs_offset)
    pix_coords -= np.asarray(wcs_offset)
    #print("DEBUG - PIX COORDS W OFFSET:", pix_coords)

    extent = [cutout.xmin_original-0.5, cutout.xmax_original+0.5,
              cutout.ymin_original-0.5, cutout.ymax_original+0.5,]

    fig, axes = plt.subplots(figsize=(16,6), ncols=3)

    norm = matplotlib.colors.AsinhNorm(vmin=0, vmax=20)
    cmap = matplotlib.cm.viridis
    cmap.set_bad('0.4')

    imcopy = cutout.data.copy()


    # Generate a combined sim including PSF scaled and background model
    mask = np.isfinite(model.data[200:400, 450:800])
    stats = astropy.stats.sigma_clipped_stats(model.data[200:400, 450:800][mask])
    background_estimate = stats[1]

    scalefactor = np.nansum(cutout.data[:,40:]) /  psf_star[1].data[:,40:].sum()

    psfmodel = psf_star[1].data*scalefactor/10 + sampled_slit*background_estimate/2


    # Plot observed data
    axes[0].imshow(imcopy, norm=norm, cmap=cmap, extent=extent, origin='lower')
    axes[0].set_title("Cutout from \n"+ os.path.basename(model.meta.filename))

    # Plot model
    axes[1].imshow(psfmodel, extent=extent,
                   norm=norm, cmap=cmap, origin='lower')
    axes[1].set_title("WebbPSF Simulation for \noff-axis host star PSF + background")

    # Plot Residuals of observed data - model
    axes[2].imshow(imcopy - psfmodel, norm=norm, cmap=cmap, extent=extent, origin='lower')
    axes[2].set_title("Residual \n Data - Model")

    axes[0].text(0.05, 0.05, f"Host star: {pix_coords[0]:.3f}, {pix_coords[1]:.3f} pix\n"+
                 f"using WCS offset={wcs_offset[0]:.3f}, {wcs_offset[1]:.3f}"+
                 (f"\ntweak offset={tweak_offset[0]:.3f}, {tweak_offset[1]:.3f}" if tweak_offset is not None else ""),
                 color='white', transform=axes[0].transAxes)

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


    outname = f'lrs_taconfirm_fm_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}.fits'
    fits.writeto(outname,
                 np.stack((imcopy, psfmodel, imcopy-psfmodel)), overwrite=True)
    plt.tight_layout()
    plt.savefig(f'lrs_taconfirm_fm_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}.pdf')




def plot_taconfirm_psf_fit_flux(model, miri, offset_star_coords=None, wcs_offset=(0,0),
                                tweak_offset=None,
                                companion_tweak_offset=None,
                                adjust_slit_center_offset=None,
                                plot=True, verbose=True,
                                box_size=80, vmax=20, saveplot=True, **kwargs):
    """ Generate and plot simulation of the PSF scene entering the LRS slit in a TACONFIRM image.
    Intended to help verify the model is setup correctly before the more time-consuming calculations of disperesed PSFs.

    Intended to have 

    Parameters
    ----------
    mostly self-evident

    tweak_offset : tuple of floats
        adjust overall WCS of the whole image
    companion_tweak_offset : tuple of floats
        adjust companion position relative to host star. Additive with the above.
    """

    # Get target coordinates at epoch, as computed from APT
    target_coords = astropy.coordinates.SkyCoord(model.meta.target.ra, model.meta.target.dec,
                                                 frame='icrs', unit=u.deg)
    target_coords_pix = np.asarray(model.meta.wcs.world_to_pixel(target_coords))
    print(f"Target coords from WCS: {target_coords_pix}")
    print(f"Using WCS offset from TA image: {wcs_offset}")
    target_coords_pix -= np.asarray(wcs_offset)
    if tweak_offset is not None:
        print(f"Using additional tweak offset: {tweak_offset}")
        target_coords_pix -= np.asarray(tweak_offset)
    else:
        tweak_offset=(0,0)

    if companion_tweak_offset is not None:
        print(f"Using additional tweak offset for companion only: {companion_tweak_offset}")
    else:
        companion_tweak_offset=(0,0)

    #print(f'DEBUG VERSION OVERRIDE COORDS')
    #target_coords_pix = np.asarray((317,300))
    #print(f"Target coords adjusted: {target_coords_pix}")

    # Find the slit center in this particular image (note, this is slightly filter-dependent)
    # Separate into integer part (used for extracting a subarray) and subpixel part (for offsetting the slit model)
    if constants.USE_WCS_FOR_SLIT_COORDS:
        slit_center = model.meta.wcs.transform('v2v3', 'detector', constants.ap_slit.V2Ref,constants.ap_slit.V3Ref )
        slit_closed_poly_points = model.meta.wcs.transform('v2v3', 'detector', *constants.ap_slit.closed_polygon_points('tel', rederive=False))
    else:
        slit_center = constants.slit_center
        #slit_center = tuple(c-1 for c in slit_center_pysiaf)  we SHOULD do this -- but something is off if we do. ???
        slit_closed_poly_points = constants.slit_closed_poly_points
        #raise NotImplementedError('need to code up subtraction for closed_poly_points (0/1 based)')

    slit_center_rounded = np.round(slit_center)
    slit_center_subpix_offset = tuple(slit_center-slit_center_rounded )
    # OVERRIDE - USE A HARD-CODED OFFSET HERE??
    #slit_center_subpix_offset_empirical = (0, -0.15)  # Empirical match
    #slit_center_subpix_offset= slit_center_subpix_offset_empirical  # Empirical match

    if adjust_slit_center_offset is not None:
        print(f'Overriding to adjust slit center by {adjust_slit_center_offset}')
        slit_center_subpix_offset = tuple(np.asarray(slit_center_subpix_offset) + np.asarray(adjust_slit_center_offset))


    print(f'LRS Slit center in that image: {slit_center} pix; converted to {slit_center_rounded} pix + {slit_center_subpix_offset}')

    # Get arrays from the provided image
    # FOR SOME REASON THE FOLLOWING ONLY WORKS RIGHT IF USING THE CONSTANTS.SLIT_CENTER, NOT THE ROUNDED VERSION
    # THERE'S AN ERRONEOUS OFFSET OTHERWISE
    cutout = astropy.nddata.Cutout2D(model.data, slit_center, box_size)
    cutout_dq = astropy.nddata.Cutout2D(model.dq, slit_center, box_size)
    cutout_err = astropy.nddata.Cutout2D(model.err, slit_center, box_size)

    if np.nansum(cutout_err.data)==0:
        print("Warning, the ERR extension for that file has no pixels with finite error around the LRS slit.")
        print("This can happen if the flat file doesn't have any valid uncertainty information")
        print("Making a new estimate of the error there from VAR_POISSON and VAR_RNOISE")
        variance_estimate = np.sqrt(model.var_poisson + model.var_rnoise)
        cutout_err = astropy.nddata.Cutout2D(np.sqrt(variance_estimate), slit_center, box_size)


    print(f"   Using slit_center_offset = {slit_center_subpix_offset} pix")
    print("Generating PSF sim for target:")
    psf_target = lrs_fm.sim_offset_source(model, miri, target_coords, wcs_offset,
                                tweak_offset=np.asarray(tweak_offset) + np.asarray(companion_tweak_offset),
                                add_distortion=True,
                                slit_center_offset=slit_center_subpix_offset,
                                verbose=verbose, **kwargs)
    if offset_star_coords:
        print("Generating PSF sim for off-axis star:")
        psf_star = lrs_fm.sim_offset_source(model, miri, offset_star_coords, wcs_offset,
                                    tweak_offset=tweak_offset,
                                    add_distortion=True,
                                    slit_center_offset=slit_center_subpix_offset,
                                    verbose=verbose, **kwargs)


    imcopy = cutout.data.copy()

    ext=3 # use the version including cruciform; this does make a difference

    # Generate a combined sim including PSF scaled and background model
    mask = np.isfinite(model.data[200:400, 450:800])
    stats = astropy.stats.sigma_clipped_stats(model.data[200:400, 450:800][mask])
    background_estimate = stats[1]

    # use left and right parts of slit to estimate flux ratio
    sampled_slit = lrs_fm.get_slit_model(npix=box_size, pixelscale=miri.pixelscale, slit_center_offset=slit_center_subpix_offset)
    y, x = np.indices((box_size, box_size))

    # Where and how much to mask out around the companion, for scaling the stellar PSF wings or conversely the companion PSF?
    trace_cen = 30.5
    trace_width = 4

    mask_slit = (sampled_slit>0.1) & np.isfinite(cutout.data) & np.isfinite(cutout_err.data)
    mask_for_offset_star = mask_slit & ((x< trace_cen-trace_width ) | (x >trace_cen+trace_width))  # mask to get pixels IN the slit, but EXCLUDING the target at its first dither pos
    mask_for_target = mask_slit & ((x> trace_cen-trace_width ) | (x <trace_cen+trace_width))  # mask to get pixels IN the slit, and ONLY INLCUDING the target at its first dither pos

    if offset_star_coords:
        # Find the scale factor and offset that best fit that PSF to the data, via a simple least-squares fit
        # Start with PSF wings
        scalefactor, background_estimate = utils.find_scale_and_offset(psf_star[ext].data[mask_for_offset_star],
                                                                       cutout.data[mask_for_offset_star],
                                                                       cutout_err.data[mask_for_offset_star])

        print(f"Initial fit flux scale factor for offset star: {scalefactor}")
        bkgd_psf_model = psf_star[ext].data*scalefactor + sampled_slit*background_estimate
    else:
        bkgd_psf_model = sampled_slit*background_estimate

    # Next do an initial fit for the companion, masked to just around the companion
    residuals = imcopy - bkgd_psf_model
    scalefactor_comp, background_estimate_comp = utils.find_scale_and_offset(psf_target[ext].data[mask_for_target],
                                                                             residuals[mask_for_target],
                                                                             cutout_err.data[mask_for_target])
    targetmodel = psf_target[ext].data*scalefactor_comp  #+ sampled_slit*background_estimate
    print(f"Initial fit flux scale factor for target: {scalefactor_comp}")

    if offset_star_coords:
        # Now let's do a joint fit of both to optimize simultaneously
        scalefactor, scalefactor_comp, background_estimate = utils.find_scales_2x_and_offset(psf_star[ext].data[mask_slit],
                                                                                 psf_target[ext].data[mask_slit],
                                                                                 cutout.data[mask_slit],
                                                                                 cutout_err.data[mask_slit],
                                                                                 initial_guess=(scalefactor, scalefactor_comp, background_estimate))
        print(f"Flux scale factor for star (joint fit): {scalefactor}")
        print(f"Flux scale factor for companion (joint fit): {scalefactor_comp}")


        # Apply those scale factors to update the scaled PSFs which comprise the forward model.
        bkgd_psf_model = psf_star[ext].data*scalefactor + sampled_slit*background_estimate
        targetmodel = psf_target[ext].data*scalefactor_comp
    else:
        scalefactor_comp, background_estimate_comp = utils.find_scale_and_offset(psf_target[ext].data[mask_slit],
                                                                                 residuals[mask_slit],
                                                                                 cutout_err.data[mask_slit])
        bkgd_psf_model = sampled_slit*background_estimate
        targetmodel = psf_target[ext].data*scalefactor_comp  #+ sampled_slit*background_estimate
    print(f"Flux scale factor for target: {scalefactor_comp}")
    scaled_targetmodel = targetmodel * u.Unit(model.meta.bunit_data) * (model.meta.photometry.pixelarea_steradians*u.sr)  # convert to MJy


    residuals = imcopy - bkgd_psf_model - targetmodel
    chisqr = (residuals**2 / cutout_err.data**2)[mask_slit].sum()
    ndof = np.isfinite((residuals**2 / cutout_err.data**2)[mask_slit]).sum() - 3
    print(f"reduced chi^2 = {chisqr/ndof}")
#    chisq = ((cutout.data[mask_slit] - scaled)**2/err**2).sum()


    print(f"Generating non-slit PSF for flux scaling")
    miri.image_mask = None
    miri.set_position_from_aperture_name('MIRIM_FULL')
    psf_offslit = lrs_fm.sim_offset_source(model, miri, target_coords, wcs_offset,
                            tweak_offset=np.asarray(tweak_offset) + np.asarray(companion_tweak_offset),
                            add_distortion=True,
                            slit_center_offset=slit_center_subpix_offset,
                                   npix=200,  # ensure nearly all flux in the aperture
                            verbose=True, **kwargs)
    miri.image_mask = 'LRS slit' # switch back after the off-slit calculation.
    miri.set_position_from_aperture_name('MIRIM_SLIT')
    targetmodel_offslit = psf_offslit[ext].data*scalefactor_comp  #+ sampled_slit*background_estimate

    scaled_targetmodel_offslit = targetmodel_offslit * u.Unit(model.meta.bunit_data) * (model.meta.photometry.pixelarea_steradians*u.sr)
    print(f"Scaled PSF model (through slit) flux in {model.meta.instrument.filter}: {scaled_targetmodel.sum().to(u.mJy)}")
    print(f"Scaled PSF model (no slit losses) flux in {model.meta.instrument.filter}: {scaled_targetmodel_offslit.sum().to(u.mJy)}")
    print('')


    if offset_star_coords:
        # Determine off-axis star coordinates in pixels (for use in plotting)
        # Note, the actual values used in the calculation happen in sim_offset_source
        # this duplicates that calculation for plotting & labeling
        pix_coords = np.asarray(model.meta.wcs.world_to_pixel(star_coords))
        pix_coords -= np.asarray(wcs_offset)
        if tweak_offset is not None:
            pix_coords -= np.asarray(tweak_offset)

    if plot:
        norm = matplotlib.colors.AsinhNorm(vmin=0, vmax=vmax)
        cmap = matplotlib.cm.viridis
        cmap.set_bad('0.4')
        extent = [cutout.xmin_original-0.5, cutout.xmax_original+0.5,
                  cutout.ymin_original-0.5, cutout.ymax_original+0.5,]


        if False:  # for now, deprecate the first two versions of this plot
            ############## Plotting ########################
            fig, axes = plt.subplots(figsize=(16,9), ncols=3, nrows=2)
            axes = axes.flat
            # Plot observed data
            axes[0].imshow(imcopy, norm=norm, cmap=cmap, extent=extent)
            axes[0].set_title("Cutout from \n"+ os.path.basename(model.meta.filename))
            #axes[0].text(0.05, 0.05, f"Host star: {pix_coords[0]:.3f}, {pix_coords[1]:.3f} pix\n"+
            axes[0].text(0.05, 0.05, f"Target: {target_coords_pix[0]:.3f}, {target_coords_pix[1]:.3f} pix\n"+
                         f"using WCS offset={wcs_offset[0]:.3f}, {wcs_offset[1]:.3f} pix"+
                         (f"\ntweak offset={tweak_offset[0]:.3f}, {tweak_offset[1]:.3f} pix" if tweak_offset is not None else "\n "),
                        color='white', transform=axes[0].transAxes)

            # Plot model
            axes[1].imshow(bkgd_psf_model, extent=extent,
                           norm=norm, cmap=cmap)
            axes[1].set_title("WebbPSF Simulation for \noff-axis host star PSF + background")

            # Plot Residuals of observed data - model
            axes[2].imshow(imcopy - bkgd_psf_model, norm=norm, cmap=cmap, extent=extent)
            axes[2].set_title("Residual \n Data - Star Model")

            # Plot substellar companion PSF
            axes[4].imshow(targetmodel, norm=norm, cmap=cmap, extent=extent)
            axes[4].set_title(f"Scaled Companion Model\n(Flux {scaled_targetmodel.sum().to(u.mJy):.3f} via LRS slit;\nFlux without slit losses {scaled_targetmodel_offslit.sum().to(u.mJy):.3f})")

            # Plot substellar companion PSF
            axes[5].imshow(residuals, norm=norm, cmap=cmap, extent=extent)
            axes[5].set_title("Residual\n Data - Star Model - Comp Model")

            # Plot the fitting mask
            axes[3].imshow(np.asarray(mask_slit, int) + np.asarray(mask_for_offset_star, int), extent=extent)
            axes[3].set_title("Mask regions for scale factor optimizations")
            #axes[3].set_visible(False)

            for ax in axes:
                if offset_star_coords:
                    ax.plot(pix_coords[0],
                           pix_coords[1],
                           marker='+', markersize=20, color='red', label='Host star position (blocked)')
                ax.plot(constants.slit_center[0],
                        constants.slit_center[1],
                        marker='+', markersize=20, color='cyan', label='LRS slit center')
                ax.plot(constants.slit_closed_poly_points[0],
                        constants.slit_closed_poly_points[1],
                        color='white', ls=':', label='LRS slit aperture')

                ax.plot(target_coords_pix[0],
                         target_coords_pix[1],
                         marker='x', markersize=30, color='gray', ls='none',
                         label='Science target pos., from WCS+offset')

                #ax.axhline(target_coords_pix[1], lw=0.1, color='white')
                #ax.axvline(target_coords_pix[0], lw=0.1, color='white')

                ax.legend(loc='upper right')
                ax.set_xlabel("X pixels")
                ax.set_ylabel("Y pixels")

            fig.suptitle("PSF model for "+model.meta.target.catalog_name +
                         f" seen in LRS Slit in {model.meta.instrument.filter}", fontweight='bold')
            fig.tight_layout()


            # Save output to FITS and PDF
            # TODO copy/add header metadata
            outname = f'lrs_taconfirm_fm_fit_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}.fits'
            fits.writeto(outname,
                         np.stack((imcopy, bkgd_psf_model, imcopy-bkgd_psf_model, targetmodel, imcopy-bkgd_psf_model-targetmodel)), overwrite=True)
            print(f" => Saved to {outname}")
            if saveplot:
                outname = f'lrs_taconfirm_fm_fit_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}.pdf'
                plt.savefig(outname)
                print(f" => Saved to {outname}")


        ######### Second, cleaner plot for paper:
        fig, axes = plt.subplots(figsize=(12,2.25), ncols=4, gridspec_kw={'width_ratios':[1,1,1,0.1], 'wspace':0.1,
                                                                      'top':0.8, 'bottom': 0.1, 'left':0.01, 'right':0.92})
        # Plot observed data
        axes[0].imshow(imcopy, norm=norm, cmap=cmap, extent=extent, interpolation='none')
        axes[0].set_title("TA Confirmation Image", fontsize=12) #\nfor "+ model.meta.target.catalog_name)

        # Plot Forward Model
        axes[1].imshow(bkgd_psf_model + targetmodel, norm=norm, cmap=cmap, extent=extent, interpolation='none')
        axes[1].set_title("Forward Model", fontsize=12)

        # Plot Residuals
        axes[2].imshow(residuals, norm=norm, cmap=cmap, extent=extent, interpolation='none')
        axes[2].set_title("Residuals", fontsize=12)

        fig.colorbar(mappable=axes[0].images[0], cax=axes[3], label='Surface Brightness [MJy/sr]', aspect=50,
                    ticks=[0.1,1,10,100])

        for ax in axes[0:3]:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_xlim(constants.slit_center[0]-30, constants.slit_center[0]+30)
            ax.set_ylim(constants.slit_center[1]-15, constants.slit_center[1]+15)

        axes[0].text(0.03, 0.95, model.meta.target.catalog_name, color='white', verticalalignment='top',
                     transform=axes[0].transAxes)
        plt.tight_layout()

        if saveplot:
            outname = f'lrs_taconfirm_forward_model_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}_3panel.png'
            plt.savefig(outname)
            plt.savefig(outname.replace('png', 'pdf'))
            print(f" => Saved to {outname} and .pdf")


        ######### Third plot showing FM components:
        xl=0.5

        def labelaxis(ax, text, above=True):
            if above:
                ax.set_title(text)
            else:
                ax.text(xl, 0.05, text, color='white', verticalalignment='bottom', horizontalalignment='center',
                     transform=ax.transAxes)


        fig, axes = plt.subplots(figsize=(12,2.5), ncols=7, gridspec_kw={'width_ratios':[1,1,1,1,1, 0.05, 0.05], 'wspace':0.02, # 01,
                                                                      'top':0.9, 'bottom': 0.05, 'left':0.01, 'right':0.92},
                                layout='constrained')
        axes[0].imshow(imcopy, norm=norm, cmap=cmap, extent=extent)
        #axes[0].set_title("Observed Data")
        axes[0].text(0.03, 0.95, model.meta.target.catalog_name, color='white', verticalalignment='top',
                     transform=axes[0].transAxes)
        labelaxis(axes[0], 'Observed data', )

        plt.tight_layout()

        axes[1].imshow(targetmodel, norm=norm, cmap=cmap, extent=extent)
        labelaxis(axes[1], 'Model: Target', )


        axes[2].imshow(bkgd_psf_model, extent=extent,
                       norm=norm, cmap=cmap)
        labelaxis(axes[2], 'Model: Background', )



        # Plot Residuals of observed data - model
        axes[3].imshow(bkgd_psf_model+targetmodel, norm=norm, cmap=cmap, extent=extent)
        labelaxis(axes[3], 'Combined forward model', )


        axes[4].imshow(residuals, norm=norm, cmap=cmap, extent=extent)
        labelaxis(axes[4], 'Residuals', )
        axes[4].text(0.03, 0.95, f"$\\chi^2_r$ = {chisqr/ndof:.3f}", color='white', verticalalignment='top',
                     transform=axes[4].transAxes)

        for ax in axes[0:5]:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            halfbox=27
            ax.set_xlim(constants.slit_center[0]-halfbox, constants.slit_center[0]+halfbox)
            ax.set_ylim(constants.slit_center[1]-halfbox, constants.slit_center[1]+halfbox)

        axes[5].set_visible(False) # this is just for padding cosmetics, skip some space
        fig.colorbar(mappable=axes[0].images[0], cax=axes[6], label='Surface Brightness [MJy/sr]', aspect=100,
                    ticks=[0.1,1,10,100], pad=0.9)
        fig.tight_layout()

        if saveplot:
            outname = f'plots_miri/lrs_taconfirm_forward_model_jw{model.meta.observation.program_number}obs{model.meta.observation.observation_number}_5panel.png'
            plt.savefig(outname)
            plt.savefig(outname.replace('png', 'pdf'))
            print(f" => Saved to {outname} and .pdf")


        ############ EXTRA DIAGNOSTIC FOR CHISQR

        if False:


            fig, axes = plt.subplots(figsize=(12,2.5), ncols=7, gridspec_kw={'width_ratios':[1,1,1,1,1, 0.05, 0.05], 'wspace':0.02, # 01,
                                                                          'top':0.9, 'bottom': 0.05, 'left':0.01, 'right':0.92},
                                    layout='constrained')
            axes[0].imshow(imcopy, norm=norm, cmap=cmap, extent=extent)
            axes[0].contour(mask_slit)
            #axes[0].set_title("Observed Data")
            axes[0].text(0.03, 0.95, model.meta.target.catalog_name, color='white', verticalalignment='top',
                         transform=axes[0].transAxes)
            labelaxis(axes[0], 'Observed data', )

            plt.tight_layout()

            axes[1].imshow(psfmodel, extent=extent,
                           norm=norm, cmap=cmap)
            labelaxis(axes[1], 'Model: host star + bkg.', )



            # Plot Residuals of observed data - model
            axes[2].imshow(np.abs(cutout_err.data), norm=norm, cmap=cmap, extent=extent)
            labelaxis(axes[2], '|err|', )


            # Plot Residuals of observed data - model
            axes[3].imshow(np.sqrt(residuals**2 /cutout_err.data**2 ), norm=norm, cmap=cmap, extent=extent)
            labelaxis(axes[3], '|(data-model)/err|', )


            axes[4].imshow(residuals, norm=norm, cmap=cmap, extent=extent)
            labelaxis(axes[4], 'Residuals', )
            axes[4].text(0.03, 0.95, f"$\\chi^2_r$ = {chisqr/ndof:.3f}", color='white', verticalalignment='top',
                         transform=axes[4].transAxes)

            for ax in axes[0:5]:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                halfbox=27
                ax.set_xlim(constants.slit_center[0]-halfbox, constants.slit_center[0]+halfbox)
                ax.set_ylim(constants.slit_center[1]-halfbox, constants.slit_center[1]+halfbox)

            axes[5].set_visible(False) # this is just for padding cosmetics, skip some space
            fig.colorbar(mappable=axes[0].images[0], cax=axes[6], label='Surface Brightness [MJy/sr]', aspect=100,
                        ticks=[0.1,1,10,100], pad=0.9)
            fig.tight_layout()

    return chisqr
