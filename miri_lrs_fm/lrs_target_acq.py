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
    im = ax.imshow(model.data, norm=norm, cmap=cmap)
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
    else:
        good = np.isfinite(cutout.data)

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
                    weights = 1./cutout_err.data[good])

    # Do a refit that more precisely constrains the FWHM to something reasonable
    result0.bounds['x_stddev'] = [0.5, 1.5]
    result0.bounds['y_stddev'] = [0.5, 1.5]

    result = fitter(result0, x[good], y[good], cutout.data[good]-med_bg,
                    weights = 1./cutout_err.data[good])


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
                   extent=extent)
        ax2.imshow(result(x,y),
                   norm=norm,
                   extent=extent)
        ax3.imshow(cutout.data-med_bg-result(x,y),
                   norm=norm,
                   extent=extent)
        ax1.set_title("Data Cutout")
        ax2.set_title("Model & Centroids")
        ax3.set_title("Residual vs Gauss2D")
        for ax in (ax1,ax2,ax3):
            ax.plot(result.x_mean, result.y_mean, color='white', marker='x', markersize=20)


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
                   extent=extent)
        ax2.imshow(result(x,y),
                   norm=norm,
                   extent=extent)
        ax3.imshow(cutout.data-med_bg-result(x,y),
                   norm=norm,
                   extent=extent)
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


    cutout = cutout = astropy.nddata.Cutout2D(model.data, constants.slit_center, box_size)

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
                   extent=extent)

        ax1.plot(constants.slit_center[0],
                 constants.slit_center[1],
                 marker='+', markersize=30, color='white', ls='none', label='LRS slit center')
        ax1.plot(constants.slit_closed_poly_points[0],
                 constants.slit_closed_poly_points[1],
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
    psf_star = lrs_fm.sim_offaxis_star(model, miri, star_coords, wcs_offset,
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
    axes[0].imshow(imcopy, norm=norm, cmap=cmap, extent=extent)
    axes[0].set_title("Cutout from \n"+ os.path.basename(model.meta.filename))

    # Plot model
    axes[1].imshow(psfmodel, extent=extent,
                   norm=norm, cmap=cmap)
    axes[1].set_title("WebbPSF Simulation for \noff-axis host star PSF + background")

    # Plot Residuals of observed data - model
    axes[2].imshow(imcopy - psfmodel, norm=norm, cmap=cmap, extent=extent)
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
