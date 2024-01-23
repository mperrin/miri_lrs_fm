import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy

from . import constants

def estimate_background_spectrum(model, miri, plot=True, bothsides=True):
    """Make a simple estimate of the thermal background spectrum
    from the row-by-row median of the outer 1/6th of both left & right sides

    TODO more robust or outlier-rejected median?

    """
    constants.slit_width_pix = constants.slit_width/miri.pixelscale
    n=400

    # Make a 1D model for the diffuse illumination even behind the LRS slit holder
    # extract the left and right outer parts of the slit
    sci_cropped_bg_l = model.data[0:n,
                                  int(round(constants.slit_center[0] - constants.slit_width_pix/2)):
                                  int(round(constants.slit_center[0] - constants.slit_width_pix/3))]
    sci_cropped_bg_r = model.data[0:n,
                                  int(round(constants.slit_center[0] + constants.slit_width_pix/6)):
                                  int(round(constants.slit_center[0] + constants.slit_width_pix/2))]
    # combine them together
    if bothsides:
        sci_cropped_bg = np.hstack([sci_cropped_bg_l,sci_cropped_bg_r])
    else:
        sci_cropped_bg = sci_cropped_bg_l

    # take medians per row
    bg_1d = np.nanmedian(sci_cropped_bg, axis=1)
    bg_1d_filt = scipy.signal.medfilt(bg_1d, kernel_size=7)

    if plot:
        fig, axes=plt.subplots(ncols=2)
        cmap = matplotlib.cm.viridis

        axes[0].imshow(sci_cropped_bg, cmap=cmap, norm=matplotlib.colors.AsinhNorm(vmin=.001,vmax=np.nanmax(sci_cropped_bg))
        )
        axes[0].set_aspect(.1)

        axes[1].plot(bg_1d, np.arange(len(bg_1d)), label='Median per row')
        axes[1].plot(bg_1d_filt, np.arange(len(bg_1d)), label='Median, smoothed')
        axes[1].legend(fontsize=10)
        axes[0].set_ylabel("Y pixel")
        axes[1].set_xlabel(f'Background estimate [{model.meta.bunit_data}]');
        axes[1].set_ylim(0,400)
        return bg_1d_filt, axes

    return bg_1d_filt

