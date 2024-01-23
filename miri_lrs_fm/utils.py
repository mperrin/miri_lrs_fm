import os
import copy
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

import jwst

def get_crop_region_indices(model):
    """Find the subregion in an LRS cal image that has valid pixels

    For now, this leaves the Y axis range from 0-400, keeping some NaN pixels
    for simplicity in indexing (i.e. no y offset is needed for coords)
    But the X axis range is cropped exactly to the valid pixels
    """


    if model.meta.exposure.type != 'MIR_LRS-FIXEDSLIT':
        raise RuntimeError("This function is only intended to be used on LRS fixed slit data")
    # Ymin, Ymax, Xmin, Xmax
    ymin, ymax = 0, 400

    # Infer X indices from where the CAL file has valid pixels
    inds = np.where(np.isfinite(model.data).sum(axis=0))[0]
    xmin, xmax = inds.min(), inds.max() + 1  # add 1 for inclusive indexing

    #xmin, xmax = int(round(slit_center[0])-cropwidth), int(round(slit_center[0])+cropwidth)
    return ymin, ymax, xmin, xmax


def find_and_replace_outlier_pixels(model, nsigma = 7, plot=True, save_path=".", save=True):
    """identify and mask outlier pixels that escaped flagging by the pipeline

    Method: high pass filter via unsharp masking, i.e. subtracting a smoothed version of the image
    then identify pixels which are statistical outliers at high significance.

    """

    crop_indices = get_crop_region_indices(model)

    sci_cropped = model.data[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]
    sci_cropped_dq = model.dq[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]
    sci_cropped_err = model.err[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]




    cropped_smoothed = scipy.ndimage.median_filter(sci_cropped, size=(7,1))

    unsharp_masked = sci_cropped-cropped_smoothed

    outliers = (np.abs(unsharp_masked) > sci_cropped_err*7 )


    # make a copy of the data, and flag DO_NOT_USE for the outlier pixels
    model_to_clean = copy.deepcopy(model)
    model_to_clean.dq[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]] += outliers


    # Run the pipeline Pixel Replace step on that copy
    step = jwst.pixel_replace.PixelReplaceStep(save_results=True)
    model_cleaned = step.process(model_to_clean)

    # crop out the cleaned version for comparison
    cleaned_cropped = model_cleaned.data[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]

    if save:
        outname = os.path.join(save_path, model_cleaned.meta.filename.replace('cal', 'bpclean'))
        model_cleaned.write(outname)
        print(f"Saved to {outname}")

    if plot:
        fig, axes = plt.subplots(figsize=(9,16), ncols=4)

        norm = matplotlib.colors.AsinhNorm(vmin=0, vmax=np.nanmax(sci_cropped)/50,
                                                                    linear_width=np.nanmax(sci_cropped)/300)

        axes[0].imshow(sci_cropped, norm=norm)
        axes[0].set_title("SCI image", fontsize=10)
        axes[1].imshow(unsharp_masked, norm=norm)
        axes[1].set_title("high pass filtered", fontsize=10)
        axes[2].imshow(outliers ) #| np.isnan(sci_cropped))
        axes[2].set_title(f'>{nsigma}$\sigma$ outliers', fontsize=10)
        #axes[3].imshow(sci_cropped_dq & 1)
        #axes[3].set_title(f'DQ DO_NOT_USE', fontsize=10)
        axes[3].imshow(cleaned_cropped, norm=norm)
        axes[3].set_title("Pixel Replaced", fontsize=10)

        fig.suptitle("find_and_replace_outlier_pixels:\n"+model.meta.filename)

        for i,ax in enumerate(axes):
            ax.set_ylim(0,400)
            if i>0:
                ax.yaxis.set_ticklabels([])
        if save:
            plt.savefig(outname.replace(".fits", ".pdf"))

    return model_cleaned



def find_scale_and_offset(model, data, data_err):
    # Find the scale factor and offset that best fit that model to the data, via a simple least-squares fita

    from scipy.optimize import least_squares
    def objfun_scale_offset(params, sim, data, err):
        # Simple objective function to minimize scaled PSF to image
        scaled = sim*params[0] + params[1]

        chisq = ((data - scaled)**2/err**2).sum()
        return chisq

    args = (model, data, data_err) 
    res = least_squares(objfun_scale_offset, [1,0], args=args, x_scale='jac', diff_step=1.e-1)

    scalefactor, offset = res.x
    return scalefactor, offset

