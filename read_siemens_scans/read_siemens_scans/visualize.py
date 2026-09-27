import matplotlib as plt
from matplotlib.widgets import Slider
from matplotlib.axes import Axes

import numpy as np


def imshow(ax: Axes, image: np.ndarray, only_show: bool = True, **kwargs):
    im = ax.imshow(image, cmap='Greys_r', origin='lower', **kwargs)
    ax.set_axis_off()
    return None if only_show else im



# NOTE: This function is from the repo raw_kspace_recon created by Orel. We did not check it out yet.
def plot_slider_view_2d(images):
    # Initialize figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    # Display initial slice
    current_slice = 0
    current_contrast = 1.0
    # Apply initial vmin and vmax based on images data
    vmin = np.min(images)
    vmax = np.max(images)
    img = ax.imshow(images[current_slice, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f"Slice {current_slice}")

    # Add slider
    ax_slider = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_slice = Slider(ax_slider, 'Slice', 0, images.shape[0]-1, valinit=current_slice, valstep=1)

    # Update function for slider
    def update(val):
        slice_index = int(slider_slice.val)
        img.set_data(abs(images[slice_index, :, :]))
        ax.set_title(f"Slice {slice_index}")
        fig.canvas.draw_idle()

    slider_slice.on_changed(update)

    # Add slider for contrast adjustment
    ax_contrast = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_contrast = Slider(ax_contrast, 'Contrast', 0, 1.0, valinit=current_contrast, valstep=0.1)

    # Update function for contrast slider
    def update_contrast(val):
        global current_contrast
        current_contrast = slider_contrast.val
        slice_index = int(slider_slice.val)
        img.set_clim(vmin, current_contrast* vmax)
        img.set_data(abs(images[slice_index, :, :]))
        fig.canvas.draw_idle()

    slider_contrast.on_changed(update_contrast)


    plt.show()