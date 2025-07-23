import nibabel as nib
import numpy as np
from scipy.ndimage import label

def generate_mask(input_path, output_path=None, log_fn=print, component='largest', threshold_percentile=98.9):
    """
    Generate a binary mask from a NIfTI image by thresholding and extracting connected components.
    By default, keeps the largest component. Can be extended to keep the second or third largest.
    Args:
        input_path (str or Path): Path to the input NIfTI file.
        output_path (str or Path, optional): Path to save the mask NIfTI file. If None, does not save.
        log_fn (callable): Logging function.
        component (str or int): 'largest', 2, or 3 for largest, second, or third largest component.
        threshold_percentile (float): Percentile for thresholding (default 98.9).
    Returns:
        mask (np.ndarray): The generated mask array.
    """
    img = nib.load(str(input_path))
    data = img.get_fdata()
    # Use the provided percentile threshold to separate tissue from background
    threshold = np.percentile(data, threshold_percentile)
    mask = (data > threshold).astype(np.uint8)
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        log_fn(f"No connected components found in mask for {input_path}", 'warning')
        return mask
    # Get sizes of all components
    sizes = np.bincount(labeled_mask.flat)[1:]  # skip background
    order = np.argsort(sizes)[::-1]  # descending order
    if component == 'largest' or component == 1:
        idx = order[0] if len(order) > 0 else 0
    elif component == 2 and len(order) > 1:
        idx = order[1]
    elif component == 3 and len(order) > 2:
        idx = order[2]
    else:
        idx = order[0]  # fallback to largest
    selected_mask = (labeled_mask == (idx + 1)).astype(np.uint8)
    if output_path is not None:
        mask_img = nib.Nifti1Image(selected_mask, img.affine, img.header)
        nib.save(mask_img, str(output_path))
        log_fn(f"Mask saved: {output_path}")
    return selected_mask 