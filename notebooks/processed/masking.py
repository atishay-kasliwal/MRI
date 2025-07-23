import numpy as np
import nibabel as nib
from scipy.ndimage import label
from pathlib import Path

def generate_mask(nifti_path, threshold_percentile=99.9, output_path=None, log_fn=print):
    """
    Generate a mask from a NIfTI file using percentile thresholding and largest component.
    """
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    threshold = np.percentile(data, threshold_percentile)
    mask = (data > threshold).astype(np.uint8)
    labeled_mask, num_features = label(mask)
    if num_features > 0:
        largest_component = (labeled_mask == np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1).astype(np.uint8)
    else:
        largest_component = mask
    out_path = output_path or (Path(nifti_path).parent / f"mask_{Path(nifti_path).stem}.nii.gz")
    nib.save(nib.Nifti1Image(largest_component, img.affine), str(out_path))
    log_fn(f"Mask saved: {out_path}")
    return out_path 