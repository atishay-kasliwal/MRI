def manual_generate_mask_n_largest(nifti_path, threshold_percentile=99.9, n=2, output_path=None, log_fn=print):
    """
    Generate a mask from a NIfTI file using percentile thresholding and keep the N largest components.
    """
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import label
    from pathlib import Path

    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    threshold = np.percentile(data, threshold_percentile)
    mask = (data > threshold).astype(np.uint8)
    labeled_mask, num_features = label(mask)
    if num_features > 0:
        sizes = np.bincount(labeled_mask.flat)[1:]
        largest_indices = np.argsort(sizes)[-n:] + 1
        n_largest_mask = np.isin(labeled_mask, largest_indices).astype(np.uint8)
    else:
        n_largest_mask = mask
    # Naming convention
    if output_path is not None:
        out_path = Path(output_path)
        n_str = f"{n}largest_p{threshold_percentile}_"
        if not out_path.name.startswith(n_str):
            out_path = out_path.parent / (n_str + out_path.name)
    else:
        out_path = Path(nifti_path).parent / f"{n}largest_p{threshold_percentile}_mask_{Path(nifti_path).stem}.nii.gz"
    nib.save(nib.Nifti1Image(n_largest_mask, img.affine), str(out_path))
    log_fn(f"Mask saved: {out_path}")
    return out_path 