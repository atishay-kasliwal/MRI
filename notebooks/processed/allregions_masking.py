def manual_generate_mask_all_regions(nifti_path, threshold_percentile=99.9, output_path=None, log_fn=print):
    """
    Generate a mask from a NIfTI file using percentile thresholding (keep all regions above threshold).
    """
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    threshold = np.percentile(data, threshold_percentile)
    mask = (data > threshold).astype(np.uint8)
    if output_path is not None:
        out_path = Path(output_path)
        all_percentile_str = f"all_p{threshold_percentile}_"
        if not out_path.name.startswith(all_percentile_str):
            out_path = out_path.parent / (all_percentile_str + out_path.name)
    else:
        out_path = Path(nifti_path).parent / f"all_p{threshold_percentile}_mask_{Path(nifti_path).stem}.nii.gz"
    nib.save(nib.Nifti1Image(mask, img.affine), str(out_path))
    log_fn(f"Mask saved: {out_path}")
    return out_path 