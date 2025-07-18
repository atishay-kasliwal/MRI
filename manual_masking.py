import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from pathlib import Path

def manual_generate_mask(nifti_path, threshold_percentile=99.9, output_path=None, log_fn=print):
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
    # Prepend percentile to output filename if not already present
    if output_path is not None:
        out_path = Path(output_path)
        percentile_str = f"p{threshold_percentile}_"
        if not out_path.name.startswith(percentile_str):
            out_path = out_path.parent / (percentile_str + out_path.name)
    else:
        out_path = Path(nifti_path).parent / f"p{threshold_percentile}_mask_{Path(nifti_path).stem}.nii.gz"
    nib.save(nib.Nifti1Image(largest_component, img.affine), str(out_path))
    log_fn(f"Mask saved: {out_path}")
    return out_path

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

def manual_generate_mask_brightest(nifti_path, threshold_percentile=99.9, output_path=None, log_fn=print):
    """
    Generate a mask from a NIfTI file using percentile thresholding and keep the brightest component (by mean intensity).
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
        means = [data[labeled_mask == i + 1].mean() for i in range(num_features)]
        brightest_idx = np.argmax(means) + 1
        brightest_component = (labeled_mask == brightest_idx).astype(np.uint8)
    else:
        brightest_component = mask
    # Naming convention
    if output_path is not None:
        out_path = Path(output_path)
        bright_str = f"brightest_p{threshold_percentile}_"
        if not out_path.name.startswith(bright_str):
            out_path = out_path.parent / (bright_str + out_path.name)
    else:
        out_path = Path(nifti_path).parent / f"brightest_p{threshold_percentile}_mask_{Path(nifti_path).stem}.nii.gz"
    nib.save(nib.Nifti1Image(brightest_component, img.affine), str(out_path))
    log_fn(f"Mask saved: {out_path}")
    return out_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python manual_masking.py <input_nifti_path> [output_mask_path] [percentile] [all|nlargest N|brightest]")
        sys.exit(1)
    nifti_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    if len(sys.argv) > 3:
        try:
            threshold_percentile = float(sys.argv[3])
            mode = sys.argv[4] if len(sys.argv) > 4 else None
        except ValueError:
            threshold_percentile = 99.9
            mode = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        threshold_percentile = 99.9
        mode = None
    if mode == 'all':
        manual_generate_mask_all_regions(nifti_path, threshold_percentile, output_path)
    elif mode == 'nlargest' and len(sys.argv) > 5:
        n = int(sys.argv[5])
        manual_generate_mask_n_largest(nifti_path, threshold_percentile, n, output_path)
    elif mode == 'brightest':
        manual_generate_mask_brightest(nifti_path, threshold_percentile, output_path)
    else:
        manual_generate_mask(nifti_path, threshold_percentile, output_path) 