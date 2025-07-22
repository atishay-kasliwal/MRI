import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path

def n4_bias_correction(nifti_path, output_path=None, log_fn=print):
    """
    Apply N4 bias field correction to a NIfTI file and save the result.
    """
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    if data.ndim == 4:
        corrected = np.zeros_like(data)
        for i in range(data.shape[3]):
            vol = data[..., i].astype(np.float32)
            sitk_img = sitk.GetImageFromArray(vol)
            spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
            sitk_img.SetSpacing(spacing)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected_sitk = corrector.Execute(sitk_img)
            corrected[..., i] = sitk.GetArrayFromImage(corrected_sitk)
    elif data.ndim == 3:
        sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
        spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
        sitk_img.SetSpacing(spacing)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_sitk = corrector.Execute(sitk_img)
        corrected = sitk.GetArrayFromImage(corrected_sitk)
    else:
        raise ValueError("Unsupported image dimensions")
    out_path = output_path or (Path(nifti_path).parent / f"corrected_{Path(nifti_path).name}")
    nib.save(nib.Nifti1Image(corrected, img.affine, img.header), str(out_path))
    log_fn(f"N4 bias correction saved: {out_path}") 