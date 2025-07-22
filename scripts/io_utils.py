import nibabel as nib
from pathlib import Path

def load_nifti(file_path):
    """Load a NIfTI file and return the image object and data array."""
    img = nib.load(str(file_path))
    data = img.get_fdata()
    return img, data

def save_nifti(data, affine, header, out_path):
    """Save a numpy array as a NIfTI file."""
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(out_path)) 