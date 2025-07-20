import nibabel as nib
import os

def is_valid_nifti(fpath):
    if not os.path.exists(fpath):
        return False
    if os.path.getsize(fpath) == 0:
        return False
    try:
        img = nib.load(fpath)
        _ = img.shape
        return True
    except Exception:
        return False 