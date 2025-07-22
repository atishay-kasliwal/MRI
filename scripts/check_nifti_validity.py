import os
import nibabel as nib

def check_nifti_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.nii.gz'):
                fpath = os.path.join(dirpath, fname)
                # Check if file exists and is non-empty
                if not os.path.exists(fpath):
                    print(f"Missing: {fpath}")
                    continue
                if os.path.getsize(fpath) == 0:
                    print(f"Empty: {fpath}")
                    continue
                # Try loading with nibabel
                try:
                    img = nib.load(fpath)
                    _ = img.shape  # Try accessing shape
                except Exception as e:
                    print(f"Corrupt or unreadable: {fpath} | Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_nifti_validity.py <root_dir>")
    else:
        check_nifti_files(sys.argv[1]) 