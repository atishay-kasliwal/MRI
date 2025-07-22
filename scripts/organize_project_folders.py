import os
from pathlib import Path
import shutil

# Define the new structure
ROOT = Path('.')
folders = [
    'data/raw',
    'data/processed',
    'data/radiomics',
    'data/clinical',
    'notebooks',
    'scripts',
    'models',
    'results',
]

# Create folders if they don't exist
for folder in folders:
    path = ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured folder: {path}")

# Move files into their respective folders
for file in ROOT.glob('*'):
    if file.is_dir() or file.name.startswith('.'):
        continue
    # Raw data
    if file.suffix in ['.dcm', '.nii', '.nii.gz'] and 'CORRECT' not in file.name and 'mask' not in file.name:
        dest = ROOT / 'data/raw' / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file} to {dest}")
    # Processed data
    elif 'CORRECT' in file.name or 'mask' in file.name:
        dest = ROOT / 'data/processed' / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file} to {dest}")
    # Radiomics features
    elif 'radiomics' in file.name and file.suffix == '.csv':
        dest = ROOT / 'data/radiomics' / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file} to {dest}")
    # Clinical data
    elif 'clinical' in file.name or 'Patients' in file.name:
        dest = ROOT / 'data/clinical' / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file} to {dest}")
    # Notebooks
    elif file.suffix == '.ipynb':
        dest = ROOT / 'notebooks' / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file} to {dest}")
    # Scripts
    elif file.suffix == '.py':
        dest = ROOT / 'scripts' / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file} to {dest}")
    # Results
    elif file.suffix in ['.png', '.jpg', '.pdf', '.svg']:
        dest = ROOT / 'results' / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file} to {dest}")
    # Models
    elif file.suffix in ['.pkl', '.joblib', '.h5']:
        dest = ROOT / 'models' / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file} to {dest}")
    else:
        print(f"Unclassified: {file}")

print("\nAll files have been moved to their respective folders.") 