# Duplicate Year Folders Script

This script allows you to duplicate all folders within a year directory with new names. It's particularly useful for MRI data organization where you need to create copies of patient folders with different naming conventions.

## Features

- **Flexible Naming**: Use patterns, prefixes, suffixes, or custom naming functions
- **Safe Operations**: Dry-run mode to preview changes before executing
- **Overwrite Control**: Option to overwrite existing destination folders
- **Comprehensive Logging**: Detailed logging of all operations
- **Error Handling**: Graceful handling of errors with detailed reporting

## Usage

### Basic Usage

```bash
# Duplicate folders with default naming pattern (copy_{original_name})
python duplicate_year_folders.py /path/to/year/folder

# Example: Duplicate folders in data/2024
python duplicate_year_folders.py data/2024
```

### Custom Naming Patterns

```bash
# Use a custom naming pattern
python duplicate_year_folders.py data/2024 --pattern 'backup_{original_name}_v2'

# Add prefix and suffix
python duplicate_year_folders.py data/2024 --prefix 'PRE_' --suffix '_SUF'

# Replace text in folder names
python duplicate_year_folders.py data/2024 --replace 'DE-IDENTIFIED' 'PROCESSED'
```

### Safe Operations

```bash
# Dry run - see what would be done without actually doing it
python duplicate_year_folders.py data/2024 --dry-run

# Overwrite existing destination folders
python duplicate_year_folders.py data/2024 --overwrite
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `source_path` | Path to the year folder containing patient folders | `data/2024` |
| `--pattern` | Naming pattern for new folders (use `{original_name}` as placeholder) | `backup_{original_name}_v2` |
| `--dry-run` | Show what would be done without actually doing it | `--dry-run` |
| `--overwrite` | Overwrite existing destination folders | `--overwrite` |
| `--prefix` | Add prefix to all folder names | `--prefix 'PRE_'` |
| `--suffix` | Add suffix to all folder names | `--suffix '_SUF'` |
| `--replace` | Replace OLD with NEW in folder names | `--replace 'OLD' 'NEW'` |

## Examples

### Example 1: Basic Duplication
If you have folders like:
```
data/2024/
├── DE-IDENTIFIED, 6112052.brainlab
├── DE-IDENTIFIED, 7321220.brainlab
└── DE-IDENTIFIED, 6174233.brainlab
```

Running:
```bash
python duplicate_year_folders.py data/2024
```

Will create:
```
data/2024/
├── DE-IDENTIFIED, 6112052.brainlab
├── copy_DE-IDENTIFIED, 6112052.brainlab
├── DE-IDENTIFIED, 7321220.brainlab
├── copy_DE-IDENTIFIED, 7321220.brainlab
├── DE-IDENTIFIED, 6174233.brainlab
└── copy_DE-IDENTIFIED, 6174233.brainlab
```

### Example 2: Custom Pattern
```bash
python duplicate_year_folders.py data/2024 --pattern 'processed_{original_name}_2024'
```

Will create:
```
data/2024/
├── DE-IDENTIFIED, 6112052.brainlab
├── processed_DE-IDENTIFIED, 6112052.brainlab_2024
├── DE-IDENTIFIED, 7321220.brainlab
├── processed_DE-IDENTIFIED, 7321220.brainlab_2024
└── ...
```

### Example 3: Replace Text
```bash
python duplicate_year_folders.py data/2024 --replace 'DE-IDENTIFIED' 'PROCESSED'
```

Will create:
```
data/2024/
├── DE-IDENTIFIED, 6112052.brainlab
├── PROCESSED, 6112052.brainlab
├── DE-IDENTIFIED, 7321220.brainlab
├── PROCESSED, 7321220.brainlab
└── ...
```

## Programmatic Usage

You can also use the script programmatically:

```python
from duplicate_year_folders import duplicate_folders_with_new_names, duplicate_with_custom_naming

# Basic duplication
results = duplicate_folders_with_new_names(
    source_year_path="data/2024",
    naming_pattern="backup_{original_name}",
    dry_run=True
)

# Custom naming function
def custom_naming(original_name):
    return f"processed_{original_name.replace(' ', '_')}_2024"

results = duplicate_with_custom_naming(
    source_year_path="data/2024",
    naming_function=custom_naming,
    dry_run=False
)
```

## Output

The script provides detailed output including:

- **Progress Logging**: Real-time updates on what's being copied
- **Summary Report**: Final summary of operations performed
- **Error Reporting**: Detailed error messages for any failed operations

Example output:
```
2024-01-15 10:30:15 - INFO - Found 3 folders to duplicate
2024-01-15 10:30:15 - INFO - Copying: DE-IDENTIFIED, 6112052.brainlab -> copy_DE-IDENTIFIED, 6112052.brainlab
2024-01-15 10:30:16 - INFO - Copying: DE-IDENTIFIED, 7321220.brainlab -> copy_DE-IDENTIFIED, 7321220.brainlab
2024-01-15 10:30:17 - INFO - Copying: DE-IDENTIFIED, 6174233.brainlab -> copy_DE-IDENTIFIED, 6174233.brainlab

==================================================
DUPLICATION SUMMARY
==================================================
Total folders found: 3
Successfully copied: 3
Skipped (already exist): 0
Errors: 0
==================================================
```

## Safety Features

1. **Dry Run Mode**: Always test with `--dry-run` first to see what would be done
2. **Overwrite Protection**: By default, existing folders are not overwritten
3. **Error Handling**: Individual folder errors don't stop the entire process
4. **Detailed Logging**: All operations are logged for audit purposes

## Requirements

- Python 3.6+
- Standard library modules: `os`, `shutil`, `argparse`, `pathlib`, `logging`

## Notes

- The script only duplicates directories, not individual files
- All subdirectories and files within each folder are copied recursively
- The script preserves file permissions and timestamps
- Use `--dry-run` to preview changes before executing
- Consider disk space before duplicating large folders 