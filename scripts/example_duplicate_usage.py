#!/usr/bin/env python3
"""
Example usage of the duplicate_year_folders.py script.
This script demonstrates different ways to duplicate folders with new names.
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to the path so we can import our module
sys.path.append(str(Path(__file__).parent))

from duplicate_year_folders import duplicate_folders_with_new_names, duplicate_with_custom_naming

def example_basic_duplication():
    """Example 1: Basic duplication with default naming pattern"""
    print("=== Example 1: Basic Duplication ===")
    
    # Example year folder path (replace with your actual path)
    year_folder = "data/2024"  # Change this to your actual year folder
    
    if not os.path.exists(year_folder):
        print(f"Year folder {year_folder} does not exist. Skipping example.")
        return
    
    # Duplicate with default pattern (copy_{original_name})
    results = duplicate_folders_with_new_names(
        source_year_path=year_folder,
        dry_run=True  # Set to False to actually perform the operation
    )
    
    print(f"Results: {results}")

def example_custom_pattern():
    """Example 2: Custom naming pattern"""
    print("\n=== Example 2: Custom Naming Pattern ===")
    
    year_folder = "data/2024"  # Change this to your actual year folder
    
    if not os.path.exists(year_folder):
        print(f"Year folder {year_folder} does not exist. Skipping example.")
        return
    
    # Duplicate with custom pattern
    results = duplicate_folders_with_new_names(
        source_year_path=year_folder,
        naming_pattern="backup_{original_name}_v2",
        dry_run=True
    )
    
    print(f"Results: {results}")

def example_custom_naming_function():
    """Example 3: Custom naming function"""
    print("\n=== Example 3: Custom Naming Function ===")
    
    year_folder = "data/2024"  # Change this to your actual year folder
    
    if not os.path.exists(year_folder):
        print(f"Year folder {year_folder} does not exist. Skipping example.")
        return
    
    # Define a custom naming function
    def custom_naming(original_name):
        # Add prefix, replace spaces with underscores, add suffix
        new_name = original_name.replace(' ', '_')
        new_name = f"processed_{new_name}_2024"
        return new_name
    
    results = duplicate_with_custom_naming(
        source_year_path=year_folder,
        naming_function=custom_naming,
        dry_run=True
    )
    
    print(f"Results: {results}")

def example_with_prefix_suffix():
    """Example 4: Using prefix and suffix options"""
    print("\n=== Example 4: Prefix and Suffix ===")
    
    year_folder = "data/2024"  # Change this to your actual year folder
    
    if not os.path.exists(year_folder):
        print(f"Year folder {year_folder} does not exist. Skipping example.")
        return
    
    def prefix_suffix_naming(original_name):
        return f"PRE_{original_name}_SUF"
    
    results = duplicate_with_custom_naming(
        source_year_path=year_folder,
        naming_function=prefix_suffix_naming,
        dry_run=True
    )
    
    print(f"Results: {results}")

def example_replace_text():
    """Example 5: Replace text in folder names"""
    print("\n=== Example 5: Replace Text ===")
    
    year_folder = "data/2024"  # Change this to your actual year folder
    
    if not os.path.exists(year_folder):
        print(f"Year folder {year_folder} does not exist. Skipping example.")
        return
    
    def replace_naming(original_name):
        # Replace "DE-IDENTIFIED" with "PROCESSED"
        return original_name.replace("DE-IDENTIFIED", "PROCESSED")
    
    results = duplicate_with_custom_naming(
        source_year_path=year_folder,
        naming_function=replace_naming,
        dry_run=True
    )
    
    print(f"Results: {results}")

def main():
    """Run all examples"""
    print("Duplicate Year Folders - Usage Examples")
    print("=" * 50)
    print("Note: All examples use --dry-run=True to show what would be done")
    print("Set dry_run=False to actually perform the operations")
    print("=" * 50)
    
    # Run examples
    example_basic_duplication()
    example_custom_pattern()
    example_custom_naming_function()
    example_with_prefix_suffix()
    example_replace_text()
    
    print("\n" + "=" * 50)
    print("Command Line Usage Examples:")
    print("=" * 50)
    print("# Basic duplication with default naming")
    print("python duplicate_year_folders.py data/2024")
    print()
    print("# Custom naming pattern")
    print("python duplicate_year_folders.py data/2024 --pattern 'backup_{original_name}_v2'")
    print()
    print("# Add prefix and suffix")
    print("python duplicate_year_folders.py data/2024 --prefix 'PRE_' --suffix '_SUF'")
    print()
    print("# Replace text in names")
    print("python duplicate_year_folders.py data/2024 --replace 'DE-IDENTIFIED' 'PROCESSED'")
    print()
    print("# Dry run (see what would be done without doing it)")
    print("python duplicate_year_folders.py data/2024 --dry-run")
    print()
    print("# Overwrite existing folders")
    print("python duplicate_year_folders.py data/2024 --overwrite")

if __name__ == "__main__":
    main() 