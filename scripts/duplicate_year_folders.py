#!/usr/bin/env python3
"""
Script to duplicate folders with different names for each folder in a year folder.
This script will create copies of each patient folder with a new naming convention.
"""

import os
import shutil
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def duplicate_folders_with_new_names(source_year_path, naming_pattern="copy_{original_name}", 
                                   dry_run=False, overwrite=False):
    """
    Duplicate all folders in a year directory with new names.
    
    Args:
        source_year_path (str): Path to the year folder containing patient folders
        naming_pattern (str): Pattern for new folder names. Use {original_name} as placeholder
        dry_run (bool): If True, only show what would be done without actually doing it
        overwrite (bool): If True, overwrite existing destination folders
    
    Returns:
        dict: Summary of operations performed
    """
    source_path = Path(source_year_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    
    if not source_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_path}")
    
    # Find all subdirectories (patient folders)
    patient_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    if not patient_folders:
        logger.warning(f"No subdirectories found in {source_path}")
        return {"total": 0, "copied": 0, "skipped": 0, "errors": 0}
    
    logger.info(f"Found {len(patient_folders)} folders to duplicate")
    
    results = {"total": len(patient_folders), "copied": 0, "skipped": 0, "errors": 0}
    
    for patient_folder in patient_folders:
        original_name = patient_folder.name
        
        # Generate new name using the pattern
        new_name = naming_pattern.format(original_name=original_name)
        destination_path = source_path / new_name
        
        try:
            if destination_path.exists() and not overwrite:
                logger.warning(f"Skipping {original_name} -> {new_name} (destination already exists)")
                results["skipped"] += 1
                continue
            
            if dry_run:
                logger.info(f"[DRY RUN] Would copy: {original_name} -> {new_name}")
                results["copied"] += 1
            else:
                logger.info(f"Copying: {original_name} -> {new_name}")
                shutil.copytree(patient_folder, destination_path, dirs_exist_ok=overwrite)
                results["copied"] += 1
                
        except Exception as e:
            logger.error(f"Error copying {original_name}: {str(e)}")
            results["errors"] += 1
    
    return results

def duplicate_with_custom_naming(source_year_path, naming_function=None, dry_run=False, overwrite=False):
    """
    Duplicate folders using a custom naming function.
    
    Args:
        source_year_path (str): Path to the year folder
        naming_function (callable): Function that takes original name and returns new name
        dry_run (bool): If True, only show what would be done
        overwrite (bool): If True, overwrite existing destination folders
    
    Returns:
        dict: Summary of operations performed
    """
    source_path = Path(source_year_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    
    if not source_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_path}")
    
    patient_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    if not patient_folders:
        logger.warning(f"No subdirectories found in {source_path}")
        return {"total": 0, "copied": 0, "skipped": 0, "errors": 0}
    
    logger.info(f"Found {len(patient_folders)} folders to duplicate")
    
    results = {"total": len(patient_folders), "copied": 0, "skipped": 0, "errors": 0}
    
    for patient_folder in patient_folders:
        original_name = patient_folder.name
        
        # Generate new name using the custom function
        if naming_function:
            new_name = naming_function(original_name)
        else:
            new_name = f"copy_{original_name}"
        
        destination_path = source_path / new_name
        
        try:
            if destination_path.exists() and not overwrite:
                logger.warning(f"Skipping {original_name} -> {new_name} (destination already exists)")
                results["skipped"] += 1
                continue
            
            if dry_run:
                logger.info(f"[DRY RUN] Would copy: {original_name} -> {new_name}")
                results["copied"] += 1
            else:
                logger.info(f"Copying: {original_name} -> {new_name}")
                shutil.copytree(patient_folder, destination_path, dirs_exist_ok=overwrite)
                results["copied"] += 1
                
        except Exception as e:
            logger.error(f"Error copying {original_name}: {str(e)}")
            results["errors"] += 1
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Duplicate folders with different names in a year folder')
    parser.add_argument('source_path', help='Path to the year folder containing patient folders')
    parser.add_argument('--pattern', default='copy_{original_name}', 
                       help='Naming pattern for new folders (use {original_name} as placeholder)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without actually doing it')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing destination folders')
    parser.add_argument('--prefix', help='Add prefix to all folder names')
    parser.add_argument('--suffix', help='Add suffix to all folder names')
    parser.add_argument('--replace', nargs=2, metavar=('OLD', 'NEW'), 
                       help='Replace OLD with NEW in folder names')
    
    args = parser.parse_args()
    
    # Create custom naming function if specific options are provided
    naming_function = None
    if args.prefix or args.suffix or args.replace:
        def custom_naming(original_name):
            new_name = original_name
            if args.replace:
                new_name = new_name.replace(args.replace[0], args.replace[1])
            if args.prefix:
                new_name = f"{args.prefix}{new_name}"
            if args.suffix:
                new_name = f"{new_name}{args.suffix}"
            return new_name
        naming_function = custom_naming
    
    try:
        if naming_function:
            results = duplicate_with_custom_naming(
                args.source_path, 
                naming_function=naming_function,
                dry_run=args.dry_run, 
                overwrite=args.overwrite
            )
        else:
            results = duplicate_folders_with_new_names(
                args.source_path, 
                naming_pattern=args.pattern,
                dry_run=args.dry_run, 
                overwrite=args.overwrite
            )
        
        # Print summary
        print("\n" + "="*50)
        print("DUPLICATION SUMMARY")
        print("="*50)
        print(f"Total folders found: {results['total']}")
        print(f"Successfully copied: {results['copied']}")
        print(f"Skipped (already exist): {results['skipped']}")
        print(f"Errors: {results['errors']}")
        print("="*50)
        
        if args.dry_run:
            print("\nThis was a dry run. No actual files were copied.")
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 