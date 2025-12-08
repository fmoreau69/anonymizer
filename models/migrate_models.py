#!/usr/bin/env python
"""
Migration script to move model files from root to categorized subdirectories.

This script helps organize YOLO models by type. It's safe to run multiple times
and will skip files that are already in the correct location.

Usage:
    python migrate_models.py [--dry-run] [--cleanup]

Options:
    --dry-run   Show what would be done without making changes
    --cleanup   Remove duplicate models from root after verifying copies exist
"""

import os
import shutil
import sys
from pathlib import Path

# Model categorization based on filename patterns
MODEL_CATEGORIES = {
    'detect': [
        'yolov8n.pt',
        'yolov8s_faces',
        'yolov8m_faces',
        'yolov8n_faces',
        'yolov9t-face',
        'yolov9s-face',
    ],
    'segment': [
        '-seg.pt',
    ],
    'classify': [
        '-cls.pt',
    ],
    'pose': [
        '-pose.pt',
    ],
    'obb': [
        '-obb.pt',
    ],
}


def get_model_category(filename: str) -> str:
    """Determine the category of a model based on its filename."""
    for category, patterns in MODEL_CATEGORIES.items():
        for pattern in patterns:
            if pattern in filename:
                return category
    # Default to detect if no pattern matches
    return 'detect'


def migrate_models(dry_run=False, cleanup=False):
    """Migrate models from root to subdirectories."""
    script_dir = Path(__file__).parent
    models_root = script_dir

    print(f"Models directory: {models_root}\n")

    # Find all .pt files in root
    root_models = [f for f in os.listdir(models_root)
                   if f.endswith('.pt') and os.path.isfile(os.path.join(models_root, f))]

    if not root_models:
        print("No model files found in root directory.")
        return

    print(f"Found {len(root_models)} model(s) in root directory:\n")

    migrations = []
    for model_file in sorted(root_models):
        category = get_model_category(model_file)
        source = os.path.join(models_root, model_file)
        dest_dir = os.path.join(models_root, category)
        dest = os.path.join(dest_dir, model_file)

        # Check if already exists in destination
        already_exists = os.path.isfile(dest)

        migrations.append({
            'file': model_file,
            'category': category,
            'source': source,
            'dest': dest,
            'dest_dir': dest_dir,
            'exists': already_exists
        })

        status = "[OK] Already exists" if already_exists else "[->] Will copy"
        print(f"  {model_file:50} -> {category:10} {status}")

    print()

    if dry_run:
        print("DRY RUN - No changes made")
        return

    # Perform migrations
    print("Copying models to subdirectories...\n")
    copied = 0
    for migration in migrations:
        if not migration['exists']:
            os.makedirs(migration['dest_dir'], exist_ok=True)
            shutil.copy2(migration['source'], migration['dest'])
            print(f"  [OK] Copied {migration['file']} -> {migration['category']}/")
            copied += 1
        else:
            print(f"  [-] Skipped {migration['file']} (already exists)")

    print(f"\n[SUCCESS] Migration complete! Copied {copied} file(s).")

    # Cleanup phase
    if cleanup:
        print("\n[WARNING] CLEANUP MODE: Removing duplicate models from root directory...\n")
        removed = 0
        for migration in migrations:
            if migration['exists'] or os.path.isfile(migration['dest']):
                try:
                    os.remove(migration['source'])
                    print(f"  [OK] Removed {migration['file']} from root")
                    removed += 1
                except Exception as e:
                    print(f"  [ERROR] Failed to remove {migration['file']}: {e}")

        print(f"\n[SUCCESS] Cleanup complete! Removed {removed} file(s) from root.")
        print("\n[NOTE] Models in root directory have been removed.")
        print("       Ensure all systems are updated to use the new categorized paths.")
    else:
        print("\nTo remove duplicate models from root after verification, run:")
        print("  python migrate_models.py --cleanup")


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    cleanup = '--cleanup' in sys.argv

    if dry_run:
        print("=== DRY RUN MODE ===\n")

    if cleanup and not dry_run:
        print("=== CLEANUP MODE ===")
        print("This will REMOVE models from the root directory after copying.\n")
        response = input("Are you sure? This cannot be undone. (yes/no): ")
        if response.lower() != 'yes':
            print("Cleanup cancelled.")
            sys.exit(0)
        print()

    migrate_models(dry_run=dry_run, cleanup=cleanup)
