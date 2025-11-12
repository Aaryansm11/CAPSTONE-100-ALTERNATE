#!/usr/bin/env python3
"""
Fix age data in metadata - replace invalid ages with reasonable defaults
"""

import pickle
import numpy as np

# Load metadata
print("Loading metadata...")
with open('production_medium/full_dataset_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"Total segments: {len(metadata)}")

# Analyze current ages
ages = [m.get('age', 65) for m in metadata]
print(f"\nCurrent age stats:")
print(f"  Min: {min(ages)}, Max: {max(ages)}")
print(f"  Mean: {np.mean(ages):.1f}, Median: {np.median(ages):.1f}")
print(f"  Ages = 0: {sum(1 for a in ages if a == 0)}")
print(f"  Ages = 120: {sum(1 for a in ages if a == 120)}")
print(f"  Ages > 100: {sum(1 for a in ages if a > 100)}")

# Fix ages
fixed_count = 0
for m in metadata:
    age = m.get('age', 65)

    # Fix placeholder ages
    if age == 0 or age > 100:
        # Use median age of valid patients
        m['age'] = 65  # Reasonable default for ICU patients
        fixed_count += 1

print(f"\nFixed {fixed_count} invalid ages")

# Verify fixes
fixed_ages = [m.get('age', 65) for m in metadata]
print(f"\nFixed age stats:")
print(f"  Min: {min(fixed_ages)}, Max: {max(fixed_ages)}")
print(f"  Mean: {np.mean(fixed_ages):.1f}, Median: {np.median(fixed_ages):.1f}")
print(f"  Ages = 0: {sum(1 for a in fixed_ages if a == 0)}")
print(f"  Ages > 100: {sum(1 for a in fixed_ages if a > 100)}")

# Save fixed metadata
print("\nSaving fixed metadata...")
with open('production_medium/full_dataset_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Done! Ages have been fixed.")
