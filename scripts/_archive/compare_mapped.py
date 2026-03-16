#!/usr/bin/env python3
"""
Compare two files with correct split mappings.
File 1 (ptDecoder_ctc_both) -> File 2 (brain2text24_log.pkl):
  train -> train
  test -> val
  competition -> test
"""

import pickle
import torch
import numpy as np
from pathlib import Path


def load_pickle_file(filepath):
    """Load pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def extract_tensors_from_item(item, prefix=""):
    """Extract all tensors from a single data item."""
    tensors = {}

    if isinstance(item, dict):
        for key, value in item.items():
            new_prefix = f"{prefix}.{key}" if prefix else key

            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    tensor_key = f"{new_prefix}[{i}]"
                    if isinstance(v, torch.Tensor):
                        tensors[tensor_key] = v
                    elif isinstance(v, np.ndarray):
                        tensors[tensor_key] = torch.from_numpy(v)
            elif isinstance(value, torch.Tensor):
                tensors[new_prefix] = value
            elif isinstance(value, np.ndarray):
                tensors[new_prefix] = torch.from_numpy(value)

    return tensors


def compare_items(item1, item2, idx, split_name):
    """Compare two data items."""
    tensors1 = extract_tensors_from_item(item1)
    tensors2 = extract_tensors_from_item(item2)

    keys1 = set(tensors1.keys())
    keys2 = set(tensors2.keys())

    if keys1 != keys2:
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        return {
            'index': idx,
            'status': 'key_mismatch',
            'only_in_1': only_in_1,
            'only_in_2': only_in_2
        }

    # Compare tensors
    all_identical = True
    differences = []

    for key in sorted(keys1):
        t1 = tensors1[key]
        t2 = tensors2[key]

        if t1.shape != t2.shape:
            all_identical = False
            differences.append({
                'key': key,
                'type': 'shape_mismatch',
                'shape1': t1.shape,
                'shape2': t2.shape
            })
            continue

        if t1.dtype != t2.dtype:
            # Convert to same dtype
            if t1.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
                t1 = t1.float()
            if t2.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
                t2 = t2.float()

        if not torch.equal(t1, t2):
            all_identical = False

            # Convert to float if needed
            if t1.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
                t1 = t1.float()
            if t2.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
                t2 = t2.float()

            diff = (t1 - t2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            differences.append({
                'key': key,
                'type': 'value_difference',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'shape': t1.shape
            })

    if all_identical:
        return {'index': idx, 'status': 'identical'}
    else:
        return {
            'index': idx,
            'status': 'different',
            'num_differences': len(differences),
            'differences': differences
        }


def compare_splits(data1, data2, split_map):
    """Compare splits with mapping."""

    print(f"\n{'='*80}")
    print("SPLIT-WISE COMPARISON")
    print("="*80)

    for split1, split2 in split_map.items():
        print(f"\n{'-'*80}")
        print(f"Comparing: File1['{split1}'] ({len(data1[split1])} items) <-> File2['{split2}'] ({len(data2[split2])} items)")
        print(f"{'-'*80}")

        items1 = data1[split1]
        items2 = data2[split2]

        if len(items1) != len(items2):
            print(f"⚠️  WARNING: Different number of items!")
            print(f"  File 1: {len(items1)} items")
            print(f"  File 2: {len(items2)} items")
            min_len = min(len(items1), len(items2))
            print(f"  Will compare first {min_len} items")
        else:
            min_len = len(items1)

        # Compare each item
        identical_count = 0
        different_count = 0
        key_mismatch_count = 0

        all_results = []

        for i in range(min_len):
            result = compare_items(items1[i], items2[i], i, split1)
            all_results.append(result)

            if result['status'] == 'identical':
                identical_count += 1
            elif result['status'] == 'different':
                different_count += 1
            elif result['status'] == 'key_mismatch':
                key_mismatch_count += 1

        # Summary
        print(f"\nResults:")
        print(f"  ✅ Identical items: {identical_count}/{min_len} ({100*identical_count/min_len:.1f}%)")
        print(f"  ❌ Different items: {different_count}/{min_len} ({100*different_count/min_len:.1f}%)")
        if key_mismatch_count > 0:
            print(f"  ⚠️  Key mismatches: {key_mismatch_count}/{min_len}")

        # Show details for non-identical items
        if different_count > 0:
            print(f"\n  Details for different items (showing first 5):")
            shown = 0
            for result in all_results:
                if result['status'] == 'different' and shown < 5:
                    print(f"\n    Item {result['index']}:")
                    print(f"      Number of differences: {result['num_differences']}")

                    # Show top differences
                    diffs = result['differences']
                    shape_diffs = [d for d in diffs if d['type'] == 'shape_mismatch']
                    value_diffs = [d for d in diffs if d['type'] == 'value_difference']

                    if shape_diffs:
                        print(f"      Shape mismatches: {len(shape_diffs)}")
                        for d in shape_diffs[:3]:
                            print(f"        {d['key']}: {d['shape1']} vs {d['shape2']}")

                    if value_diffs:
                        print(f"      Value differences: {len(value_diffs)}")
                        # Sort by max_diff
                        value_diffs.sort(key=lambda x: x['max_diff'], reverse=True)
                        for d in value_diffs[:3]:
                            print(f"        {d['key']}: max_diff={d['max_diff']:.2e}, mean_diff={d['mean_diff']:.2e}")

                    shown += 1

        if key_mismatch_count > 0:
            print(f"\n  Items with key mismatches:")
            for result in all_results:
                if result['status'] == 'key_mismatch':
                    print(f"    Item {result['index']}:")
                    if result['only_in_1']:
                        print(f"      Keys only in File 1: {list(result['only_in_1'])[:5]}")
                    if result['only_in_2']:
                        print(f"      Keys only in File 2: {list(result['only_in_2'])[:5]}")


def main():
    file1 = Path("/home/ebrahim/data2/brain2text/b2t_24/ptDecoder_ctc_both")
    file2 = Path("/home/ebrahim/data2/brain2text/b2t_24/brain2text24_log.pkl")

    print("Loading files...")
    data1 = load_pickle_file(file1)
    data2 = load_pickle_file(file2)

    print(f"\nFile 1 splits: {list(data1.keys())}")
    print(f"File 2 splits: {list(data2.keys())}")

    # Correct mapping
    split_map = {
        'train': 'train',
        'test': 'val',
        'competition': 'test'
    }

    print(f"\nUsing mapping:")
    for s1, s2 in split_map.items():
        print(f"  File1['{s1}'] <-> File2['{s2}']")

    compare_splits(data1, data2, split_map)

    print(f"\n{'='*80}")
    print("OVERALL CONCLUSION")
    print("="*80)
    print("Comparison complete. See details above for each split.")


if __name__ == "__main__":
    main()
