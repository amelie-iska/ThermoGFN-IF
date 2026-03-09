import torch
import sys
import collections

"""
Script to read and inspect the contents of a PyTorch .pt file.
Usage:
    python read_pt.py path/to/file.pt
"""

file_path = sys.argv[1]

try:
    data = torch.load(file_path, map_location=torch.device('cpu'))
    print(f"--- Contents of {file_path} ---")

    if isinstance(data, dict) or isinstance(data, collections.OrderedDict):
        print("File contains a dictionary (e.g., model state_dict or checkpoint). Keys and first few values:")
        for key, value in data.items():
            print(f"* Key: '{key}', Type: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            # Print a small slice/head of the actual tensor data if possible
            if hasattr(value, 'shape') and len(value.shape) > 0:
                print(f"  Head of value: {value.flatten()[:5]}...")
    elif hasattr(data, 'shape'):
        print(f"File contains a single tensor. Type: {type(data)}, Shape: {data.shape}")
        print(f"Head of tensor: {data.flatten()[:10]}...")
    else:
        print(f"File contains a non-tensor/non-dict object. Type: {type(data)}")

except Exception as e:
    print(f"Error loading the file: {e}")
    print("Ensure the file is a valid PyTorch file and torch is installed correctly.")
