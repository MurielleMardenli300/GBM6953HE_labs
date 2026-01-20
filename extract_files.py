# import tarfile


# url = "https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.2d.v1.0.tar"
# tar_path = "/home/boadem/Work/School/neurite-oasis.2d.v1.0.tar"

# def extract_tar(tar_path, extract_path="/home/boadem/Work/School/neurite_data/"):
#     with tarfile.open(tar_path, "r") as tar:
#         tar.extractall(path=extract_path)
# extract_tar(tar_path)

# import lightning as pl 

# print(pl.__version__)

# from sklearn.model_selection import train_test_split
# import os

# with open("/home/boadem/Work/School/neurite_data/subjects.txt") as f:
#     subjects_list = f.read().splitlines()


# all_subjects_paths = [{"image" : os.path.join(r"/home/boadem/Work/School/neurite_data", f"{subj}", "slice_norm.nii.gz"), 
#                          "segmentation": os.path.join(r"/home/boadem/Work/School/neurite_data", f"{subj}", "slice_seg24.nii.gz")}
#                          for subj in subjects_list
#     ]

# train_set, test_set = train_test_split(all_subjects_paths, test_size=0.2, random_state=42, shuffle=True)

# import json
# with open(r"/home/boadem/Work/School/train_set_paths.json", "w") as f:
#     json.dump(train_set, f)
# with open(r"/home/boadem/Work/School/test_set_paths.json", "w") as f:
#     json.dump(test_set, f)

import torch
import sys

print("=" * 70)
print("PYTORCH CUDA DIAGNOSTICS")
print("=" * 70)

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (torch): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

if torch.cuda.is_available():
    print(f"\nNumber of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        
    # Test if we can create a tensor on GPU
    try:
        test_tensor = torch.tensor([1, 2, 3]).cuda()
        print(f"\n✓ Successfully created tensor on GPU")
        print(f"  Tensor device: {test_tensor.device}")
    except Exception as e:
        print(f"\n✗ Failed to create tensor on GPU: {e}")
else:
    print("\n✗ CUDA is NOT available to PyTorch!")
    print("\nPossible reasons:")
    print("1. PyTorch installed without CUDA support")
    print("2. CUDA drivers not installed or outdated")
    print("3. CUDA version mismatch")

print("\n" + "=" * 70)