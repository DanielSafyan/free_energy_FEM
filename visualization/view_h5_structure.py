import h5py
import sys

def print_h5_structure(name, obj):
    """Prints the name and type of an HDF5 object."""
    print(name, end="")
    if isinstance(obj, h5py.Group):
        print(" (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f" (Dataset: shape {obj.shape}, dtype {obj.dtype})")
    if obj.attrs:
        for key, val in obj.attrs.items():
            print(f"    - Attribute: {key} = {val}")

def view_h5_file(file_path):
    """Opens an HDF5 file and prints its structure."""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Structure of {file_path}:\n---")
            f.visititems(print_h5_structure)
            print("\n---")
    except (FileNotFoundError, IOError):
        print(f"Error: Could not open file {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_h5_structure.py <path_to_h5_file>")
    else:
        view_h5_file(sys.argv[1])
