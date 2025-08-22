"""
Test script to check which sparse matrix formats in CuPy support item assignment.
"""

try:
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse
    
    print("CuPy version:", cp.__version__)
    print("CUDA available:", cp.cuda.is_available())
    
    if cp.cuda.is_available():
        print("\nTesting CuPy sparse matrix formats...")
        
        # Check what sparse matrix formats are available
        print("Available sparse matrix classes in CuPy:")
        sparse_classes = []
        for attr in dir(cupy_sparse):
            if not attr.startswith('_') and 'matrix' in attr.lower():
                print(f"  - {attr}")
                sparse_classes.append(attr)
        
        # Test each format for item assignment support
        print("\nTesting item assignment support:")
        for cls_name in sparse_classes:
            if cls_name in ['coo_matrix', 'csr_matrix', 'csc_matrix', 'dia_matrix']:
                cls = getattr(cupy_sparse, cls_name)
                try:
                    # Create a small matrix
                    matrix = cls((3, 3))
                    # Try item assignment
                    matrix[0, 1] = 1.0
                    print(f"  - {cls_name}: SUPPORTS item assignment")
                except Exception as e:
                    print(f"  - {cls_name}: Does NOT support item assignment ({type(e).__name__})")
    else:
        print("CUDA is not available.")
        
except ImportError:
    print("CuPy is not installed.")
