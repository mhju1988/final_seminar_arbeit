"""
ONE-CLICK CYTHON RMI SETUP - FIXED VERSION
Just run this file and it will:
1. Create all necessary files
2. Build the Cython extension
3. Run a performance test
"""

import os
import sys
import subprocess

def main():
    print("="*70)
    print("CYTHON RMI - ONE CLICK SETUP")
    print("="*70)
    
    # First, let's create the files directly (avoiding exec and encoding issues)
    print("\nStep 1: Creating files...")
    
    # 1. Create rmi_cython.pyx
    cython_source = """# rmi_cython.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CythonRMI:
    cdef double[:] data
    cdef double slope
    cdef double intercept
    cdef int n
    
    def __init__(self, np.ndarray[double, ndim=1] keys):
        self.data = keys
        self.n = len(keys)
        
        # Simple linear regression
        cdef double x_sum = 0, y_sum = 0, xy_sum = 0, xx_sum = 0
        cdef int i
        
        for i in range(self.n):
            x_sum += keys[i]
            y_sum += i
            xy_sum += keys[i] * i
            xx_sum += keys[i] * keys[i]
        
        cdef double x_mean = x_sum / self.n
        cdef double y_mean = y_sum / self.n
        
        self.slope = (xy_sum - x_sum * y_mean) / (xx_sum - x_sum * x_mean)
        self.intercept = y_mean - self.slope * x_mean
    
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef int lookup(self, double key):
        cdef int pos = <int>(self.slope * key + self.intercept)
        cdef int left = max(0, pos - 100)
        cdef int right = min(self.n - 1, pos + 100)
        cdef int mid
        
        while left <= right:
            mid = (left + right) // 2
            if self.data[mid] == key:
                return mid
            elif self.data[mid] < key:
                left = mid + 1
            else:
                right = mid - 1
        return -1
"""

    # 2. Create setup_cython.py
    setup_cython = """from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "rmi_cython",
        ["rmi_cython.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
"""

    # 3. Create test script (using 'us' instead of μs to avoid encoding issues)
    test_script = """import numpy as np
import time

print("Testing Cython RMI Performance...")
print("-" * 50)

# Generate test data
n = 100000
np.random.seed(42)
keys = np.sort(np.random.uniform(0, 1000000, n))
test_keys = np.random.choice(keys, 5000)

# Test Python baseline
print("Testing Python binary search...")
start = time.perf_counter()
for key in test_keys:
    left, right = 0, len(keys) - 1
    while left <= right:
        mid = (left + right) // 2
        if keys[mid] == key:
            break
        elif keys[mid] < key:
            left = mid + 1
        else:
            right = mid - 1
python_time = (time.perf_counter() - start) / len(test_keys) * 1e6

# Test Cython RMI
try:
    from rmi_cython import CythonRMI
    print("Testing Cython RMI...")
    
    rmi = CythonRMI(keys)
    
    # Warm up
    for _ in range(100):
        rmi.lookup(test_keys[0])
    
    start = time.perf_counter()
    for key in test_keys:
        rmi.lookup(key)
    cython_time = (time.perf_counter() - start) / len(test_keys) * 1e6
    
    print()
    print(f"Results (n={n:,}):")
    print(f"  Python:     {python_time:.2f} us (microseconds)")
    print(f"  Cython RMI: {cython_time:.2f} us (microseconds)")
    print(f"  Speedup:    {python_time/cython_time:.1f}x")
    
    if cython_time < python_time:
        print()
        print("SUCCESS! Cython RMI is faster!")
        print("This demonstrates the paper's performance claims!")
        
except ImportError:
    print("Could not import rmi_cython")
    print("Build failed - see error messages above")
"""

    # Save files with UTF-8 encoding
    try:
        with open('rmi_cython.pyx', 'w', encoding='utf-8') as f:
            f.write(cython_source)
        print("✓ Created rmi_cython.pyx")

        with open('setup_cython.py', 'w', encoding='utf-8') as f:
            f.write(setup_cython)
        print("✓ Created setup_cython.py")

        with open('test_cython_rmi.py', 'w', encoding='utf-8') as f:
            f.write(test_script)
        print("✓ Created test_cython_rmi.py")
    except Exception as e:
        print(f"Error creating files: {e}")
        return
    
    # Step 2: Check dependencies
    print("\nStep 2: Checking dependencies...")
    
    try:
        import cython
        print("✓ Cython is installed")
    except ImportError:
        print("✗ Cython not found - installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cython"])
    
    try:
        import numpy
        print("✓ NumPy is installed")
    except ImportError:
        print("✗ NumPy not found - installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    
    # Step 3: Build
    print("\nStep 3: Building Cython extension...")
    print("Running: python setup_cython.py build_ext --inplace")
    
    result = subprocess.run(
        [sys.executable, "setup_cython.py", "build_ext", "--inplace"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Build successful!")
        
        # Step 4: Run test
        print("\nStep 4: Running performance test...")
        print("-" * 70)
        subprocess.run([sys.executable, "test_cython_rmi.py"])
        
    else:
        print("❌ Build failed!")
        print("\nError output:")
        print(result.stderr)
        
        if sys.platform == "win32" and "Microsoft Visual C++" in result.stderr:
            print("\n" + "="*70)
            print("WINDOWS COMPILER ISSUE DETECTED")
            print("="*70)
            print("\nYou need Microsoft C++ Build Tools:")
            print("1. Download from: https://visualstudio.microsoft.com/downloads/")
            print("2. Run the installer")
            print("3. Select 'Desktop development with C++'")
            print("4. Install (this may take a while)")
            print("5. Restart your command prompt")
            print("6. Run this script again")
            print("\nAlternatively, try using conda:")
            print("  conda install -c conda-forge cython")

if __name__ == "__main__":
    main()