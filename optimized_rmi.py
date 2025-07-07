"""
Optimized Recursive Model Index implementation
Addresses performance issues in the original implementation
"""

import numpy as np
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
from BTrees.OOBTree import OOBTree

class OptimizedLinearModel:
    """Optimized linear model with minimal overhead"""
    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
    
    def fit(self, X, y):
        """Fast linear regression using normal equations"""
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y)
        
        # Add bias term
        X_with_bias = np.c_[np.ones(len(X)), X]
        
        # Normal equation: (X^T X)^-1 X^T y
        try:
            coeffs = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
            self.intercept = coeffs[0]
            self.slope = coeffs[1]
        except:
            # Fallback for edge cases
            self.slope = 0.0
            self.intercept = np.mean(y) if len(y) > 0 else 0.0
    
    def predict(self, x):
        """Fast prediction without sklearn overhead"""
        return self.slope * x + self.intercept


class OptimizedRMI:
    """Optimized Recursive Model Index"""
    
    def __init__(self, num_second_stage_models=10000):
        self.num_models = num_second_stage_models
        self.root_model = OptimizedLinearModel()
        self.models = []
        self.data = None
        self.n = 0
        # Store bounds more efficiently
        self.model_bounds = []  # (start_pos, end_pos) for each model
        self.search_bounds = []  # max search distance for each model
    
    def train(self, keys):
        """Train the RMI on sorted keys"""
        self.data = np.asarray(keys, dtype=np.float64)
        self.n = len(keys)
        positions = np.arange(self.n, dtype=np.float64)
        
        # Train root model
        self.root_model.fit(self.data, positions)
        
        # Distribute data to second-stage models
        model_assignments = []
        for i, key in enumerate(self.data):
            pred = self.root_model.predict(key)
            model_idx = int(pred * self.num_models / self.n)
            model_idx = np.clip(model_idx, 0, self.num_models - 1)
            model_assignments.append(model_idx)
        
        # Group data by model
        model_data = [[] for _ in range(self.num_models)]
        for i, model_idx in enumerate(model_assignments):
            model_data[model_idx].append((self.data[i], positions[i]))
        
        # Train second-stage models
        self.models = []
        self.model_bounds = []
        self.search_bounds = []
        
        for model_idx in range(self.num_models):
            if not model_data[model_idx]:
                # Empty model
                self.models.append(None)
                self.model_bounds.append((0, 0))
                self.search_bounds.append(0)
                continue
            
            # Extract keys and positions
            model_keys = np.array([x[0] for x in model_data[model_idx]])
            model_positions = np.array([x[1] for x in model_data[model_idx]])
            
            # Train model
            model = OptimizedLinearModel()
            model.fit(model_keys, model_positions)
            self.models.append(model)
            
            # Store actual position bounds
            min_pos = int(model_positions.min())
            max_pos = int(model_positions.max())
            self.model_bounds.append((min_pos, max_pos))
            
            # Calculate max prediction error for this model
            predictions = np.array([model.predict(k) for k in model_keys])
            errors = np.abs(predictions - model_positions)
            max_error = int(np.max(errors)) + 1 if len(errors) > 0 else 1
            self.search_bounds.append(max_error)
    
    def lookup(self, key):
        """Optimized lookup with minimal overhead"""
        # Stage 1: root model prediction
        pred = self.root_model.predict(key)
        model_idx = int(pred * self.num_models / self.n)
        model_idx = np.clip(model_idx, 0, self.num_models - 1)
        
        # Check if model exists
        if self.models[model_idx] is None:
            return -1
        
        # Stage 2: leaf model prediction
        pos_pred = self.models[model_idx].predict(key)
        pos = int(np.clip(pos_pred, 0, self.n - 1))  # Ensure position is in bounds
        
        # Get search bounds
        search_bound = self.search_bounds[model_idx]
        min_pos = max(0, pos - search_bound)
        max_pos = min(self.n - 1, pos + search_bound)
        
        # Quick check at predicted position
        if 0 <= pos < self.n and self.data[pos] == key:
            return pos
        
        # Exponential search to narrow the range
        if pos < self.n - 1 and self.data[pos] < key:
            # Search forward
            step = 1
            while pos + step <= max_pos and self.data[pos + step] < key:
                step *= 2
            min_pos = pos + step // 2
            max_pos = min(pos + step, max_pos)
        elif pos > 0 and self.data[pos] > key:
            # Search backward
            step = 1
            while pos - step >= min_pos and self.data[pos - step] > key:
                step *= 2
            max_pos = pos - step // 2
            min_pos = max(pos - step, min_pos)
        
        # Binary search in the narrowed range
        while min_pos <= max_pos:
            mid = (min_pos + max_pos) // 2
            if self.data[mid] == key:
                return mid
            elif self.data[mid] < key:
                min_pos = mid + 1
            else:
                max_pos = mid - 1
        
        return -1
    
    def memory_size_mb(self):
        """Estimate memory usage"""
        # Root model: 2 floats
        size = 2 * 8
        
        # Each second-stage model: 2 floats + bounds
        size += self.num_models * (2 * 8 + 3 * 4)  # model params + bounds
        
        return size / (1024 * 1024)


class OptimizedBTreeWrapper:
    """Optimized B+ tree wrapper with caching"""
    def __init__(self):
        self.tree = OOBTree()
        self._size = 0
    
    def train(self, keys):
        """Build tree from keys"""
        for i, key in enumerate(keys):
            self.tree[key] = i
        self._size = len(keys)
    
    def lookup(self, key):
        """Cached lookup"""
        return self.tree.get(key, -1)
    
    def memory_size_mb(self):
        """Estimate memory - more accurate"""
        # Each entry: key (8 bytes) + value (8 bytes) + pointers (~24 bytes)
        return (self._size * 40) / (1024 * 1024)


def run_optimized_benchmark():
    """Run optimized benchmark with better measurements"""
    print("=== Optimized Learned Index Benchmark ===\n")
    
    # Test different sizes to see scaling
    sizes = [10000, 50000, 100000]
    
    for n in sizes:
        print(f"\n--- Dataset size: {n:,} ---")
        
        # Generate data
        np.random.seed(42)
        keys = np.sort(np.random.lognormal(0, 2, n))
        
        # B+ Tree
        print("B+ Tree:")
        btree = OptimizedBTreeWrapper()
        start = time.time()
        btree.train(keys)
        build_time = time.time() - start
        
        # Warm up cache
        for _ in range(100):
            btree.lookup(keys[n//2])
        
        # Benchmark
        num_lookups = min(10000, n)
        lookup_keys = np.random.choice(keys, num_lookups)
        
        start = time.perf_counter()
        for key in lookup_keys:
            btree.lookup(key)
        btree_time = (time.perf_counter() - start) / num_lookups * 1e6
        
        print(f"  Build time: {build_time:.3f}s")
        print(f"  Lookup time: {btree_time:.2f} μs")
        print(f"  Memory: {btree.memory_size_mb():.2f} MB")
        
        # Optimized RMI
        for num_models in [1000, 10000]:
            print(f"\nRMI with {num_models} models:")
            rmi = OptimizedRMI(num_models)
            
            start = time.time()
            rmi.train(keys)
            build_time = time.time() - start
            
            # Warm up
            for _ in range(100):
                rmi.lookup(keys[n//2])
            
            # Benchmark
            start = time.perf_counter()
            for key in lookup_keys:
                rmi.lookup(key)
            rmi_time = (time.perf_counter() - start) / num_lookups * 1e6
            
            speedup = btree_time / rmi_time
            
            print(f"  Build time: {build_time:.3f}s")
            print(f"  Lookup time: {rmi_time:.2f} μs (Speedup: {speedup:.2f}x)")
            print(f"  Memory: {rmi.memory_size_mb():.2f} MB")
            
            # Analyze search bounds
            avg_bound = np.mean(rmi.search_bounds)
            max_bound = np.max(rmi.search_bounds) if rmi.search_bounds else 0
            print(f"  Avg search bound: {avg_bound:.1f}, Max: {max_bound}")


def analyze_performance_issues():
    """Analyze why the original implementation was slow"""
    print("\n=== Performance Analysis ===\n")
    
    n = 50000
    keys = np.sort(np.random.uniform(0, 1000, n))
    
    # Time different components
    print("Component timing analysis:")
    
    # 1. Sklearn model prediction overhead
    from sklearn.linear_model import LinearRegression
    X = keys.reshape(-1, 1)
    y = np.arange(n)
    
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    
    opt_model = OptimizedLinearModel()
    opt_model.fit(keys, y)
    
    # Time predictions
    num_preds = 10000
    test_keys = np.random.choice(keys, num_preds)
    
    start = time.perf_counter()
    for k in test_keys:
        sklearn_model.predict([[k]])
    sklearn_time = (time.perf_counter() - start) / num_preds * 1e6
    
    start = time.perf_counter()
    for k in test_keys:
        opt_model.predict(k)
    opt_time = (time.perf_counter() - start) / num_preds * 1e6
    
    print(f"  Sklearn prediction: {sklearn_time:.2f} μs per prediction")
    print(f"  Optimized prediction: {opt_time:.2f} μs per prediction")
    print(f"  Speedup: {sklearn_time/opt_time:.1f}x")
    
    # 2. Binary search performance
    data_array = np.array(keys)
    
    def binary_search_range(arr, key, min_pos, max_pos):
        while min_pos <= max_pos:
            mid = (min_pos + max_pos) // 2
            if arr[mid] == key:
                return mid
            elif arr[mid] < key:
                min_pos = mid + 1
            else:
                max_pos = mid - 1
        return -1
    
    # Test with different search ranges
    print("\nBinary search performance by range size:")
    for range_size in [10, 100, 1000]:
        start_pos = n // 2
        
        start = time.perf_counter()
        for _ in range(1000):
            binary_search_range(data_array, keys[start_pos], 
                              start_pos - range_size, 
                              start_pos + range_size)
        search_time = (time.perf_counter() - start) / 1000 * 1e6
        
        print(f"  Range ±{range_size}: {search_time:.2f} μs")


def create_performance_visualization():
    """Create a visualization of RMI vs B-tree performance"""
    print("\n=== Creating Performance Visualization ===")
    
    # Test on different dataset sizes
    sizes = [1000, 5000, 10000, 50000]
    btree_times = []
    rmi_times = []
    rmi_memory = []
    btree_memory = []
    
    for n in sizes:
        print(f"Testing n={n}...")
        keys = np.sort(np.random.lognormal(0, 2, n))
        
        # B-tree
        btree = OptimizedBTreeWrapper()
        btree.train(keys)
        
        test_keys = np.random.choice(keys, min(1000, n))
        start = time.perf_counter()
        for key in test_keys:
            btree.lookup(key)
        btree_times.append((time.perf_counter() - start) / len(test_keys) * 1e6)
        btree_memory.append(btree.memory_size_mb())
        
        # RMI
        rmi = OptimizedRMI(num_second_stage_models=int(np.sqrt(n) * 10))
        rmi.train(keys)
        
        start = time.perf_counter()
        for key in test_keys:
            rmi.lookup(key)
        rmi_times.append((time.perf_counter() - start) / len(test_keys) * 1e6)
        rmi_memory.append(rmi.memory_size_mb())
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Lookup time comparison
    ax1.plot(sizes, btree_times, 'b-o', label='B+ Tree', markersize=8)
    ax1.plot(sizes, rmi_times, 'g-s', label='Optimized RMI', markersize=8)
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Lookup Time (μs)')
    ax1.set_title('Lookup Performance Scaling')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Memory comparison
    ax2.plot(sizes, btree_memory, 'b-o', label='B+ Tree', markersize=8)
    ax2.plot(sizes, rmi_memory, 'g-s', label='Optimized RMI', markersize=8)
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Memory Usage Scaling')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print speedup summary
    print("\nSpeedup Summary:")
    for i, n in enumerate(sizes):
        speedup = btree_times[i] / rmi_times[i]
        memory_reduction = (1 - rmi_memory[i] / btree_memory[i]) * 100
        print(f"  n={n:,}: {speedup:.2f}x faster, {memory_reduction:.1f}% less memory")


if __name__ == "__main__":
    run_optimized_benchmark()
    analyze_performance_issues()
    create_performance_visualization()
