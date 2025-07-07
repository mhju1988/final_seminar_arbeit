import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import time
import sys
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# For B+ tree, we'll use BTrees library (install with: pip install BTrees)
try:
    from BTrees.OOBTree import OOBTree
except ImportError:
    print("Please install BTrees: pip install BTrees")
    sys.exit(1)

class RecursiveModelIndex:
    """
    Recursive Model Index (RMI) implementation based on the paper
    "The Case for Learned Index Structures"
    """
    
    def __init__(self, num_stages=2, stage_sizes=[1, 100], model_type='linear'):
        """
        Initialize RMI with configurable stages and model types
        
        Args:
            num_stages: Number of stages in the hierarchy
            stage_sizes: Number of models at each stage
            model_type: 'linear' or 'nn' for neural network
        """
        self.num_stages = num_stages
        self.stage_sizes = stage_sizes
        self.model_type = model_type
        self.models = [[] for _ in range(num_stages)]
        self.min_errors = [[] for _ in range(num_stages)]
        self.max_errors = [[] for _ in range(num_stages)]
        self.data = None
        self.n = 0
        
    def _create_model(self):
        """Create a model based on the specified type"""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'nn':
            # Small neural network as used in the paper
            return MLPRegressor(hidden_layer_sizes=(32, 16), 
                              activation='relu',
                              max_iter=1000,
                              random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, keys, positions=None):
        """
        Train the RMI on sorted keys
        
        Args:
            keys: Sorted array of keys
            positions: Optional positions (defaults to range(len(keys)))
        """
        self.data = np.array(keys)
        self.n = len(keys)
        
        if positions is None:
            positions = np.arange(self.n)
        
        # Stage-wise training
        tmp_records = [[]]
        tmp_records[0] = [(keys[i], positions[i]) for i in range(self.n)]
        
        for stage in range(self.num_stages):
            num_models = self.stage_sizes[stage]
            self.models[stage] = []
            self.min_errors[stage] = []
            self.max_errors[stage] = []
            
            if stage < self.num_stages - 1:
                next_tmp_records = [[] for _ in range(self.stage_sizes[stage + 1])]
            
            for model_idx in range(num_models):
                if len(tmp_records[model_idx]) == 0:
                    self.models[stage].append(None)
                    self.min_errors[stage].append(0)
                    self.max_errors[stage].append(0)
                    continue
                
                # Extract keys and positions for this model
                model_keys = np.array([r[0] for r in tmp_records[model_idx]]).reshape(-1, 1)
                model_positions = np.array([r[1] for r in tmp_records[model_idx]])
                
                # Train model
                model = self._create_model()
                model.fit(model_keys, model_positions)
                self.models[stage].append(model)
                
                # Calculate min/max errors for last stage
                if stage == self.num_stages - 1:
                    predictions = model.predict(model_keys)
                    errors = predictions - model_positions
                    self.min_errors[stage].append(int(np.min(errors)))
                    self.max_errors[stage].append(int(np.max(errors)))
                else:
                    self.min_errors[stage].append(0)
                    self.max_errors[stage].append(0)
                
                # Distribute records to next stage models
                if stage < self.num_stages - 1:
                    for key, pos in tmp_records[model_idx]:
                        pred = model.predict([[key]])[0]
                        next_model_idx = int(pred / self.n * self.stage_sizes[stage + 1])
                        next_model_idx = max(0, min(next_model_idx, self.stage_sizes[stage + 1] - 1))
                        next_tmp_records[next_model_idx].append((key, pos))
            
            if stage < self.num_stages - 1:
                tmp_records = next_tmp_records
    
    def predict(self, key):
        """Predict the position of a key"""
        # Start from the root model
        model_idx = 0
        
        for stage in range(self.num_stages):
            model = self.models[stage][model_idx]
            if model is None:
                return 0
            
            pred = model.predict([[key]])[0]
            
            if stage < self.num_stages - 1:
                # Use prediction to select next model
                model_idx = int(pred / self.n * self.stage_sizes[stage + 1])
                model_idx = max(0, min(model_idx, self.stage_sizes[stage + 1] - 1))
            else:
                # Return final prediction with bounds
                min_err = self.min_errors[stage][model_idx]
                max_err = self.max_errors[stage][model_idx]
                return int(pred), int(pred + min_err), int(pred + max_err)
        
        return 0, 0, 0
    
    def lookup(self, key):
        """Look up a key and return its position"""
        pred, min_pos, max_pos = self.predict(key)
        
        # Binary search in the predicted range
        min_pos = max(0, min_pos)
        max_pos = min(self.n - 1, max_pos)
        
        while min_pos <= max_pos:
            mid = (min_pos + max_pos) // 2
            if self.data[mid] == key:
                return mid
            elif self.data[mid] < key:
                min_pos = mid + 1
            else:
                max_pos = mid - 1
        
        return -1  # Key not found
    
    def memory_size_mb(self):
        """Estimate memory size in MB"""
        # Rough estimation: 
        # Linear model: ~16 bytes per coefficient
        # NN model: ~4 bytes per weight
        total_size = 0
        
        for stage in range(self.num_stages):
            for model in self.models[stage]:
                if model is None:
                    continue
                    
                if self.model_type == 'linear':
                    # Linear regression: 2 parameters (slope, intercept)
                    total_size += 2 * 8  # 8 bytes per float
                elif self.model_type == 'nn':
                    # Estimate NN size based on architecture
                    total_size += (1 * 32 + 32 * 16 + 16 * 1 + 32 + 16 + 1) * 4
        
        # Add error bounds storage
        total_size += sum(len(errors) * 4 * 2 for errors in self.min_errors)
        
        return total_size / (1024 * 1024)


class BPlusTreeWrapper:
    """Wrapper around BTree for consistent interface"""
    
    def __init__(self):
        self.tree = OOBTree()
        self.data = None
        
    def train(self, keys, positions=None):
        """Build B+ tree from sorted keys"""
        self.data = np.array(keys)
        if positions is None:
            positions = np.arange(len(keys))
            
        for i, key in enumerate(keys):
            self.tree[key] = positions[i]
    
    def lookup(self, key):
        """Look up a key in the B+ tree"""
        if key in self.tree:
            return self.tree[key]
        return -1
    
    def memory_size_mb(self):
        """Estimate memory size of B+ tree in MB"""
        # Rough estimation: each node ~100 bytes
        num_nodes = len(self.tree) // 100  # Approximate number of nodes
        return (num_nodes * 100) / (1024 * 1024)


def generate_datasets(n=1000000):
    """Generate synthetic datasets similar to the paper"""
    datasets = {}
    
    # Lognormal distribution
    np.random.seed(42)
    lognormal = np.sort(np.random.lognormal(0, 2, n))
    datasets['Lognormal'] = lognormal
    
    # Uniform distribution (simulating Maps dataset)
    uniform = np.sort(np.random.uniform(0, 1000000, n))
    datasets['Uniform'] = uniform
    
    # Non-uniform with patterns (simulating Weblogs)
    # Mix of uniform and concentrated regions
    weblogs = []
    for i in range(10):
        # Create clusters
        center = i * 100000
        cluster = np.random.normal(center, 1000, n // 10)
        weblogs.extend(cluster)
    weblogs = np.sort(np.array(weblogs))[:n]
    datasets['Weblogs'] = weblogs
    
    return datasets


def benchmark_index(index, keys, num_lookups=100):
    """Benchmark an index structure"""
    # Random lookup keys
    lookup_keys = np.random.choice(keys, num_lookups)
    
    # Measure lookup time
    start_time = time.time()
    for key in lookup_keys:
        _ = index.lookup(key)
    end_time = time.time()
    
    avg_lookup_time_us = (end_time - start_time) / num_lookups * 1e6
    memory_mb = index.memory_size_mb()
    
    return {
        'avg_lookup_time_us': avg_lookup_time_us,
        'memory_mb': memory_mb
    }


def compare_indexes(datasets, stage_sizes_list=[[1, 300],   [1, 500], [1, 1000]]):
    """Compare RMI with different configurations against B+ tree"""
    results = []
    
    for dataset_name, keys in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        
        # B+ Tree baseline
        print("Training B+ Tree...")
        btree = BPlusTreeWrapper()
        btree.train(keys)
        btree_results = benchmark_index(btree, keys)
        
        results.append({
            'Dataset': dataset_name,
            'Index': 'B+ Tree',
            'Config': 'Default',
            'Lookup Time (μs)': btree_results['avg_lookup_time_us'],
            'Memory (MB)': btree_results['memory_mb']
        })
        
        # RMI with different configurations
        for stage_sizes in stage_sizes_list:
            for model_type in ['linear', 'nn']:
                config_name = f"stage_sizes-{model_type}"
                print(f"Training RMI {config_name}...")
                
                rmi = RecursiveModelIndex(
                    num_stages=2,
                    stage_sizes=stage_sizes,
                    model_type=model_type
                )
                rmi.train(keys)
                rmi_results = benchmark_index(rmi, keys)
                
                results.append({
                    'Dataset': dataset_name,
                    'Index': 'RMI',
                    'Config': config_name,
                    'Lookup Time (μs)': rmi_results['avg_lookup_time_us'],
                    'Memory (MB)': rmi_results['memory_mb']
                })
    
    return pd.DataFrame(results)


def visualize_results(results_df):
    """Create visualizations similar to the paper"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Lookup time comparison
    ax = axes[0, 0]
    pivot_time = results_df.pivot_table(
        values='Lookup Time (μs)', 
        index='Config', 
        columns='Dataset'
    )
    pivot_time.plot(kind='bar', ax=ax)
    ax.set_title('Average Lookup Time Comparison')
    ax.set_ylabel('Lookup Time (μs)')
    ax.legend(title='Dataset')
    ax.grid(True, alpha=0.3)
    
    # 2. Memory usage comparison
    ax = axes[0, 1]
    pivot_memory = results_df.pivot_table(
        values='Memory (MB)', 
        index='Config', 
        columns='Dataset'
    )
    pivot_memory.plot(kind='bar', ax=ax)
    ax.set_title('Memory Usage Comparison')
    ax.set_ylabel('Memory (MB)')
    ax.legend(title='Dataset')
    ax.grid(True, alpha=0.3)
    
    # 3. Speedup over B+ Tree
    ax = axes[1, 0]
    for dataset in results_df['Dataset'].unique():
        dataset_df = results_df[results_df['Dataset'] == dataset]
        btree_time = dataset_df[dataset_df['Index'] == 'B+ Tree']['Lookup Time (μs)'].values[0]
        rmi_df = dataset_df[dataset_df['Index'] == 'RMI'].copy()
        rmi_df['Speedup'] = btree_time / rmi_df['Lookup Time (μs)']
        
        ax.plot(rmi_df['Config'], rmi_df['Speedup'], marker='o', label=dataset)
    
    ax.set_title('Speedup over B+ Tree')
    ax.set_xlabel('RMI Configuration')
    ax.set_ylabel('Speedup Factor')
    ax.legend(title='Dataset')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    # 4. Memory savings
    ax = axes[1, 1]
    for dataset in results_df['Dataset'].unique():
        dataset_df = results_df[results_df['Dataset'] == dataset]
        btree_memory = dataset_df[dataset_df['Index'] == 'B+ Tree']['Memory (MB)'].values[0]
        rmi_df = dataset_df[dataset_df['Index'] == 'RMI'].copy()
        rmi_df['Memory Reduction'] = (1 - rmi_df['Memory (MB)'] / btree_memory) * 100
        
        ax.plot(rmi_df['Config'], rmi_df['Memory Reduction'], marker='o', label=dataset)
    
    ax.set_title('Memory Reduction vs B+ Tree')
    ax.set_xlabel('RMI Configuration')
    ax.set_ylabel('Memory Reduction (%)')
    ax.legend(title='Dataset')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def create_detailed_comparison_table(results_df):
    """Create a detailed comparison table similar to the paper"""
    # Calculate additional metrics
    enhanced_results = []
    
    for dataset in results_df['Dataset'].unique():
        dataset_df = results_df[results_df['Dataset'] == dataset]
        btree_row = dataset_df[dataset_df['Index'] == 'B+ Tree'].iloc[0]
        
        for _, row in dataset_df.iterrows():
            enhanced_row = row.to_dict()
            
            if row['Index'] == 'RMI':
                # Calculate relative metrics
                enhanced_row['Speedup'] = btree_row['Lookup Time (μs)'] / row['Lookup Time (μs)']
                enhanced_row['Memory Savings'] = (1 - row['Memory (MB)'] / btree_row['Memory (MB)']) * 100
            else:
                enhanced_row['Speedup'] = 1.0
                enhanced_row['Memory Savings'] = 0.0
            
            enhanced_results.append(enhanced_row)
    
    enhanced_df = pd.DataFrame(enhanced_results)
    
    # Format for display
    display_df = enhanced_df.copy()
    display_df['Lookup Time (μs)'] = display_df['Lookup Time (μs)'].round(2)
    display_df['Memory (MB)'] = display_df['Memory (MB)'].round(2)
    display_df['Speedup'] = display_df['Speedup'].round(2)
    display_df['Memory Savings'] = display_df['Memory Savings'].round(1).astype(str) + '%'
    
    return display_df


def main():
    """Main function to run the experiments"""
    print("Generating synthetic datasets...")
    datasets = generate_datasets(n=200000)  # Using smaller size for demo
    
    print("\nDataset statistics:")
    for name, data in datasets.items():
        print(f"{name}: {len(data)} keys, range [{data.min():.2f}, {data.max():.2f}]")
    
    print("\nRunning benchmarks...")
    results_df = compare_indexes(datasets)
    
    print("\nResults Summary:")
    print("=" * 80)
    
    # Display detailed comparison table
    detailed_df = create_detailed_comparison_table(results_df)
    print(detailed_df.to_string(index=False))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_results(results_df)
    
    # Additional analysis
    print("\nKey Findings:")
    print("-" * 40)
    
    # Find best configurations
    for dataset in results_df['Dataset'].unique():
        dataset_df = results_df[results_df['Dataset'] == dataset]
        
        # Fastest lookup
        fastest = dataset_df.loc[dataset_df['Lookup Time (μs)'].idxmin()]
        print(f"\n{dataset} - Fastest lookup: {fastest['Config']} ({fastest['Lookup Time (μs)']:.2f} μs)")
        
        # Most memory efficient
        smallest = dataset_df.loc[dataset_df['Memory (MB)'].idxmin()]
        print(f"{dataset} - Smallest memory: {smallest['Config']} ({smallest['Memory (MB)']:.2f} MB)")
        
        # Best RMI configuration
        rmi_df = dataset_df[dataset_df['Index'] == 'RMI']
        if not rmi_df.empty:
            best_rmi = rmi_df.loc[rmi_df['Lookup Time (μs)'].idxmin()]
            btree_time = dataset_df[dataset_df['Index'] == 'B+ Tree']['Lookup Time (μs)'].values[0]
            speedup = btree_time / best_rmi['Lookup Time (μs)']
            print(f"{dataset} - Best RMI: {best_rmi['Config']} (speedup: {speedup:.2f}x)")


if __name__ == "__main__":
    main()
