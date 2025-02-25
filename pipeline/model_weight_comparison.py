"""
Model Weight Comparison Visualizer

This script helps visualize the differences between two models (e.g., standard and reasoning-tuned models)
by comparing their weights and visualizing the differences using various techniques.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
import os
from pathlib import Path
import gc

# Create directory for saving visualizations
Path("visualizations").mkdir(exist_ok=True)

def load_models(model_path_original: str, model_path_reasoning: str) -> Tuple[HookedTransformer, HookedTransformer]:
    """
    Load both models using HookedTransformer.
    
    Args:
        model_path_original: Path to the original (non-reasoning) model
        model_path_reasoning: Path to the reasoning-tuned model
        
    Returns:
        Tuple of (original model, reasoning model)
    """
    print(f"Loading original model from {model_path_original}...")
    model_original = HookedTransformer.from_pretrained_no_processing(
        model_path_original,
        dtype=torch.bfloat16,
        device="cpu"  # Load on CPU first to avoid OOM
    )
    
    print(f"Loading reasoning model from {model_path_reasoning}...")
    model_reasoning = HookedTransformer.from_pretrained_no_processing(
        model_path_reasoning,
        dtype=torch.bfloat16,
        device="cpu"
    )
    
    return model_original, model_reasoning

def extract_model_weights(model: HookedTransformer) -> Dict[str, torch.Tensor]:
    """
    Extract relevant weights from a model for comparison.
    
    Args:
        model: HookedTransformer model
        
    Returns:
        Dictionary of weight tensors
    """
    weights = {}
    
    # Extract attention weights
    for layer_idx in range(model.cfg.n_layers):
        # Attention weights (query, key, value, output)
        weights[f"layer_{layer_idx}.attn.W_Q"] = model.blocks[layer_idx].attn.W_Q.float().cpu()
        weights[f"layer_{layer_idx}.attn.W_K"] = model.blocks[layer_idx].attn.W_K.float().cpu()
        weights[f"layer_{layer_idx}.attn.W_V"] = model.blocks[layer_idx].attn.W_V.float().cpu()
        weights[f"layer_{layer_idx}.attn.W_O"] = model.blocks[layer_idx].attn.W_O.float().cpu()
        
        # MLP weights
        weights[f"layer_{layer_idx}.mlp.W_in"] = model.blocks[layer_idx].mlp.W_in.float().cpu()
        weights[f"layer_{layer_idx}.mlp.W_out"] = model.blocks[layer_idx].mlp.W_out.float().cpu()
        
        # Layer norms
        weights[f"layer_{layer_idx}.ln1.w"] = model.blocks[layer_idx].ln1.w.float().cpu()
        weights[f"layer_{layer_idx}.ln2.w"] = model.blocks[layer_idx].ln2.w.float().cpu()
    
    return weights

def visualize_weight_histograms(
    weights_original: Dict[str, torch.Tensor], 
    weights_reasoning: Dict[str, torch.Tensor],
    components: List[str] = None,
    max_plots: int = 6
):
    """
    Create histograms comparing weights between models.
    
    Args:
        weights_original: Dictionary of original model weights
        weights_reasoning: Dictionary of reasoning model weights
        components: List of weight keys to visualize (None means all)
        max_plots: Maximum number of plots to create
    """
    if components is None:
        # Sample some representative components
        all_keys = list(weights_original.keys())
        components = []
        
        # Get attention components from different layers
        attn_keys = [k for k in all_keys if 'attn' in k]
        if attn_keys:
            components.extend(np.random.choice(attn_keys, min(max_plots // 2, len(attn_keys)), replace=False))
        
        # Get MLP components from different layers
        mlp_keys = [k for k in all_keys if 'mlp' in k]
        if mlp_keys:
            components.extend(np.random.choice(mlp_keys, min(max_plots // 2, len(mlp_keys)), replace=False))
    
    # Create histograms
    for i, component in enumerate(components[:max_plots]):
        if component not in weights_original or component not in weights_reasoning:
            continue
            
        # Get weights and flatten
        w_orig = weights_original[component].flatten().numpy()
        w_reason = weights_reasoning[component].flatten().numpy()
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Original weights histogram
        plt.subplot(1, 2, 1)
        plt.hist(w_orig, bins=50, alpha=0.7, color='blue')
        plt.title(f"Original Model: {component}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        
        # Reasoning weights histogram
        plt.subplot(1, 2, 2)
        plt.hist(w_reason, bins=50, alpha=0.7, color='green')
        plt.title(f"Reasoning Model: {component}")
        plt.xlabel("Weight Value")
        
        plt.tight_layout()
        plt.savefig(f"visualizations/hist_{component.replace('.', '_').replace('/', '_')}.png")
        plt.close()

def visualize_weight_differences(
    weights_original: Dict[str, torch.Tensor], 
    weights_reasoning: Dict[str, torch.Tensor],
    components: List[str] = None,
    max_plots: int = 6
):
    """
    Create difference histograms and heatmaps showing changes between models.
    
    Args:
        weights_original: Dictionary of original model weights
        weights_reasoning: Dictionary of reasoning model weights
        components: List of weight keys to visualize (None means all)
        max_plots: Maximum number of plots to create
    """
    if components is None:
        # Sample some representative components
        all_keys = list(weights_original.keys())
        components = []
        
        # Get attention components from different layers
        attn_keys = [k for k in all_keys if 'attn' in k]
        if attn_keys:
            components.extend(np.random.choice(attn_keys, min(max_plots // 2, len(attn_keys)), replace=False))
        
        # Get MLP components from different layers
        mlp_keys = [k for k in all_keys if 'mlp' in k]
        if mlp_keys:
            components.extend(np.random.choice(mlp_keys, min(max_plots // 2, len(mlp_keys)), replace=False))
    
    # Create visualizations
    for i, component in enumerate(components[:max_plots]):
        if component not in weights_original or component not in weights_reasoning:
            continue
            
        # Get weights and compute difference
        w_orig = weights_original[component]
        w_reason = weights_reasoning[component]
        
        if w_orig.shape != w_reason.shape:
            print(f"Shapes don't match for {component}: {w_orig.shape} vs {w_reason.shape}")
            continue
            
        w_diff = w_reason - w_orig
        w_diff_flat = w_diff.flatten().numpy()
        
        # Create figure for histogram
        plt.figure(figsize=(10, 6))
        plt.hist(w_diff_flat, bins=50, color='purple', alpha=0.7)
        plt.title(f"Weight Differences: {component}")
        plt.xlabel("Difference Value (Reasoning - Original)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.savefig(f"visualizations/diff_hist_{component.replace('.', '_').replace('/', '_')}.png")
        plt.close()
        
        # Create a heatmap for 2D visualization (sample if too large)
        plt.figure(figsize=(12, 10))
        
        if len(w_diff.shape) == 1:
            # For 1D tensors, reshape to 2D for heatmap
            w_diff_2d = w_diff.reshape(-1, 1)
        elif len(w_diff.shape) == 2:
            # For 2D tensors, use directly
            w_diff_2d = w_diff
        else:
            # For higher dimensions, flatten to 2D
            w_diff_2d = w_diff.reshape(w_diff.shape[0], -1)
        
        # Sample if too large
        if w_diff_2d.shape[0] > 200 or w_diff_2d.shape[1] > 200:
            sample_rows = min(w_diff_2d.shape[0], 200)
            sample_cols = min(w_diff_2d.shape[1], 200)
            row_indices = np.sort(np.random.choice(w_diff_2d.shape[0], sample_rows, replace=False))
            col_indices = np.sort(np.random.choice(w_diff_2d.shape[1], sample_cols, replace=False))
            w_diff_sample = w_diff_2d[row_indices][:, col_indices]
        else:
            w_diff_sample = w_diff_2d
            
        # Plot heatmap
        sns.heatmap(w_diff_sample.numpy(), cmap='coolwarm', center=0, 
                   vmin=-abs(w_diff_sample).max().item(), vmax=abs(w_diff_sample).max().item())
        plt.title(f"Weight Differences Heatmap: {component}")
        plt.tight_layout()
        plt.savefig(f"visualizations/diff_heatmap_{component.replace('.', '_').replace('/', '_')}.png")
        plt.close()

def layer_wise_comparison(
    weights_original: Dict[str, torch.Tensor], 
    weights_reasoning: Dict[str, torch.Tensor]
):
    """
    Create layer-wise comparison statistics for model weights.
    
    Args:
        weights_original: Dictionary of original model weights
        weights_reasoning: Dictionary of reasoning model weights
    """
    # Group weights by layer
    layer_stats = {}
    
    for key in weights_original:
        if key not in weights_reasoning:
            continue
            
        # Extract layer number
        if 'layer_' not in key:
            continue
            
        layer_num = int(key.split('.')[0].replace('layer_', ''))
        
        if layer_num not in layer_stats:
            layer_stats[layer_num] = {
                'l2_diff': 0,
                'max_diff': 0,
                'mean_diff': 0,
                'num_components': 0
            }
        
        # Calculate statistics
        w_orig = weights_original[key]
        w_reason = weights_reasoning[key]
        
        if w_orig.shape != w_reason.shape:
            continue
            
        w_diff = w_reason - w_orig
        
        # L2 norm of difference
        l2_diff = torch.norm(w_diff).item()
        
        # Max absolute difference
        max_diff = torch.max(torch.abs(w_diff)).item()
        
        # Mean absolute difference
        mean_diff = torch.mean(torch.abs(w_diff)).item()
        
        # Update stats
        layer_stats[layer_num]['l2_diff'] += l2_diff
        layer_stats[layer_num]['max_diff'] = max(layer_stats[layer_num]['max_diff'], max_diff)
        layer_stats[layer_num]['mean_diff'] += mean_diff
        layer_stats[layer_num]['num_components'] += 1
    
    # Calculate averages
    for layer in layer_stats:
        if layer_stats[layer]['num_components'] > 0:
            layer_stats[layer]['mean_diff'] /= layer_stats[layer]['num_components']
    
    # Plot layer-wise statistics
    layers = sorted(layer_stats.keys())
    
    # L2 norm differences
    plt.figure(figsize=(12, 6))
    plt.bar(layers, [layer_stats[l]['l2_diff'] for l in layers])
    plt.title("Layer-wise L2 Norm of Weight Differences")
    plt.xlabel("Layer")
    plt.ylabel("L2 Norm of Difference")
    plt.grid(alpha=0.3)
    plt.savefig("visualizations/layer_l2_diff.png")
    plt.close()
    
    # Mean absolute differences
    plt.figure(figsize=(12, 6))
    plt.bar(layers, [layer_stats[l]['mean_diff'] for l in layers])
    plt.title("Layer-wise Mean Absolute Weight Differences")
    plt.xlabel("Layer")
    plt.ylabel("Mean Absolute Difference")
    plt.grid(alpha=0.3)
    plt.savefig("visualizations/layer_mean_diff.png")
    plt.close()
    
    # Max absolute differences
    plt.figure(figsize=(12, 6))
    plt.bar(layers, [layer_stats[l]['max_diff'] for l in layers])
    plt.title("Layer-wise Maximum Absolute Weight Differences")
    plt.xlabel("Layer")
    plt.ylabel("Max Absolute Difference")
    plt.grid(alpha=0.3)
    plt.savefig("visualizations/layer_max_diff.png")
    plt.close()
    
    return layer_stats

def component_type_comparison(
    weights_original: Dict[str, torch.Tensor], 
    weights_reasoning: Dict[str, torch.Tensor]
):
    """
    Compare differences by component type (attention, MLP, etc.)
    
    Args:
        weights_original: Dictionary of original model weights
        weights_reasoning: Dictionary of reasoning model weights
    """
    # Group components by type
    component_types = {
        'attention_q': [],
        'attention_k': [],
        'attention_v': [],
        'attention_o': [],
        'mlp_in': [],
        'mlp_out': [],
        'layer_norm': []
    }
    
    # Collect keys by type
    for key in weights_original:
        if key not in weights_reasoning:
            continue
            
        if 'attn.W_Q' in key:
            component_types['attention_q'].append(key)
        elif 'attn.W_K' in key:
            component_types['attention_k'].append(key)
        elif 'attn.W_V' in key:
            component_types['attention_v'].append(key)
        elif 'attn.W_O' in key:
            component_types['attention_o'].append(key)
        elif 'mlp.W_in' in key:
            component_types['mlp_in'].append(key)
        elif 'mlp.W_out' in key:
            component_types['mlp_out'].append(key)
        elif 'ln' in key:
            component_types['layer_norm'].append(key)
    
    # Calculate statistics by type
    type_stats = {}
    
    for comp_type, keys in component_types.items():
        if not keys:
            continue
            
        diffs = []
        
        for key in keys:
            w_orig = weights_original[key]
            w_reason = weights_reasoning[key]
            
            if w_orig.shape != w_reason.shape:
                continue
                
            w_diff = w_reason - w_orig
            
            # Flatten and collect all differences for this type
            diffs.extend(w_diff.flatten().numpy().tolist())
        
        if diffs:
            type_stats[comp_type] = {
                'mean_abs_diff': np.mean(np.abs(diffs)),
                'median_abs_diff': np.median(np.abs(diffs)),
                'max_abs_diff': np.max(np.abs(diffs)),
                'std_diff': np.std(diffs)
            }
    
    # Plot statistics by component type
    comp_types = list(type_stats.keys())
    
    # Mean absolute difference by component type
    plt.figure(figsize=(14, 7))
    plt.bar(comp_types, [type_stats[t]['mean_abs_diff'] for t in comp_types])
    plt.title("Mean Absolute Difference by Component Type")
    plt.xlabel("Component Type")
    plt.ylabel("Mean Absolute Difference")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/component_type_mean_diff.png")
    plt.close()
    
    # Create boxplots for each component type
    plt.figure(figsize=(14, 7))
    
    all_diffs_by_type = []
    for comp_type in comp_types:
        diffs = []
        
        for key in component_types[comp_type]:
            if key in weights_original and key in weights_reasoning:
                w_orig = weights_original[key]
                w_reason = weights_reasoning[key]
                
                if w_orig.shape != w_reason.shape:
                    continue
                    
                w_diff = w_reason - w_orig
                
                # Sample to keep boxplot manageable
                flat_diff = w_diff.flatten().numpy()
                if len(flat_diff) > 10000:
                    indices = np.random.choice(len(flat_diff), 10000, replace=False)
                    flat_diff = flat_diff[indices]
                
                diffs.append(flat_diff)
        
        if diffs:
            all_diffs_by_type.append(np.concatenate(diffs))
            
    plt.boxplot(all_diffs_by_type, labels=comp_types)
    plt.title("Distribution of Weight Differences by Component Type")
    plt.xlabel("Component Type")
    plt.ylabel("Difference Value")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/component_type_boxplot.png")
    plt.close()
    
    return type_stats

def compare_model_weights(
    model_path_original: str, 
    model_path_reasoning: str,
    visualize: bool = True
):
    """
    Main function to compare weights between two models.
    
    Args:
        model_path_original: Path to the original model
        model_path_reasoning: Path to the reasoning model
        visualize: Whether to create visualizations
        
    Returns:
        Dictionary with comparison statistics
    """
    # Load models
    model_original, model_reasoning = load_models(model_path_original, model_path_reasoning)
    
    # Extract weights
    print("Extracting weights from original model...")
    weights_original = extract_model_weights(model_original)
    
    print("Extracting weights from reasoning model...")
    weights_reasoning = extract_model_weights(model_reasoning)
    
    # Clean up models to save memory
    del model_original, model_reasoning
    gc.collect()
    torch.cuda.empty_cache()
    
    # Statistics and results to return
    results = {
        "models": {
            "original": model_path_original,
            "reasoning": model_path_reasoning
        }
    }
    
    if visualize:
        print("Creating weight histograms...")
        visualize_weight_histograms(weights_original, weights_reasoning)
        
        print("Visualizing weight differences...")
        visualize_weight_differences(weights_original, weights_reasoning)
        
        print("Analyzing layer-wise differences...")
        layer_stats = layer_wise_comparison(weights_original, weights_reasoning)
        results["layer_stats"] = layer_stats
        
        print("Analyzing component type differences...")
        type_stats = component_type_comparison(weights_original, weights_reasoning)
        results["component_type_stats"] = type_stats
    
    print(f"Visualization results saved to 'visualizations/' directory")
    return results

if __name__ == "__main__":
    # Example usage
    MODEL_PATH_ORIGINAL = "meta-llama/Llama-3.1-8B-Instruct"  # Non-reasoning model
    MODEL_PATH_REASONING = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Reasoning model
    
    results = compare_model_weights(MODEL_PATH_ORIGINAL, MODEL_PATH_REASONING) 