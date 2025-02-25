"""
Activation Comparison Visualizer

This script compares activations between two models (original and reasoning-tuned)
when processing the same inputs, helping to understand differences in their internal processing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer, utils
from typing import Dict, List, Tuple, Optional, Union, Callable
from tqdm import tqdm
from pathlib import Path
import gc
from sklearn.decomposition import PCA
from datasets import load_dataset

# Create directory for saving visualizations
Path("visualizations/activations").mkdir(exist_ok=True, parents=True)

def prepare_prompts(problems: List[str], n_problems: int = 10) -> List[str]:
    """Prepare math problem prompts."""
    return [f"{problem}\n\nPlease provide your answer first, then your reasoning." 
            for problem in problems[:n_problems]]

def load_models_and_data(
    model_path_original: str, 
    model_path_reasoning: str,
    n_problems: int = 10
) -> Tuple[HookedTransformer, HookedTransformer, List[str]]:
    """
    Load both models and prepare example data.
    
    Args:
        model_path_original: Path to the original model
        model_path_reasoning: Path to the reasoning model
        n_problems: Number of math problems to use
        
    Returns:
        Tuple of (original model, reasoning model, prompts)
    """
    # Load models
    print(f"Loading original model from {model_path_original}...")
    model_original = HookedTransformer.from_pretrained_no_processing(
        model_path_original, 
        dtype=torch.bfloat16,
        default_padding_side='left'
    )
    model_original.tokenizer.padding_side = 'left'
    model_original.tokenizer.pad_token = model_original.tokenizer.eos_token
    
    print(f"Loading reasoning model from {model_path_reasoning}...")
    model_reasoning = HookedTransformer.from_pretrained_no_processing(
        model_path_reasoning,
        dtype=torch.bfloat16,
        default_padding_side='left'
    )
    model_reasoning.tokenizer.padding_side = 'left'
    model_reasoning.tokenizer.pad_token = model_reasoning.tokenizer.eos_token
    
    # Load GSM8K dataset for math problems
    print("Loading GSM8K dataset for test problems...")
    gsm8k = load_dataset("gsm8k", "main")
    problems = [item["question"] for item in gsm8k["train"]]
    
    # Prepare prompts
    prompts = prepare_prompts(problems, n_problems)
    
    return model_original, model_reasoning, prompts

def tokenize_inputs(
    model: HookedTransformer, 
    prompts: List[str],
    chat_template: str
) -> torch.Tensor:
    """
    Tokenize inputs using the model's tokenizer.
    
    Args:
        model: The model to use for tokenization
        prompts: List of prompts to tokenize
        chat_template: Chat template to format prompts
        
    Returns:
        Tokenized inputs tensor
    """
    formatted_prompts = [chat_template.format(instruction=prompt) for prompt in prompts]
    tokens = model.tokenizer(formatted_prompts, padding=True, return_tensors="pt").input_ids
    return tokens

def collect_model_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    activation_names: List[str] = None,
    batch_size: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Collect activations from a model.
    
    Args:
        model: The model to collect activations from
        tokens: Tokenized inputs
        activation_names: List of activation names to collect (None means all residuals)
        batch_size: Batch size for processing
        
    Returns:
        Dictionary of activations
    """
    if activation_names is None:
        activation_names = [f"blocks.{l}.attn.hook_result" for l in range(model.cfg.n_layers)]
        activation_names += [f"blocks.{l}.mlp.hook_result" for l in range(model.cfg.n_layers)]
        activation_names += [f"blocks.{l}.hook_resid_pre" for l in range(model.cfg.n_layers)]
        activation_names += [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]
    
    all_activations = {}
    
    for i in tqdm(range(0, len(tokens), batch_size), desc="Collecting activations"):
        batch = tokens[i:i+batch_size].to(model.cfg.device)
        
        # Run the model and collect activations
        _, cache = model.run_with_cache(
            batch,
            names_filter=lambda name: any(act_name in name for act_name in activation_names)
        )
        
        # First batch, initialize the dictionary
        if not all_activations:
            all_activations = {name: [tensor.cpu()] for name, tensor in cache.items()}
        else:
            # Append to existing activations
            for name, tensor in cache.items():
                if name in all_activations:
                    all_activations[name].append(tensor.cpu())
        
        # Clear memory
        del cache
        torch.cuda.empty_cache()
        gc.collect()
    
    # Concatenate all batches
    all_activations = {name: torch.cat(tensors, dim=0) for name, tensors in all_activations.items()}
    
    return all_activations

def visualize_activation_differences(
    original_activations: Dict[str, torch.Tensor],
    reasoning_activations: Dict[str, torch.Tensor],
    key_acts: List[str] = None,
    max_plots: int = 10
):
    """
    Visualize differences between activations from the two models.
    
    Args:
        original_activations: Activations from the original model
        reasoning_activations: Activations from the reasoning model
        key_acts: List of activation keys to visualize (None means sample some)
        max_plots: Maximum number of plots to create
    """
    if key_acts is None:
        # Sample some keys to visualize
        common_keys = set(original_activations.keys()).intersection(set(reasoning_activations.keys()))
        key_acts = list(common_keys)
        
        if len(key_acts) > max_plots:
            key_acts = np.random.choice(key_acts, max_plots, replace=False).tolist()
    
    # Create visualizations for each key
    for key in key_acts:
        if key not in original_activations or key not in reasoning_activations:
            continue
            
        orig_act = original_activations[key]
        reason_act = reasoning_activations[key]
        
        # Make sure shapes match - select last token if needed
        if orig_act.shape != reason_act.shape:
            # Take the last token's activations if shapes differ
            orig_act = orig_act[:, -1, :]
            reason_act = reason_act[:, -1, :]
        
        # Calculate differences and l2 norms
        act_diff = reason_act - orig_act
        
        # Check dimensions and flatten appropriately
        if len(act_diff.shape) == 3:  # [batch, seq_len, hidden_dim]
            # Average across batch and sequence dimension
            act_diff_flat = act_diff.mean(dim=[0, 1]).numpy()
            l2_norms = torch.norm(act_diff, dim=2).mean(dim=1).numpy()
        else:  # [batch, hidden_dim]
            # Average across batch dimension
            act_diff_flat = act_diff.mean(dim=0).numpy()
            l2_norms = torch.norm(act_diff, dim=1).numpy()
        
        # Histogram of differences
        plt.figure(figsize=(10, 6))
        plt.hist(act_diff_flat, bins=50, alpha=0.7, color='purple')
        plt.title(f"Activation Differences: {key}")
        plt.xlabel("Difference Value (Reasoning - Original)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.savefig(f"visualizations/activations/diff_hist_{key.replace('.', '_')}.png")
        plt.close()
        
        # L2 norms of differences across batch
        plt.figure(figsize=(10, 6))
        plt.hist(l2_norms, bins=30, alpha=0.7, color='teal')
        plt.title(f"L2 Norms of Activation Differences: {key}")
        plt.xlabel("L2 Norm")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.savefig(f"visualizations/activations/l2_norm_{key.replace('.', '_')}.png")
        plt.close()

def analyze_layer_patterns(
    original_activations: Dict[str, torch.Tensor],
    reasoning_activations: Dict[str, torch.Tensor],
    n_layers: int
):
    """
    Analyze patterns across layers.
    
    Args:
        original_activations: Activations from the original model
        reasoning_activations: Activations from the reasoning model
        n_layers: Number of layers in the models
    """
    # Dict to store layer-wise statistics
    layer_stats = {
        'attn_l2': np.zeros(n_layers),
        'mlp_l2': np.zeros(n_layers),
        'resid_pre_l2': np.zeros(n_layers),
        'resid_post_l2': np.zeros(n_layers)
    }
    
    # Collect statistics for each layer
    for layer in range(n_layers):
        # Attention
        attn_key = f"blocks.{layer}.attn.hook_result"
        if attn_key in original_activations and attn_key in reasoning_activations:
            orig_act = original_activations[attn_key]
            reason_act = reasoning_activations[attn_key]
            
            if orig_act.shape == reason_act.shape:
                act_diff = reason_act - orig_act
                layer_stats['attn_l2'][layer] = torch.norm(act_diff).item() / act_diff.numel()
        
        # MLP
        mlp_key = f"blocks.{layer}.mlp.hook_result"
        if mlp_key in original_activations and mlp_key in reasoning_activations:
            orig_act = original_activations[mlp_key]
            reason_act = reasoning_activations[mlp_key]
            
            if orig_act.shape == reason_act.shape:
                act_diff = reason_act - orig_act
                layer_stats['mlp_l2'][layer] = torch.norm(act_diff).item() / act_diff.numel()
        
        # Residual pre
        resid_pre_key = f"blocks.{layer}.hook_resid_pre"
        if resid_pre_key in original_activations and resid_pre_key in reasoning_activations:
            orig_act = original_activations[resid_pre_key]
            reason_act = reasoning_activations[resid_pre_key]
            
            if orig_act.shape == reason_act.shape:
                act_diff = reason_act - orig_act
                layer_stats['resid_pre_l2'][layer] = torch.norm(act_diff).item() / act_diff.numel()
        
        # Residual post
        resid_post_key = f"blocks.{layer}.hook_resid_post"
        if resid_post_key in original_activations and resid_post_key in reasoning_activations:
            orig_act = original_activations[resid_post_key]
            reason_act = reasoning_activations[resid_post_key]
            
            if orig_act.shape == reason_act.shape:
                act_diff = reason_act - orig_act
                layer_stats['resid_post_l2'][layer] = torch.norm(act_diff).item() / act_diff.numel()
    
    # Plot layer-wise statistics
    plt.figure(figsize=(12, 8))
    layers = list(range(n_layers))
    
    plt.plot(layers, layer_stats['attn_l2'], 'o-', label='Attention')
    plt.plot(layers, layer_stats['mlp_l2'], 's-', label='MLP')
    plt.plot(layers, layer_stats['resid_pre_l2'], '^-', label='Residual Pre')
    plt.plot(layers, layer_stats['resid_post_l2'], 'D-', label='Residual Post')
    
    plt.title("Layer-wise Activation Differences")
    plt.xlabel("Layer")
    plt.ylabel("Average L2 Norm of Difference")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("visualizations/activations/layer_pattern.png")
    plt.close()
    
    return layer_stats

def pca_analysis(
    original_activations: Dict[str, torch.Tensor],
    reasoning_activations: Dict[str, torch.Tensor],
    layer_indices: List[int] = None
):
    """
    Perform PCA analysis on activation differences.
    
    Args:
        original_activations: Activations from the original model
        reasoning_activations: Activations from the reasoning model
        layer_indices: Indices of layers to analyze (None means all)
    """
    if layer_indices is None:
        # Find how many layers we have
        layer_keys = [k for k in original_activations.keys() if "hook_resid_post" in k]
        n_layers = len(layer_keys)
        layer_indices = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]  # Sample early, middle, late layers
    
    for layer in layer_indices:
        # Residual post activations
        key = f"blocks.{layer}.hook_resid_post"
        
        if key not in original_activations or key not in reasoning_activations:
            continue
            
        orig_act = original_activations[key]
        reason_act = reasoning_activations[key]
        
        if orig_act.shape != reason_act.shape:
            continue
        
        # Take mean across sequence dimension if it exists
        if len(orig_act.shape) == 3:  # [batch, seq_len, hidden_dim]
            orig_act = orig_act.mean(dim=1)
            reason_act = reason_act.mean(dim=1)
        
        # Calculate differences
        act_diff = reason_act - orig_act
        
        # Perform PCA
        pca = PCA(n_components=2)
        
        # Concatenate both original and reasoning activations
        combined = torch.cat([orig_act, reason_act], dim=0).numpy()
        pca_result = pca.fit_transform(combined)
        
        # Split back into original and reasoning
        n_samples = orig_act.shape[0]
        pca_original = pca_result[:n_samples]
        pca_reasoning = pca_result[n_samples:]
        
        # Plot PCA
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_original[:, 0], pca_original[:, 1], alpha=0.7, label="Original Model", color="blue")
        plt.scatter(pca_reasoning[:, 0], pca_reasoning[:, 1], alpha=0.7, label="Reasoning Model", color="green")
        
        # Connect corresponding points with lines
        for i in range(n_samples):
            plt.plot([pca_original[i, 0], pca_reasoning[i, 0]], 
                     [pca_original[i, 1], pca_reasoning[i, 1]], 
                     'k-', alpha=0.2)
        
        plt.title(f"PCA of Activations at Layer {layer}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"visualizations/activations/pca_layer_{layer}.png")
        plt.close()

def compare_model_activations(
    model_path_original: str, 
    model_path_reasoning: str,
    chat_template_original: str = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    chat_template_reasoning: str = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    n_problems: int = 10,
    batch_size: int = 4
):
    """
    Main function to compare activations between models.
    
    Args:
        model_path_original: Path to the original model
        model_path_reasoning: Path to the reasoning model
        chat_template_original: Chat template for the original model
        chat_template_reasoning: Chat template for the reasoning model
        n_problems: Number of problems to analyze
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with comparison statistics
    """
    # Load models and data
    model_original, model_reasoning, prompts = load_models_and_data(
        model_path_original, model_path_reasoning, n_problems
    )
    
    # Tokenize inputs
    print("Tokenizing inputs for original model...")
    tokens_original = tokenize_inputs(model_original, prompts, chat_template_original)
    
    print("Tokenizing inputs for reasoning model...")
    tokens_reasoning = tokenize_inputs(model_reasoning, prompts, chat_template_reasoning)
    
    # Collect activations
    print("Collecting activations from original model...")
    original_activations = collect_model_activations(model_original, tokens_original, batch_size=batch_size)
    
    print("Collecting activations from reasoning model...")
    reasoning_activations = collect_model_activations(model_reasoning, tokens_reasoning, batch_size=batch_size)
    
    # Clean up models to save memory
    del model_original, model_reasoning
    gc.collect()
    torch.cuda.empty_cache()
    
    # Visualize activation differences
    print("Visualizing activation differences...")
    visualize_activation_differences(original_activations, reasoning_activations)
    
    # Analyze layer patterns
    print("Analyzing layer patterns...")
    n_layers = len([k for k in original_activations.keys() if "hook_resid_post" in k])
    layer_stats = analyze_layer_patterns(original_activations, reasoning_activations, n_layers)
    
    # PCA analysis
    print("Performing PCA analysis...")
    pca_analysis(original_activations, reasoning_activations)
    
    print(f"Visualization results saved to 'visualizations/activations/' directory")
    
    return {
        "models": {
            "original": model_path_original,
            "reasoning": model_path_reasoning
        },
        "layer_stats": layer_stats
    }

if __name__ == "__main__":
    # Example usage
    MODEL_PATH_ORIGINAL = "meta-llama/Llama-3.1-8B-Instruct"  # Non-reasoning model
    MODEL_PATH_REASONING = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Reasoning model
    
    # Adjust this based on your model chat templates
    ORIGINAL_CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    REASONING_CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    results = compare_model_activations(
        MODEL_PATH_ORIGINAL, 
        MODEL_PATH_REASONING,
        ORIGINAL_CHAT_TEMPLATE,
        REASONING_CHAT_TEMPLATE,
        n_problems=5  # Start with a small number for testing
    ) 