# # Reasoning Direction
#
# This notebook aims to estimate the "reasoning" direction within the LLM activation space.
# We're basing this approach on the methodology used to find the "refusal" direction, but with a key difference:
#
# - **Refusal paper approach**: Used 1 LLM with 2 types of prompts (harmful vs harmless)
# - **Our approach**: Use 2 models (original vs reasoning-tuned) with the same prompts (GSM8K math problems)
#
# We'll collect activations from both models, calculate the difference (reasoning direction),
# and then test if adding this direction to the non-reasoning model enhances its reasoning capabilities.

# Setup and imports
# !pip install transformers transformers_stream_generator tiktoken transformer_lens einops jaxtyping colorama scikit-learn datasets

# !pip uninstall numpy -y

# +
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc
import transformers
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable, Dict, Tuple, Optional, Union
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float, Int
from colorama import Fore
from huggingface_hub import snapshot_download

# We turn off automatic differentiation to save GPU memory
torch.set_grad_enabled(False)
# -

# ## Load models
#
# We'll load both the original model and the reasoning-tuned model using HookedTransformer.
# If using a HuggingFace model, we can download it directly. If using a local model, make sure it's 
# in the correct directory structure.

# Define model paths - adjust these based on your models
MODEL_PATH_ORIGINAL = "meta-llama/Llama-3.1-8B-Instruct"  # Non-reasoning model
MODEL_PATH_REASONING = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Reasoning model

# ### Load both models

# +
# Login to HuggingFace
from huggingface_hub import login

# Replace with your HF token from https://huggingface.co/settings/tokens
login(token="") 

# +
# Load the original (non-reasoning) model
model_original = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH_ORIGINAL,
    dtype=torch.bfloat16,
    default_padding_side='left'
)
model_original.tokenizer.padding_side = 'left'
model_original.tokenizer.pad_token = model_original.tokenizer.eos_token

print(f"Loaded non-reasoning model {MODEL_PATH_ORIGINAL}")
# -

# free cuda memory
torch.cuda.empty_cache()
gc.collect()


# #### Optional: Download the reasoning model if needed
# This step can be skipped if you already have the models locally

# Uncomment this to download the reasoning model
model_path = snapshot_download(
    repo_id=MODEL_PATH_REASONING,
    local_dir=MODEL_PATH_ORIGINAL,
    local_dir_use_symlinks=False
)

# +
# Load the reasoning model
model_reasoning = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH_ORIGINAL,
    local_files_only=True,  # Set to True if using local models
    dtype=torch.bfloat16,
    default_padding_side='left'
)
model_reasoning.tokenizer.padding_side = 'left'
model_reasoning.tokenizer.pad_token = model_reasoning.tokenizer.eos_token

print(f"Loaded reasoning model {MODEL_PATH_REASONING}")

# +
# Test both models with a simple math problem
test_prompt = "What is 5*125?"

# Test original model
print("Original model response:")
output_original = model_original.generate(
    test_prompt, 
    temperature=0.0,
    max_new_tokens=100  # Increase max tokens to generate longer response
)
print(output_original)

print("\nReasoning model response:") 
output_reasoning = model_reasoning.generate(
    test_prompt,
    temperature=0.0,
    max_new_tokens=100  # Increase max tokens to generate longer response
)
print(output_reasoning)

# -

# ## Set up chat templates and data processing functions
#
# We need to define chat templates for both models and create functions to process our data.
# Different models may have different chat templates, so we adjust accordingly.

# +
# Define chat templates for both models
# Adjust these templates based on your specific models
ORIGINAL_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""  # Llama-3 template

REASONING_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""  # DeepSeek template

# +
# Define utility functions for processing data and collecting activations

def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    chat_template: str
) -> Int[Tensor, 'batch_size seq_len']:
    """Tokenize instructions using the specified chat template."""
    prompts = [chat_template.format(instruction=instruction) for instruction in instructions]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

def collect_activations(
    model: HookedTransformer,
    tokenized_inputs: Int[Tensor, 'batch_size seq_len'],
    batch_size: int = 8
) -> Dict[str, Tensor]:
    """Collect activations from a model for the given inputs."""
    activations = {}
    
    for i in tqdm(range(0, len(tokenized_inputs), batch_size)):
        batch = tokenized_inputs[i:i+batch_size]
        
        # Run the model and cache activations
        logits, cache = model.run_with_cache(
            batch, 
            names_filter=lambda hook_name: 'resid' in hook_name, 
            device='cpu'  # Use CPU to avoid OOM errors; switch to 'cuda' if you have enough VRAM
        )
        
        # First batch, initialize the dictionary
        if not activations:
            activations = {key: [cache[key]] for key in cache}
        else:
            # Append to existing cache
            for key in cache:
                activations[key].append(cache[key])
                
        # Clear memory
        del logits, cache
        gc.collect()
        torch.cuda.empty_cache()
    
    # Concatenate all batches
    activations = {k: torch.cat(v) for k, v in activations.items()}
    return activations


# -

# ## Load and prepare GSM8K dataset
#
# We'll use the GSM8K dataset which contains math problems that require reasoning to solve.
# We'll append "please provide your answer first, then your reasoning" to each problem.

# +
# Load GSM8K dataset
gsm8k = load_dataset("gsm8k", "main")
print(f"Loaded GSM8K dataset with {len(gsm8k['train'])} training examples and {len(gsm8k['test'])} test examples")

# Look at a sample problem
print("\nSample problem:")
print(gsm8k["train"][0]["question"])
print("\nSample answer:")
print(gsm8k["train"][0]["answer"])

# +
# Define functions to prepare prompts
def prepare_prompts(problems: List[str]) -> List[str]:
    """Add the reasoning instruction to each problem."""
    return [f"{problem}\n\nPlease provide your answer first, then your reasoning." for problem in problems]

# Prepare prompts for training and testing
train_problems = [item["question"] for item in gsm8k["train"]]
test_problems = [item["question"] for item in gsm8k["test"]]

# Limit the number of problems to reduce computation time
N_PROBLEMS = 100  # Adjust based on available compute
train_problems = train_problems[:N_PROBLEMS]
test_problems = test_problems[:min(20, len(test_problems))]  # Smaller test set

train_prompts = prepare_prompts(train_problems)
test_prompts = prepare_prompts(test_problems)

print(f"Prepared {len(train_prompts)} training prompts and {len(test_prompts)} test prompts")
# -

# ## Collect activations from both models
#
# Now we'll run the same prompts through both models and collect their activations.
# This is the key step where we gather the data needed to compute the reasoning direction.

# +
# Tokenize the math problems for both models
tokenized_prompts_original = tokenize_instructions(
    model_original.tokenizer, 
    train_prompts, 
    ORIGINAL_CHAT_TEMPLATE
)

tokenized_prompts_reasoning = tokenize_instructions(
    model_reasoning.tokenizer, 
    train_prompts, 
    REASONING_CHAT_TEMPLATE
)

print(f"Tokenized prompts shape (original): {tokenized_prompts_original.shape}")
print(f"Tokenized prompts shape (reasoning): {tokenized_prompts_reasoning.shape}")
# -

# ### Collect activations (this may take a while)
# We'll run both models on the same inputs and collect their activations.

# +
print("Collecting activations from the original model...")
original_activations = collect_activations(model_original, tokenized_prompts_original)
print("Done collecting activations from the original model")

print("Collecting activations from the reasoning model...")
reasoning_activations = collect_activations(model_reasoning, tokenized_prompts_reasoning)
print("Done collecting activations from the reasoning model")

# Optional: Save activations to disk to avoid recomputing
torch.save(original_activations, 'original_activations.pth')
torch.save(reasoning_activations, 'reasoning_activations.pth')


# -

# ## Calculate the reasoning direction
#
# Now we'll calculate the reasoning direction by taking the difference between
# activations from the reasoning model and the original model.

# +
def get_act_idx(cache_dict, act_name, layer):
    """Helper function to get activations from a specific layer."""
    key = (act_name, layer,)
    return cache_dict[utils.get_act_name(*key)]

# The activation layers to analyze
activation_layers = ['resid_pre', 'resid_mid', 'resid_post']

# Calculate reasoning directions
reasoning_directions = {k: [] for k in activation_layers}

for layer_num in tqdm(range(1, model_original.cfg.n_layers)):
    pos = -1  # Focus on the last token position
    
    for layer in activation_layers:
        # Get mean activations for each model
        original_mean_act = get_act_idx(original_activations, layer, layer_num)[:, pos, :].mean(dim=0)
        reasoning_mean_act = get_act_idx(reasoning_activations, layer, layer_num)[:, pos, :].mean(dim=0)
        
        # Calculate the difference and normalize to get the direction
        reasoning_dir = reasoning_mean_act - original_mean_act
        reasoning_dir = reasoning_dir / reasoning_dir.norm()
        
        reasoning_directions[layer].append(reasoning_dir)

# Save the reasoning directions
torch.save(reasoning_directions, 'reasoning_dirs.pth')
print("Reasoning directions calculated and saved")
# -

# ## Score and rank reasoning directions
#
# Now we'll sort the reasoning directions by their magnitude to identify
# which layers might have the strongest reasoning signal.

# +
# Get all calculated potential reasoning dirs, sort them in descending order
# based on their mean() magnitude
activation_layers = ['resid_pre']  # We can start with just this layer as it's often sufficient

# Flatten and score all directions
activation_scored = sorted(
    [reasoning_directions[layer][l-1] for l in range(1, model_original.cfg.n_layers) for layer in activation_layers], 
    key=lambda x: abs(x.mean()), 
    reverse=True
)

print(f"Ranked {len(activation_scored)} potential reasoning directions")


# -

# ## Test the reasoning direction
#
# Now we'll define a hook to add the reasoning direction to the model's activations
# during inference and test if it enhances the model's reasoning capabilities.

# +
def reasoning_enhancement_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    strength: float = 1.0
):
    """Hook to add the reasoning direction to activations."""
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    
    # Project the activation onto the reasoning direction and add it back
    # Unlike refusal where we subtract, here we're adding more of the reasoning direction
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation + (strength * direction)

def generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 100,
    fwd_hooks = [],
) -> List[str]:
    """Generate text with specified hooks applied."""
    all_toks = torch.zeros(
        (toks.shape[0], toks.shape[1] + max_tokens_generated), 
        dtype=torch.long, 
        device=toks.device
    )
    all_toks[:, :toks.shape[1]] = toks
    
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)  # greedy sampling
            all_toks[:, -max_tokens_generated + i] = next_tokens
    
    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)


# -

# ## Evaluate the reasoning enhancement
#
# Now we'll test our reasoning direction on a few example problems and compare
# the baseline model outputs with the reasoning-enhanced outputs.

# +
# Select the top reasoning direction to test
top_reasoning_dir = activation_scored[0]
print("Selected top reasoning direction for testing")

# Create hooks to inject reasoning direction
strength = 1.0  # Adjust this value to control the magnitude of the effect
hook_fn = functools.partial(reasoning_enhancement_hook, direction=top_reasoning_dir, strength=strength)

# Create hooks for all layers (or you can target specific layers)
fwd_hooks = [
    (utils.get_act_name(act_name, l), hook_fn) 
    for l in range(model_original.cfg.n_layers) 
    for act_name in ['resid_pre', 'resid_mid', 'resid_post']
]

# +
# Test on a few examples
N_TEST_EXAMPLES = 3
print(f"Testing on {N_TEST_EXAMPLES} examples from the test set")

for i in range(N_TEST_EXAMPLES):
    test_prompt = test_prompts[i]
    print(f"\n\n--- EXAMPLE {i+1} ---")
    print(f"PROBLEM:\n{test_prompt}")
    
    # Tokenize the test prompt
    test_tokens = tokenize_instructions(
        model_original.tokenizer, 
        [test_prompt], 
        ORIGINAL_CHAT_TEMPLATE
    )
    
    # Generate baseline response (without reasoning enhancement)
    baseline_response = generate_with_hooks(model_original, test_tokens)
    print("\nBASELINE RESPONSE:")
    print(baseline_response[0])
    
    # Generate reasoning-enhanced response
    enhanced_response = generate_with_hooks(model_original, test_tokens, fwd_hooks=fwd_hooks)
    print("\nREASONING-ENHANCED RESPONSE:")
    print(enhanced_response[0])


# -

# ## Systematic Evaluation
#
# To properly evaluate the effectiveness of our reasoning direction,
# we'll test it on more examples and compare the quality of the responses.

def evaluate_reasoning(
    model: HookedTransformer,
    problems: List[str],
    chat_template: str,
    reasoning_hooks=None,
    max_tokens: int = 150,
    batch_size: int = 4
) -> List[Dict]:
    """Evaluate model performance with and without reasoning enhancement."""
    results = []
    
    for i in tqdm(range(0, len(problems), batch_size)):
        batch_problems = problems[i:i+batch_size]
        tokens = tokenize_instructions(model.tokenizer, batch_problems, chat_template)
        
        # Generate without reasoning enhancement
        baseline_responses = generate_with_hooks(model, tokens, max_tokens_generated=max_tokens)
        
        # Generate with reasoning enhancement if hooks provided
        if reasoning_hooks:
            enhanced_responses = generate_with_hooks(
                model, tokens, max_tokens_generated=max_tokens, fwd_hooks=reasoning_hooks
            )
        else:
            enhanced_responses = ["No enhancement applied"] * len(batch_problems)
        
        # Store results
        for j, (problem, baseline, enhanced) in enumerate(
            zip(batch_problems, baseline_responses, enhanced_responses)
        ):
            results.append({
                "problem": problem,
                "baseline": baseline,
                "enhanced": enhanced
            })
    
    return results

# Evaluate on a larger test set
evaluation_results = evaluate_reasoning(
    model_original,
    test_prompts[:10],  # Use more examples for a better evaluation
    ORIGINAL_CHAT_TEMPLATE,
    reasoning_hooks=fwd_hooks
)

# Print and analyze evaluation results
for i, result in enumerate(evaluation_results):
    print(f"\n--- EVALUATION EXAMPLE {i+1} ---")
    print(f"PROBLEM:\n{result['problem']}")
    print(f"\nBASELINE SOLUTION:\n{result['baseline']}")
    print(f"\nREASONING-ENHANCED SOLUTION:\n{result['enhanced']}")
    print("-" * 80)

# ## Future Directions
#
# Some potential improvements and extensions to this work:
#
# 1. **Fine-tune the strength parameter**: Experiment with different values of the strength parameter
# 2. **Layer-specific intervention**: Apply the reasoning direction to specific layers only
# 3. **Quantitative evaluation**: Develop metrics to measure reasoning quality
# 4. **Orthogonalization**: Create a model with permanent reasoning enhancement by orthogonalizing the weights
# 5. **Multiple reasoning directions**: Identify different aspects of reasoning (deduction, induction, etc.)
#
# This approach of comparing model activations to find meaningful directions in the activation space
# could be extended to many other capabilities beyond reasoning.
