"""
.. _torch_export_llama3.2_1B:

Compiling Llama3.2 1B using the dynamo backend
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular Llama3.2 model."""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import export_llm, generate

# %%
# Define the parameters and initialize the model
MAX_TOKENS = 32
DEVICE = torch.device("cuda:0")

# Define the Phi3.5 mini model from hugging face
model_id = "microsoft/Phi-3.5-mini-instruct"

with torch.no_grad():
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager")
        .eval()
        .half()
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
# Tokenize a sample input prompt and get pytorch model outputs
prompt = "Who are you?"
model_inputs = tokenizer(prompt, return_tensors="pt")
input_ids = model_inputs.input_ids

# Auto-regressive generation loop for greedy decoding using PyTorch model
# We use a custom generate function which is very similar to the huggingface one.
pyt_gen_tokens = generate(model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

# %%
# Compilation with `Torch-TensorRT` using dynamo backend and generate TensorRT outputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Export the llama3.2-1B model into an ExportedProgram which is input of TRT compilation
# To compile the model in FP16, we do the following
# 1) Cast the model to FP16 via model.half()
# 2) Enable use_explicit_typing=True. Certain layers are explicitly casted to FP32 within the pytorch model and this flag respects this behavior during TRT compilation
# 3) Enable use_fp32_acc=True. This ensures all the matmuls are accumulated in FP32 precision (similar to PyTorch)
llama3_ep = export_llm(model, input_ids, max_seq_len=64)
trt_model = torch_tensorrt.dynamo.compile(
    llama3_ep,
    inputs=[input_ids],
    enabled_precision={torch.float32},
    truncate_doule=True,
    device=DEVICE,
    disable_tf32=True,
    use_explicit_typing=True,
    use_fp32_acc=True,
)

# Auto-regressive generation loop for greedy decoding using TensorRT model
# We use a custom generate function which is very similar to the huggingface one.
# Move inputs to GPU
input_ids = input_ids.to(DEVICE)
trt_gen_tokens = generate(trt_model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

print("=============================")
print(
    "Pytorch model generated text: ",
    tokenizer.batch_decode(
        pyt_gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0],
)
print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.batch_decode(
        trt_gen_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0],
)
