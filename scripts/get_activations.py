import argparse
import torch
import numpy as np
import zarr
from zarr.codecs import BloscCodec
from tqdm import tqdm
import gc

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import warnings

import os
os.sys.path.append("./")
from scripts.utils import *

def batched_get_hiddens_to_zarr(
    model,
    tokenizer,
    inputs,
    activation_ids,
    zarr_path,
    hidden_layers=None,
    max_length=None,
    batch_size=16,
):
    if hidden_layers is None:
        try:
            hidden_dim = model.config.hidden_size
            hidden_layers = list(range(model.config.num_hidden_layers))
        except:
            config_dict = model.config.to_dict()["text_config"]
            hidden_layers = list(range(config_dict["num_hidden_layers"]))
            hidden_dim = config_dict["hidden_size"]

    num_samples = len(inputs)
    num_layers = len(hidden_layers)

    root = zarr.open(zarr_path, mode="w")

    act_array = root.create_dataset(
        "activations",
        shape=(num_samples, num_layers, hidden_dim),
        chunks=(batch_size, 1, hidden_dim),
        dtype="float32",
        compressor=BloscCodec(cname="zstd", clevel=3),
    )

    max_len = max(len(str(x)) for x in activation_ids)
    id_array = root.create_dataset(
        "activation_id",
        shape=(num_samples,),
        dtype=f"<U{max_len}",
    )

    root.attrs["layers"] = hidden_layers
    root.attrs["model_name"] = model.config._name_or_path

    sample_idx = 0

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size)):
            batch = inputs[start:start+batch_size]
            ids_batch = activation_ids[start:start+batch_size]

            encoded = tokenizer(
                batch,
                padding="max_length",
                truncation=bool(max_length),
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)

            out = model(**encoded, output_hidden_states=True)
            
            mask = encoded["attention_mask"]
            hidden_states = out.hidden_states[1:]

            B = len(batch)
            batch_out = np.zeros((B, num_layers, hidden_dim), dtype=np.float32)

            for i in range(B):
                last_idx = mask[i].nonzero(as_tuple=True)[0][-1].item()
                for li, layer in enumerate(hidden_layers):
                    batch_out[i, li] = hidden_states[layer][i, last_idx].float().cpu().numpy()

            act_array[start:start+B, :, :] = batch_out
            id_array[start:start+B] = ids_batch

            del out, encoded, batch_out, mask, hidden_states
            gc.collect()

    return zarr_path


# ============================================================
# MAIN
# ============================================================

def parse_args():
    # 1. First, parse a preliminary parser just to find the config path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--exp_config", 
        type=str, 
        default="./configs/exp_config.json",
        help="Path to the JSON configuration file."
    )
    pre_args, unknown  = pre_parser.parse_known_args()

    # 2. Load the JSON config file if it exists
    file_config = load_config_dict(pre_args.exp_config, verbose=False)

    # 3. Set up the main parser, using JSON values as defaults if available
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline with JSON config and CLI overrides."
    )
    
    parser.add_argument(
        "--exp_config", 
        type=str, 
        default=pre_args.exp_config
    )
    parser.add_argument(
        "--data_config_path", 
        type=str, 
        default=file_config.get("data_config_path", "./configs/data_config.json"),
        help="Path to the data configuration file."
    )
    parser.add_argument(
        "--dataset_name", 
        type=str,
        default=file_config.get("dataset_name", "twinviews"),
        help="Name of the dataset."
    )
    parser.add_argument(
        "--device_name", 
        type=str,
        default=file_config.get("device_name", None),
        help="Name of the device (cuda)."
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=file_config.get("batch_size", 32),
        help="Batch sizes when gathering activations."
    )
    parser.add_argument(
        "--max_length", 
        type=int,
        default=file_config.get("max_length", 512),
        help="Max sequences lengths when gathering activations."
    )
    parser.add_argument(
        "--models_names", 
        nargs="+",
        default=file_config.get("models_names", [
                "meta-llama/Llama-3.1-8B",
                "meta-llama/Llama-3.1-8B-Instruct",
                "allenai/OLMo-2-1124-7B-Instruct"
            ]
        ),
        help="Names of the models to use."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        default=file_config.get("verbose", False),
        help="Enable verbose output."
    )

    return parser.parse_args()

if __name__ == "__main__":
    import logging # to avoid warnings polluting logs
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    # 0. Parse Arguments
    args = parse_args()
    verbose = args.verbose
    DATA_DICT = load_config_dict(
        config_path=args.data_config_path, 
        verbose=verbose
    )
    dataset_name = args.dataset_name
    models_names = args.models_names
    updated_models_names = []
    for m in models_names:
        if not '/' in m:
            if 'Llama' in m:
                updated_models_names += [f'meta-llama/{m}']
            elif 'OLMo' in m:
                updated_models_names += [f"allenai/{m}"]
            elif 'gemma' in m:
                updated_models_names += [f"google/{m}"]
            else:
                updated_models_names += [m]
        else:
            updated_models_names += [m]
    models_names = updated_models_names

    data_path = DATA_DICT[dataset_name]["dataframe_path"]
    text_col = DATA_DICT[dataset_name]["text_col"]
    id_col = DATA_DICT[dataset_name]["id_col"]

    base_output_folder = DATA_DICT[dataset_name]["hs_path"]
    os.makedirs(base_output_folder, exist_ok=True)

    bnb_config = {
        "load_in_8bit":True,
        "torch_dtype":torch.float16,
    }
    device_name = args.device_name
    if device_name is None:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = args.batch_size
    max_length = args.max_length

    if verbose:
        print(f"===\nComputing hidden states on dataset: {dataset_name}\nfor models: {models_names}.\n===")

    # 1. Load data
    data = pd.read_csv(data_path)
    ids = data[id_col].astype(str).to_list()
    inputs = data[text_col].to_list()

    # For each model
    for model_name in models_names:
        if verbose:
            print(f"\n==========================================")
            print(f"Processing Model: {model_name}")
            print(f"==========================================")

        model_output_path = os.path.join(base_output_folder, f"{model_name.split('/')[1] if '/' in model_name else model_name}.zarr")
        os.makedirs(model_output_path, exist_ok=True)

        # 2. Load the model
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            quantization_config=BitsAndBytesConfig(**bnb_config),
            local_files_only=True,
            device_map=device_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        batched_get_hiddens_to_zarr(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            activation_ids=ids,  # same length as inputs
            zarr_path=model_output_path,
            hidden_layers=None,
            batch_size=batch_size,
            max_length=max_length,
        )

        print(f"✅ Hidden states successfuly stored at {model_output_path} for {model_name}.")
        
        del model, tokenizer
        gc.collect()