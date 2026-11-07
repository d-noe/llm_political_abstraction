"""
General util functions re-used across the pipeline.
"""
import os
import zarr
import json
import pickle
import numpy as np
import pandas as pd

# ======================================================================
# LOADING UTILS
# ======================================================================

def load_config_dict(
    config_path:str="./configs/data_config.json",
    verbose:bool=True,
) -> dict:
    """
    Load the JSON-formatted config file as a python dictionary.
    The config must be a dict[dict]

    Input:
        - config_path [str] | default = "./configs/data_config.json"
            The path to the config file
        - verbose [bool] | default = True
            If set to True: print the keys of the loaded config
    Output:
        - [dict]
            Configuration file loaded as a Python dictionary
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    f.close()

    if verbose:
        print(f"The loaded config file (from {config_path}), contains:")
        for k, v in config_dict.items():
            print(f"\t{k}: {list(v.keys())}")

    return config_dict

def load_data(
    model_name:str,
    dataset_name:str,
    config_dict:dict=None,
    config_path:str="./configs/data_config.json",
    verbose:bool=True,
) -> tuple:
    """
    Load the pre-computed activations and the source dataframe based on the 
    data contained in the config file, for a given dataset and model.
    Activations must be pre-computed and stored at the path provided by the
    config file.

    This function considers that the source dataframe is stored 
    the path given by:
        config_dict[<dataset_name>]['dataframe_path']
    and that the pre-computed activation vectors are stored at:
        f"{config_dict[<dataset_name>]['hs_path']/<model_name.split('/').zarr>}"
        as a .zarr compressed file.

    Inputs:
        - model_name [str] | required
            Name of the model
        - dataset_name [str] | required
            Name of the dataset
         - config_dict [dict] | default = None
            The pre-loaded config dict
            If not given (None, default), loads the config from `config_path`
        - config_path [str] | default = "./configs/data_config.json"
            The path to the config file
        - verbose [bool] | default = True
            If set to True: print the keys of the loaded config
    Outputs:
        tuple(pd.DataFrame, np.ndarray)
            0: [pd.DataFrame]
                source dataframe loaded as a pd.DataFrame
            1: [np.ndarray]
                pre-computed activation vectors
    """
    # Load the config
    if config_dict is None:
        config_dict = load_config_dict(config_path=config_path, verbose=verbose)
    # Load the source dataframe
    df = pd.read_csv(config_dict[dataset_name]["dataframe_path"])
    # Load the pre-computed activations
    z = zarr.open(
        os.path.join(
            config_dict[dataset_name]['hs_path'], 
            f"{model_name.split('/')[1] if '/' in model_name else model_name}.zarr"
        ),
        mode="r"
    )
    acts = z["activations"]          # (N, L, H)
    act_ids = np.array(z["activation_id"]).astype(str)

    del z

    if verbose:
        print(f"- Dataset loaded with {len(df)} rows.\n\t- Columns are: {list(df.columns)}.")
        print(f"- Pre-computed activations loaded, with shape: {acts.shape}.")

    return df, acts


# ======================================================================
# SAVING UTILS
# ======================================================================

def _make_folder_path(
    dataset_name:str,
    base_dir:str,
):
    # folder_path: base_dir/dataset_name/
    folder_path = os.path.join(base_dir, dataset_name)
    if not os.path.exists(folder_path): # create it if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
    return folder_path

## Arrays

def save_array_npy(
    data_array:dict,
    dataset_name:str,
    file_name:str,
    base_dir:str,
):
    # Repository path
    base_folder_path = _make_folder_path(
        dataset_name=dataset_name,
        base_dir=base_dir
    )

    file_path = os.path.join(base_folder_path, f"{file_name}.npy")

    np.save(file_path, data_array, allow_pickle=True)

    return file_path

def load_array_npy(
    file_path:str,
):
    return np.load(file_path)

## Dict[pd.DataFrame]

def save_dataframe_dict_pickle( # for dataframes?
    data_dict:dict,
    dataset_name:str,
    file_name:str,
    base_dir:str,
):
    # Repository path
    base_folder_path = _make_folder_path(
        dataset_name=dataset_name,
        base_dir=base_dir
    )

    # File path with chosen extension
    file_path = os.path.join(base_folder_path, f"{file_name}.pkl")

    # Save dictionary of Python objects
    with open(file_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    return file_path

def load_pickle_dict(
    file_path:str,
):
    with open(file_path, "rb") as f: 
        d = pickle.load(f)
    f.close()
    return d

## Dict[list|np.ndarray]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_array_dict_json( # for everything else?
    data_dict:dict,
    dataset_name:str,
    file_name:str,
    base_dir:str,
):
    # Repository path
    base_folder_path = _make_folder_path(
        dataset_name=dataset_name,
        base_dir=base_dir
    )

    file_path = os.path.join(base_folder_path, f"{file_name}.json")
    
    with open(file_path, 'w') as f: 
        json.dump(data_dict, f, indent=4, cls=NumpyEncoder)
    f.close()

    return file_path

def load_json_dict(
    file_path:str,
):
    with open(file_path, 'r') as f:
        d = json.load(f)
    f.close()

    return d
