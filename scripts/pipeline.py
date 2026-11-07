import os
os.sys.path.append("./")
import argparse
import numpy as np
import pandas as pd

# Import functions from your metric scripts
from scripts.metrics.ccgp import (
    get_ccgp_per_layer, 
    ccgp_cv_tfidf
)
from scripts.metrics.participation_ratio import (
    get_observed_pr_per_layer,
    get_pr_per_layer,
    get_null_pr_per_layer,
)
from scripts.utils import *
from sklearn.metrics import balanced_accuracy_score

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
        "--models_names", 
        nargs="+",
        default=file_config.get("models_names", ["Llama-3.1-8B","Llama-3.1-8B-Instruct","OLMo-2-1124-7B-Instruct"]),
        help="Names of the models to use."
    )
    parser.add_argument(
        "--random_seed", 
        type=int,
        default=file_config.get("random_seed", 92),
        help="Random seed."
    )
    parser.add_argument(
        "--cv_folds", 
        type=int,
        default=file_config.get("cv_folds", 5),
        help="Number of cross-validation folds for probe classifications."
    )
    parser.add_argument(
        "--compute_tfidf", "-tfidf",
        action="store_true",
        default=file_config.get("compute_tfidf", True),
        help="Compute lexical (tfidf) baseline for CCGP."
    )
    parser.add_argument(
        "--bootstrap_samples", 
        type=int,
        default=file_config.get("bootstrap_samples", 100),
        help="Number of bootstrap samples for PR computation."
    )
    parser.add_argument(
        "--n_permutations", 
        type=int,
        default=file_config.get("n_permutations", 200),
        help="Number of permutations for null PR computation."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        default=file_config.get("verbose", False),
        help="Enable verbose output."
    )
    parser.add_argument(
        "--raw_outputs_folder", 
        type=str,
        default=file_config.get("raw_outputs_folder", "./outputs/raw_outputs/"),
        help="Base folder to store the outputs of the computed metrics"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # 0. Parse Arguments
    args = parse_args()
    verbose = args.verbose
    DATA_DICT = load_config_dict(
        config_path=args.data_config_path, 
        verbose=verbose
    )

    base_output_folder = args.raw_outputs_folder
    base_ccgp_dir = os.path.join(base_output_folder, "ccgp")
    os.makedirs(base_ccgp_dir, exist_ok=True)
    base_pr_dir = os.path.join(base_output_folder, "pr")
    os.makedirs(base_pr_dir, exist_ok=True)

    dataset_name = args.dataset_name
    random_seed = args.random_seed
    cv_folds = args.cv_folds
    B = args.bootstrap_samples
    n_permutations = args.n_permutations
    models_names = args.models_names

    compute_tfidf = args.compute_tfidf

    label_col = DATA_DICT[dataset_name]["label_col"]
    label_values = DATA_DICT[dataset_name]["labels"]
    domain_col = DATA_DICT[dataset_name]["domain_col"]
    group_col = DATA_DICT[dataset_name].get("pairwiseid_col", None)

    print(f"Running pipeline for dataset: {args.dataset_name}.")

    # 1. Compute Lexical Baseline (TFIDF for CCGP) if requested
    if compute_tfidf:
        if args.verbose:
            print("=== Computing TF-IDF Lexical Baseline (CCGP) ===")
        df_full = pd.read_csv(DATA_DICT[dataset_name]["dataframe_path"])
        res_tfidf = ccgp_cv_tfidf(
            texts=np.array(df_full[DATA_DICT[dataset_name]["text_col"]]),
            decoding_labels=np.array(df_full[label_col]),
            conditions=np.array(df_full[domain_col]),
            clf_args={'max_iter': 5000},
            shuffled=False,
            random_seed=random_seed,
            return_detailed=True,
            cv_folds=cv_folds,
            groups=np.array(df_full[group_col]) if group_col else None,
            indomain_eval=True,
            scorer=balanced_accuracy_score,
        )
        save_array_dict_json(
            data_dict=res_tfidf, 
            dataset_name=dataset_name, 
            file_name="TFIDF", 
            base_dir=base_ccgp_dir
        )

    # 2. Iterate Over Models (Loading activations once per model for both metrics)
    for model_name in models_names:
        if verbose:
            print(f"\n==========================================")
            print(f"Processing Model: {model_name}")
            print(f"==========================================")

        df, acts = load_data(
            dataset_name=dataset_name, 
            model_name=model_name, 
            config_dict=DATA_DICT
        )
        labels = np.array(df[label_col])
        domains = np.array(df[domain_col])

        # --- Metric A: CCGP ---
        if verbose:
            print(f"[{model_name}] Computing CCGP per layer...")
        layers_ccgps = get_ccgp_per_layer(
            activations=acts,
            source_data=df,
            label_col=label_col,
            condition_col=domain_col,
            group_col=group_col,
            clf_args={'max_iter': 5000},
            shuffled=False,
            random_seed=random_seed,
            return_detailed=True,
            cv_folds=cv_folds,
            indomain_eval=True,
            scorer=balanced_accuracy_score,
        )
        save_array_dict_json(
            data_dict=layers_ccgps, 
            dataset_name=dataset_name, 
            file_name=f"{model_name}", 
            base_dir=base_ccgp_dir
        )
        

        # --- Metric B: Participation Ratio (PR) ---
        if verbose:
            print(f"[{model_name}] Computing Observed, Bootstrapped, and Null PR...")
        
        obs_grams, obs_prs = get_observed_pr_per_layer(
            activations=acts,
            labels=labels,
            domains=domains,
            label_values=label_values
        )
        ov_res_grams, ov_res_prs = get_pr_per_layer(
            activations=acts,
            source_data=df,
            label_col=label_col,
            label_values=label_values,
            domain_col=domain_col,
            group_col=group_col,
            B=B, 
            random_seed=random_seed
        )
        null_prs = get_null_pr_per_layer(
            activations=acts,
            labels=labels,
            domains=domains,
            label_values=label_values,
            n_permutations=n_permutations,
            random_seed=random_seed
        )

        # Save PR results
        save_array_npy(
            data_array=obs_prs, 
            dataset_name=dataset_name, 
            file_name=f"observed_prs_{model_name}", 
            base_dir=base_pr_dir
        )
        save_array_npy(
            data_array=obs_grams,
            dataset_name=dataset_name,
            file_name=f"observed_grams_{model_name}",
            base_dir=base_pr_dir
        )
        save_array_npy(
            data_array=ov_res_prs,
            dataset_name=dataset_name,
            file_name=f"bootstraped_{B}_prs_{model_name}",
            base_dir=base_pr_dir
        )
        save_array_npy(
            data_array=ov_res_grams,
            dataset_name=dataset_name,
            file_name=f"bootstraped_{B}_grams_{model_name}",
            base_dir=base_pr_dir
        )
        save_array_npy(
            data_array=null_prs,
            dataset_name=dataset_name,
            file_name=f"null_{n_permutations}_prs_{model_name}",
            base_dir=base_pr_dir
        )

    print("\n✅ Full evaluation pipeline completed successfully!")
    print(f"\n ==== Results are stored at: {base_output_folder}.")
