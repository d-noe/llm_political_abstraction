import os
os.sys.path.append("./")

import numpy as np
from tqdm import tqdm
from scripts.sampling.bootstrap import *

# ======================================================================
# Compute Contrast Vectors
# ======================================================================

def compute_contrast_vectors(
    activations: np.ndarray,
    domains: list | np.ndarray,
    labels: list | np.ndarray,
    label_values: list,
    domains_unique: list | np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute normalized contrast vectors for each domain.

    For each domain, the contrast vector is defined as the difference
    between the mean activation vectors of two label classes, normalized
    to unit Euclidean norm.

    Inputs:
        - activations [np.ndarray] | required
            Activation matrix of shape (N, D),
            where N is the number of samples and D is the activation dimension.
        - domains [list | np.ndarray] | required
            Domain assignment for each sample.
        - labels [list | np.ndarray] | required 
            Label associated with each sample.
        - label_values [list] | required
            Two label values defining the contrast.
        - domains_unique [list | np.ndarray | None] | default = None
            Domains to process.
            If None, the unique domains are inferred from `domains`.

    Output:
        - np.ndarray
            Contrast vectors of shape (n_domains, D),
            with one unit-norm vector per domain.
    """
    domains = np.asarray(domains)
    labels = np.asarray(labels)

    if len(label_values) != 2:
        raise ValueError("label_values must contain exactly two labels.")

    if domains_unique is None:
        domains_unique = np.unique(domains)

    vectors = np.empty(
        (len(domains_unique), activations.shape[1]),
        dtype=float,
    )

    for i, dom in enumerate(domains_unique):
        mask = domains == dom

        X = activations[mask]
        y = labels[mask]

        X0 = X[y == label_values[0]]
        X1 = X[y == label_values[1]]

        if len(X0) == 0 or len(X1) == 0:
            raise ValueError(
                f"Both label classes must be present in domain {dom!r}."
            )

        diff = X0.mean(axis=0) - X1.mean(axis=0)
        norm = np.linalg.norm(diff)

        if norm == 0:
            raise ValueError(
                f"Contrast vector has zero norm in domain {dom!r}."
            )

        vectors[i] = diff / norm

    return vectors

# ======================================================================
# Compute Gram Matrix & Obtain Participation Ratio
# ======================================================================

def compute_gram_matrix(
    contrast_vectors:np.ndarray
) -> np.ndarray:
    """
    Compute the Gram Matrix G (dim: NxN) based on contrast vectors matrix V (dim: Nxd)
    — with N: the number of contrast vectors, and d: the dimension of the activation space

    Inputs:
        - contrast_vectors [np.ndarray] | required
    Output:
        - [np.ndarray]
            np.array[np.array]: Gram Matrix G = VV^T
    """
    return contrast_vectors @ contrast_vectors.T


def participation_ratio(
    G:np.ndarray
) -> np.ndarray:
    """
    Compute Participation Ratio (PR) based on the Gram Matrix G.

    Note the PR is originally defined based on the eigenvalues lambda_k of G:
        PR = ((sum_k lambda_k)^2)(sum_k lambda_k^2)
    Since G is a squared matrix (NxN) by construction, this formula is strictly 
    equivalent to the following expression, which avoids explicitly computing 
    the eigenvalues:
        PR = trace(G)^2 / trace(G @ G)

    Inputs:
        - G [np.ndarray]
            A squared Gram matrix
    Output:
        - float:
            Participation Ratio score
    """
    tr = np.trace(G)
    tr2 = np.trace(G @ G)
    return tr * tr / tr2

# ======================================================================
# Observed PR across all layers
# ======================================================================

def get_observed_pr_per_layer(
    activations, 
    labels, 
    domains, 
    label_values
):
    """Compute true observed Gram matrices and PRs across all layers."""
    n_layers = activations.shape[1]
    domains_unique = np.array(sorted(list(set(domains))))
    
    obs_grams = np.empty((n_layers, len(domains_unique), len(domains_unique)))
    obs_prs = np.empty(n_layers)

    for layer in range(n_layers):
        X = activations[:, layer, :]
        V = compute_contrast_vectors(
            X, 
            domains, 
            labels, 
            label_values
        )
        G = compute_gram_matrix(V)
        obs_grams[layer] = G
        obs_prs[layer] = participation_ratio(G)
        
    return obs_grams, obs_prs

# ======================================================================
# Null PR across all layers
# ======================================================================

def permute_labels_within_domains(
    labels, 
    domains, 
    rng
):
    """Randomly permute labels independently within each domain."""
    permuted = labels.copy()
    for dom in np.unique(domains):
        idx = np.where(domains == dom)[0]
        permuted[idx] = rng.permutation(permuted[idx])
    return permuted

def get_null_pr_per_layer(
    activations,
    labels,
    domains, 
    label_values, 
    n_permutations: int = 100, 
    random_seed: int = 0
):
    """Compute null PRs across all layers via label permutation."""
    rng = np.random.default_rng(random_seed)
    n_layers = activations.shape[1]
    null_prs = np.empty((n_layers, n_permutations))

    for p in tqdm(range(n_permutations), desc="Computing null PRs..."):
        permuted_labels = permute_labels_within_domains(labels, domains, rng)
        for layer in range(n_layers):
            X = activations[:, layer, :]
            V = compute_contrast_vectors(X, domains, permuted_labels, label_values)
            G = compute_gram_matrix(V)
            null_prs[layer, p] = participation_ratio(G)
            
    return null_prs

# ======================================================================
# Bootstrap PR across all layers
# ======================================================================

def get_pr_per_layer(
    activations, # N x L x D
    source_data,
    label_col,
    label_values,
    domain_col,
    group_col=None,
    B:int=100,
    random_seed:int=92,
):

    labels = np.array(source_data[label_col])
    domains = np.array(source_data[domain_col])
    domains_unique = np.array(sorted(list(set(domains))))
    if not group_col is None:
        groups = np.array(source_data[group_col])
    else:
        groups = None

    group_lookup = make_group_lookup(groups)
    n_layers = activations.shape[1]

    ov_res_grams = np.empty((n_layers, B, len(domains_unique), len(domains_unique)))
    ov_res_prs = np.empty((n_layers, B))

    for layer in tqdm(range(n_layers), desc="PR over layers..."):
        layer_res_grams, layer_res_prs = pr_btstrp(
            layer_acts=activations[:,layer,:],
            labels=labels,
            domains=domains,
            label_values=label_values,
            group_lookup=group_lookup,
            B=B,
            random_seed=random_seed,
        )
        ov_res_grams[layer] = layer_res_grams
        ov_res_prs[layer] = layer_res_prs
    
    return ov_res_grams, ov_res_prs

def pr_btstrp(
    layer_acts, # N x D
    labels, # N
    domains, # N
    label_values,
    group_lookup,
    B:int=100,
    random_seed:int=92,
):
    rng = np.random.default_rng(random_seed)

    res_grams = np.empty((B, len(set(domains)), len(set(domains))))
    res_prs = np.empty(B)

    for b in tqdm(range(B), leave=False, desc="Bootstraping PR..."):
        idx = bootstrap_indices(group_lookup, rng)

        boot_labels = labels[idx]
        boot_domains = domains[idx]

        X = layer_acts[idx]
        V = compute_contrast_vectors(
            X,
            boot_domains,
            boot_labels,
            label_values
        )
        G = compute_gram_matrix(V)

        res_grams[b] = G
        res_prs[b] = participation_ratio(G)

    return res_grams, res_prs

# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments with configurable models and parameters."
    )

    parser.add_argument(
        "--bootstraped_samples", "-B",
        type=int,
        default=100,
        help="Number of bootstraps (default: 100).",
    )

    parser.add_argument(
        "--n_permutations",
        type=int,
        default=100,
        help="Number of permutations for the null distribution (default: 100).",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=92,
        help="Random seed for the experiments (default: 92).",
    )

    parser.add_argument(
        "--config_dict_path",
        type=str,
        default="./configs/data_config.json",
        help="Path to the configuration dictionary (default: ./configs/data_config.json).",
    )

    parser.add_argument(
        "--base_output_folder_path",
        type=str,
        default="./outputs/raw_outputs/pr/",
        help="Base folder path to save the results (default: ./outputs/raw_outputs/pr/).",
    )

    parser.add_argument(
        "--no_save_results",
        action="store_false",
        dest="save_results",
        help="Disable saving results (results are saved by default).",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="twinviews",
        help="Name of the dataset (default: twinviews).",
    )

    parser.add_argument(
        "--models_names",
        nargs="+",
        default=[
            "Llama-3.1-8B",
            "Llama-3.1-8B-Instruct",
            "OLMo-2-1124-7B-Instruct",
        ],
        help="Names of the models to use (provide one or more).",
    )

    return parser.parse_args()


if __name__ == '__main__':
    # Load functions
    import argparse
    from scripts.utils import *
    from scripts.sampling.bootstrap import *
    # 0. Parse arguments & load variables
    args = parse_args()
     
    dataset_name = args.dataset_name
    models_names = args.models_names

    B = args.bootstraped_samples
    n_permutations = args.n_permutations
    random_seed = args.random_seed
    verbose = args.verbose
    save_results = args.save_results
    base_output_folder_path = args.base_output_folder_path

    DATA_DICT = load_config_dict(config_path=args.config_dict_path, verbose=verbose)
    # PARAM_DICT = load_config_dict() # TODO! -> make file and load from alternative path
    
    if verbose:
        print(f"Preparing PR computation for\n\t-dataset: {dataset_name}\n\t-models: {models_names}")

    # Iterate over each model
    for model_name in models_names:
        # 1. Load the data
        # 1.A Raw data and pre-computed activations
        df, acts = load_data(
            dataset_name=dataset_name,
            model_name=model_name,
            config_dict=DATA_DICT,
        )
        if verbose:
            print(f"Computing PR for {model_name}. Loaded activations with shape: {acts.shape}.")
        # 1.B Extract values of interest
        label_col = DATA_DICT[dataset_name]["label_col"]
        label_values = DATA_DICT[dataset_name]["labels"]
        domain_col = DATA_DICT[dataset_name]["domain_col"]
        group_col = DATA_DICT[dataset_name]["pairwiseid_col"] if "pairwiseid_col" in DATA_DICT[dataset_name].keys() else None

        labels = np.array(df[label_col])
        domains = np.array(df[domain_col])

        # 2. Compute Participation Ratio
        # 2.A Observed
        if verbose:
            print(f"Computing observed PR for {model_name}...")
        obs_grams, obs_prs = get_observed_pr_per_layer(
            activations=acts,
            labels=labels,
            domains=domains,
            label_values=label_values
        )

        # 2.B Bootstraped
        if verbose:
            print(f"Computing bootstrapped PR (B={B}) for {model_name}...")
        ov_res_grams, ov_res_prs = get_pr_per_layer(
            activations=acts,
            source_data=df,
            label_col=label_col,
            label_values=label_values,
            domain_col=domain_col,
            group_col=group_col,
            B=B,
            random_seed=random_seed,
        )

        # 2.C Null
        if verbose:
            print(f"Computing null PR (n_perm={n_permutations}) for {model_name}...")
        null_prs = get_null_pr_per_layer(
            activations=acts,
            labels=labels,
            domains=domains,
            label_values=label_values,
            n_permutations=n_permutations,
            random_seed=random_seed,
        )

        # 3. Save results
        if save_results:
            if verbose:
                print(f"Saving results for {model_name}:")
            # Save Observed
            saved_flpth_obs_prs = save_array_npy(
                data_array=obs_prs,
                dataset_name=dataset_name,
                file_name=f"observed_prs_{model_name}",
                base_dir=base_output_folder_path,
            )
            saved_flpth_obs_grams = save_array_npy(
                data_array=obs_grams,
                dataset_name=dataset_name,
                file_name=f"observed_grams_{model_name}",
                base_dir=base_output_folder_path,
            )
            if verbose:
                print(f"\tObserved:\n\t\t-PRs   saved at: {saved_flpth_obs_prs}\n\t\t-Grams saved at: {saved_flpth_obs_grams}")

            # Save Bootstrap
            saved_flpth_btstrp_prs = save_array_npy(
                data_array=ov_res_prs,
                dataset_name=dataset_name,
                file_name=f"bootstraped_{B}_prs_{model_name}", # without extension
                base_dir=base_output_folder_path,
            )
            saved_flpth_btstrp_grams = save_array_npy(
                data_array=ov_res_grams,
                dataset_name=dataset_name,
                file_name=f"bootstraped_{B}_grams_{model_name}", # without extension
                base_dir=base_output_folder_path,
            )
            if verbose:
                print(f"\tBootstraped:\n\t\t-PRs   saved at: {saved_flpth_btstrp_prs}\n\t\t-Grams saved at: {saved_flpth_btstrp_grams}")

            # Save Null
            saved_flpth_null = save_array_npy(
                data_array=null_prs,
                dataset_name=dataset_name,
                file_name=f"null_{n_permutations}_prs_{model_name}",
                base_dir=base_output_folder_path,
            )
            if verbose:
                print(f"\tNull:\n\t\t-PRs   saved at: {saved_flpth_null}")