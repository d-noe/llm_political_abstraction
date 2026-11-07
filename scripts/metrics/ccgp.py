from tqdm import tqdm
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.feature_extraction.text import TfidfVectorizer

def get_ccgp_per_layer(
    activations:np.ndarray,
    source_data,
    label_col,
    condition_col,
    group_col:str=None,
    clf_args = {},
    shuffled=False,
    random_seed=None,
    return_detailed=False,
    cv_folds:int = 5,
    indomain_eval:bool=False,
    scorer=None,
):
    results = {
        l:{} for l in range(activations.shape[1])
    }

    decoding_labels = np.array(source_data[label_col])
    conditions = np.array(source_data[condition_col])
    if not group_col is None:
        groups = np.array(source_data[group_col])
    else:
        groups = None

    for curr_layer in tqdm(range(len(results)), desc='Iterating over layers...'):
        features = activations[:,curr_layer,:]

        results[curr_layer] = ccgp_cv(
            features=features,
            decoding_labels=decoding_labels,
            conditions=conditions,
            clf_args=clf_args,
            shuffled=shuffled,
            random_seed=random_seed,
            return_detailed=return_detailed,
            cv_folds=cv_folds,
            groups=groups,
            indomain_eval=indomain_eval,
            scorer=scorer,
        )


    return results


def ccgp_cv(
    features, # N x D
    decoding_labels, # N
    conditions, # N
    clf_args = {},
    shuffled=False,
    random_seed=None,
    return_detailed=False,
    cv_folds:int = 5,
    groups:list=None,
    indomain_eval:bool=False,
    scorer=None,
):
    """
    Individual CCGP score.
    """
    def get_score(clf, x, y, scorer):
        """
        Get classification score, either using provided <scorer>
        or classifier default scorer as a fallback.
        """
        if scorer is None:
            return clf.score(
                x, y
            )
        else:
            y_pred = clf.predict(x)
            return scorer(y_true=y, y_pred=y_pred)

    # set up
    unique_conditions = sorted(list(set(conditions)))
    condition_masks = {
        c: (conditions == c)
        for c in unique_conditions
    }
    condition_data = {
        c: (
            features[condition_masks[c]],
            decoding_labels[condition_masks[c]],
            groups[condition_masks[c]] if not groups is None else None,
        )
        for c in unique_conditions
    }
    # to store results
    CCGP_per_condition_per_fold = {
        c_src:{
            c_trg: np.zeros(cv_folds)
            for c_trg in unique_conditions if (c_trg!=c_src or indomain_eval)
        }
        for c_src in unique_conditions
    }

    # Used for null-model estimates
    if shuffled:
        rng = np.random.default_rng(random_seed)
        decoding_labels = rng.permutation(decoding_labels)

    # Prepare Cross-Validation
    cv = StratifiedGroupKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_seed # re-use same random seed
    )

    for cond_src in tqdm(unique_conditions, desc="Iterating over domains..."):
        features_src, labels_src, groups_src = condition_data[cond_src]

        # cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X=features_src, y=labels_src, groups=groups_src)):

            clf_src = LinearSVC(**clf_args)
            clf_src.fit(features_src[train_idx], labels_src[train_idx])

            if indomain_eval:
                CCGP_per_condition_per_fold[cond_src][cond_src][fold_idx] = get_score(
                    clf=clf_src,
                    x=features_src[test_idx], y=labels_src[test_idx],
                    scorer=scorer
                )

            for cond_trg in unique_conditions:
                if cond_src==cond_trg: # do not evaluate
                    continue

                features_trg, labels_trg, _ = condition_data[cond_trg]
                
                CCGP_per_condition_per_fold[cond_src][cond_trg][fold_idx] = get_score(
                    clf=clf_src,
                    x=features_trg, y=labels_trg,
                    scorer=scorer
                )

    if return_detailed:
        return CCGP_per_condition_per_fold
    else:
        return np.mean([CCGP_per_condition_per_fold[k][v][f] for k in CCGP_per_condition_per_fold.keys() for v in CCGP_per_condition_per_fold[k].keys() for f in range(len(CCGP_per_condition_per_fold[k][v]))])


## TFIDF Baseline

def ccgp_cv_tfidf(
    texts, # N -> start from text
    decoding_labels, # N
    conditions, # N
    clf_args = {},
    shuffled=False,
    random_seed=None,
    return_detailed=False,
    cv_folds:int = 5,
    groups:list=None,
    indomain_eval:bool=False,
    scorer=None,
):
    """
    TFIDF-based CCGP score.
    """
    def get_score(clf, x, y, scorer):
        """
        Get classification score, either using provided <scorer>
        or classifier default scorer as a fallback.
        """
        if scorer is None:
            return clf.score(
                x, y
            )
        else:
            y_pred = clf.predict(x)
            return scorer(y_true=y, y_pred=y_pred)

    # set up
    unique_conditions = sorted(list(set(conditions)))
    condition_masks = {
        c: (conditions == c)
        for c in unique_conditions
    }
    condition_data = {
        c: (
            texts[condition_masks[c]],
            decoding_labels[condition_masks[c]],
            groups[condition_masks[c]] if not groups is None else None,
        )
        for c in unique_conditions
    }
    # to store results
    CCGP_per_condition_per_fold = {
        c_src:{
            c_trg: np.zeros(cv_folds)
            for c_trg in unique_conditions if (c_trg!=c_src or indomain_eval)
        }
        for c_src in unique_conditions
    }

    # Used for null-model estimates
    if shuffled:
        rng = np.random.default_rng(random_seed)
        decoding_labels = rng.permutation(decoding_labels)

    # Prepare Cross-Validation
    cv = StratifiedGroupKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_seed # re-use same random seed
    )

    for cond_src in tqdm(unique_conditions, desc="Iterating over domains..."):
        texts_src, labels_src, groups_src = condition_data[cond_src]

        # cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X=texts_src, y=labels_src, groups=groups_src)):

            # TF-IDF - feature extraction
            curr_vectorizer = TfidfVectorizer()
            features_src_train = curr_vectorizer.fit_transform(texts_src[train_idx])

            # CLASSIFIER
            clf_src = LinearSVC(**clf_args)
            clf_src.fit(features_src_train, labels_src[train_idx])

            if indomain_eval:
                features_src_test = curr_vectorizer.transform(texts_src[test_idx])
                CCGP_per_condition_per_fold[cond_src][cond_src][fold_idx] = get_score(
                    clf=clf_src,
                    x=features_src_test, y=labels_src[test_idx],
                    scorer=scorer
                )

            for cond_trg in unique_conditions:
                if cond_src==cond_trg: # do not evaluate
                    continue

                texts_trg, labels_trg, _ = condition_data[cond_trg]
                
                features_trg = curr_vectorizer.transform(texts_trg)
                
                CCGP_per_condition_per_fold[cond_src][cond_trg][fold_idx] = get_score(
                    clf=clf_src,
                    x=features_trg, y=labels_trg,
                    scorer=scorer
                )

    if return_detailed:
        return CCGP_per_condition_per_fold
    else:
        return np.mean([CCGP_per_condition_per_fold[k][v][f] for k in CCGP_per_condition_per_fold.keys() for v in CCGP_per_condition_per_fold[k].keys() for f in range(len(CCGP_per_condition_per_fold[k][v]))])



# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments with configurable models and parameters."
    )

    parser.add_argument(
        "--cv_folds", "-cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output.",
    )

    parser.add_argument(
        "--compute_tfidf", "-tfidf",
        action="store_true",
        help="Compute TFIDF lexical baseline.",
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
        default="./outputs/raw_outputs/ccgp/",
        help="Base folder path to save the results (default: ./outputs/raw_outputs/ccgp/).",
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


if __name__ == "__main__":
    # Load functions
    import os
    os.sys.path.append("./")
    import argparse
    from scripts.utils import *

    # 0. Parse Arguments
    args = parse_args()

    dataset_name = args.dataset_name
    random_seed = args.random_seed
    cv_folds = args.cv_folds
    models_names = args.models_names
    compute_tfidf = args.compute_tfidf

    verbose = args.verbose
    save_results = args.save_results
    base_output_folder_path = args.base_output_folder_path

    DATA_DICT = load_config_dict(config_path=args.config_dict_path, verbose=verbose)
    
    label_col = DATA_DICT[dataset_name]["label_col"]
    label_values = DATA_DICT[dataset_name]["labels"]
    domain_col = DATA_DICT[dataset_name]["domain_col"]
    group_col = DATA_DICT[dataset_name]["pairwiseid_col"] if "pairwiseid_col" in DATA_DICT[dataset_name].keys() else None

    # Compute TFIDF scores = lexical baseline
    if compute_tfidf:
        if verbose:
            print("Computing scores for lexical baselines...")
        df = pd.read_csv(DATA_DICT[dataset_name]["dataframe_path"])
        res_tfidf = ccgp_cv_tfidf(
            texts=np.array(df[DATA_DICT[dataset_name]["text_col"]]),
            decoding_labels=np.array(df[label_col]),
            conditions=np.array(df[domain_col]),
            clf_args = {'max_iter': 5000,},
            shuffled=False,
            random_seed=random_seed,
            return_detailed=True,
            cv_folds = cv_folds,
            groups=np.array(df[group_col]) if not (group_col is None) else None,
            indomain_eval=True,
            scorer=balanced_accuracy_score,
        )
        if save_results:
            saved_flpth_tfidf = save_array_dict_json(
                data_dict=res_tfidf,
                dataset_name=dataset_name,
                file_name=f"TFIDF", # without extension
                base_dir=base_output_folder_path,
            )
            if verbose:
                print(f"\tTFIDF scores saved at: {saved_flpth_tfidf}")


    # Iterate over each model
    for model_name in models_names:
        print(f"Computing probes for {model_name}.")
        # 1. Load the data
        # 1.A Raw data and pre-computed activations
        df, acts = load_data(
            dataset_name=dataset_name,
            model_name=model_name,
            config_dict=DATA_DICT,
        )
        if verbose:
            print(f"Loaded activations with shape: {acts.shape}.")

        # 2. Compute Cross-domain classification scores
        layers_ccgps = get_ccgp_per_layer(
            activations=acts,
            source_data=df,
            label_col=label_col,
            condition_col=domain_col,
            group_col=group_col,
            clf_args = {'max_iter': 5000,},
            shuffled=False,
            random_seed=random_seed,
            return_detailed=True,
            cv_folds = cv_folds,
            indomain_eval=True,
            scorer=balanced_accuracy_score,
        )

        # 3. Save results
        if save_results:
            saved_flpth = save_array_dict_json(
                data_dict=layers_ccgps,
                dataset_name=dataset_name,
                file_name=f"{model_name}", # without extension
                base_dir=base_output_folder_path,
            )
            if verbose:
                print(f"Classification results for model {model_name} saved at: {saved_flpth}")

    if verbose:
        print(f"✅ Cross-validated (CV={cv_folds}) cross-domain classification performance computed for every layers of {models_names}")