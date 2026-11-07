import numpy as np

def make_group_lookup(groups, n_obs=None):
    """
    Build a lookup mapping each group identifier to its observation indices.

    This mapping can be used for group-level bootstrap resampling, ensuring
    that all observations belonging to a sampled group are selected together.

    Inputs:
        - groups [array-like]: Group identifier for each observation,
          of length N, where N is the number of observations.
          Group identifiers must be hashable.

    Output:
        - dict: Mapping from each unique group identifier to a NumPy array
          containing the indices of observations belonging to that group.
    """
    if groups is None:
        if n_obs is None:
            raise ValueError(
                "n_obs must be provided when groups is None."
            )
        groups = np.arange(n_obs)


    group_lookup = {}

    for idx, g in enumerate(groups):
        group_lookup.setdefault(g, []).append(idx)

    group_lookup = {
        g: np.asarray(v, dtype=int)
        for g, v in group_lookup.items()
    }

    return group_lookup


def bootstrap_indices(group_lookup, rng):
    """
    Generate bootstrap observation indices by resampling groups with replacement.

    The number of sampled groups equals the number of unique groups in the
    original dataset. All observations belonging to a sampled group are
    included together, preserving the within-group structure.

    Groups may be sampled multiple times or not sampled at all. Consequently,
    the returned number of observation indices may differ from the original
    dataset size when groups have unequal sizes.

    Inputs:
        - group_lookup [dict]: Mapping from group identifiers to NumPy arrays
          of observation indices, as returned by make_group_lookup.
        - rng [np.random.Generator]: NumPy random number generator used
          to sample groups with replacement.

    Output:
        - np.ndarray: One-dimensional array of observation indices for the
          bootstrap sample. Indices may be repeated when a group is sampled
          multiple times.
    """
    unique_groups = np.array(list(group_lookup.keys()))

    sampled_groups = rng.choice(
        unique_groups,
        size=len(unique_groups),
        replace=True
    )

    return np.concatenate(
        [group_lookup[g] for g in sampled_groups]
    )