import json

import numpy as np


def load_json_data(file_path: str):
    """
    Loads JSON data from a specified file path with robust error handling.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: The loaded JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        Exception: For any other unexpected errors during file operations.
    """
    try:
        # Using utf-8 encoding for broader compatibility with various JSON files
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print(f"Please ensure '{file_path}' exists, or provide its full path.")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'.")
        print("Please check the file's content for valid JSON format (e.g., missing commas, unescaped quotes).")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while trying to load '{file_path}': {e}")
        raise


def estimate_individual_item_volumes_bayesian(packages_data: list, all_item_ids: list,
                                              verbose: bool = False) -> dict:
    """
    Estimates the true volume for each individual item using Bayesian linear regression.

    Every package observation is a noisy sum of its items' volumes: P = A·V + eps.
    With a Gaussian prior on the volume vector V and Gaussian measurement noise,
    the posterior over V has a closed form; the posterior mean is the estimate.

    Args:
        packages_data: List of dictionaries from packages.json.
        all_item_ids: A sorted list of all unique item IDs present across both datasets,
                      which defines the order of the unknown volume vector V.
        verbose: Print the estimation setup and every estimated volume.

    Returns:
        A dictionary mapping item_id to its estimated volume (posterior mean).
    """
    if not packages_data:
        print("Warning: packages.json is empty. Cannot perform Bayesian estimation. Using prior mean for all items.")
        prior_mean_volume = 5.0  # A reasonable default if no data
        return {item_id: prior_mean_volume for item_id in all_item_ids}

    # Map item IDs to column indices for the design matrix A
    item_id_to_idx = {item_id: i for i, item_id in enumerate(all_item_ids)}
    num_unique_items = len(all_item_ids)
    num_packages = len(packages_data)

    # Known measurement error variance for the total_volume of a package
    # The problem states error variance = 2
    measurement_error_variance = 2.0

    # --- Construct the Design Matrix A and Observation Vector P ---
    # P: Vector of observed total_volumes (y in y = Ax + epsilon)
    P = np.array([pkg['total_volume'] for pkg in packages_data]).reshape(-1, 1)

    # A: Design matrix (num_packages x num_unique_items)
    A = np.zeros((num_packages, num_unique_items))
    for i, pkg in enumerate(packages_data):
        for item_id in pkg['items']:
            if item_id in item_id_to_idx:  # Only include items we are tracking
                A[i, item_id_to_idx[item_id]] = 1.0

    # --- Define Prior Parameters for item volumes V (mu_0, Sigma_0) ---
    # Assuming an independent Normal prior for each item's volume V_j ~ N(mu_prior, sigma_prior_squared)

    # Estimate a global average item volume from the package data for a more informed prior.
    avg_items_per_package = np.mean([len(p['items']) for p in packages_data]) if packages_data else 1.0
    avg_volume_per_package = np.mean(P) if P.size > 0 else 5.0  # Default if no data

    mu_prior = avg_volume_per_package / avg_items_per_package if avg_items_per_package > 0 else 5.0
    mu_prior = max(0.1, mu_prior)  # Ensure prior mean is positive and not too small

    sigma_prior_squared = 100.0  # Large prior variance indicating high uncertainty initially

    # mu_0: Vector of prior means for V (each element is mu_prior)
    mu_0 = np.full((num_unique_items, 1), mu_prior)

    # Sigma_0: Covariance matrix for prior (diagonal matrix with sigma_prior_squared on diagonal)
    Sigma_0 = np.diag(np.full(num_unique_items, sigma_prior_squared))

    if verbose:
        print(f"--- Bayesian Estimation of Individual Item Volumes ---")
        print(f"  Number of unique items to estimate: {num_unique_items}")
        print(f"  Number of package observations: {num_packages}")
        print(f"  Assumed measurement error variance (sigma_epsilon^2): {measurement_error_variance}")
        print(f"  Prior for each item volume: N(mean={mu_prior:.2f}, variance={sigma_prior_squared})")

    # --- Calculate Posterior Parameters for V (mu_N, Sigma_N) ---
    A_T = A.T
    Sigma_0_inv = np.linalg.inv(Sigma_0)

    try:
        Sigma_N_inv = (A_T @ A / measurement_error_variance) + Sigma_0_inv
        Sigma_N = np.linalg.inv(Sigma_N_inv)
        mu_N = Sigma_N @ ((A_T @ P / measurement_error_variance) + (Sigma_0_inv @ mu_0))
    except np.linalg.LinAlgError as e:
        print(f"Error: Linear algebra error during Bayesian estimation: {e}")
        print("This might indicate that the system of equations is underdetermined (not enough unique package data).")
        raise RuntimeError("Failed to estimate item volumes due to linear algebra error.")

    # Map posterior means back to item IDs
    estimated_volumes = {}
    for i, item_id in enumerate(all_item_ids):
        # Ensure estimated volumes are non-negative, as real volumes can't be negative.
        # Clip at a small positive value to avoid issues with zero volume in calculations like value/volume ratio.
        estimated_volumes[item_id] = max(0.01, mu_N[i, 0])

    if verbose:
        print("\n  Estimated Volume (Posterior Mean) for each item:")
        for item_id, vol in estimated_volumes.items():
            print(f"    {item_id}: {vol:.2f} liters")
        print(f"  (Note: Items only in items.json, not packages.json, will have volumes based primarily on the prior mean.)")

    return estimated_volumes


def prepare_items(items_file: str, packages_file: str, verbose: bool = False) -> list:
    """
    Loads both datasets and attaches a Bayesian-estimated volume to every item.

    Args:
        items_file: Path to items.json (names and prices).
        packages_file: Path to packages.json (noisy package observations).
        verbose: Print the full estimation details instead of a one-line summary.

    Returns:
        list: items from items.json, each with an added 'volume' key.
    """
    packages_data = load_json_data(packages_file)
    items_data = load_json_data(items_file)

    all_unique_item_ids = {item['name'] for item in items_data}
    for pkg in packages_data:
        all_unique_item_ids.update(pkg['items'])
    all_unique_item_ids_list = sorted(all_unique_item_ids)

    estimated_volumes = estimate_individual_item_volumes_bayesian(
        packages_data, all_unique_item_ids_list, verbose=verbose)
    for item in items_data:
        item['volume'] = estimated_volumes.get(item['name'], 0.01)

    if not verbose:
        volumes = [item['volume'] for item in items_data]
        print(f"Estimated volumes for {len(items_data)} items from {len(packages_data)} package "
              f"observations ({min(volumes):.2f}-{max(volumes):.2f} L, use --verbose for details).")
    return items_data
