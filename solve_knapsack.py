import json
import numpy as np
import time
import tracemalloc
from dp_solver import solve_knapsack_dp_discretized
from dp_solver_memory_optimized import solve_knapsack_dp_discretized_memory_optimized

# --- Helper function to calculate total volume of selected items ---
def calculate_total_volume(items_data: list, selected_ids: list) -> float:
    """
    Calculates total volume for selected item names.

    Args:
        items_data: List of item dictionaries with 'name' and 'volume'.
        selected_ids: List of item names selected by the algorithm.

    Returns:
        float: Total volume in liters.
    """
    id_to_item = {item["name"]: item for item in items_data}
    return sum(id_to_item[item_id]["volume"] for item_id in selected_ids if item_id in id_to_item)

# --- Helper function for JSON loading ---
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
            data = json.load(f)
        print(f"Successfully loaded data from '{file_path}'")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print(f"Please ensure '{file_path}' exists in the same directory as the script, or provide its full path.")
        raise # Re-raise the exception to be caught in the main block for graceful exit
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'.")
        print("Please check the file's content for valid JSON format (e.g., missing commas, unescaped quotes).")
        raise # Re-raise the exception
    except Exception as e:
        print(f"An unexpected error occurred while trying to load '{file_path}': {e}")
        raise # Re-raise the exception


# --- Part 1: Bayesian Estimation for Item Volumes ---

def estimate_individual_item_volumes_bayesian(packages_data: list, all_item_ids: list) -> dict:
    """
    Estimates the true volume for each individual item using Bayesian linear regression.
    
    Args:
        packages_data: List of dictionaries from packages.json.
        all_item_ids: A sorted list of all unique item IDs present across both datasets,
                      which defines the order of the unknown volume vector V.
    
    Returns:
        A dictionary mapping item_id to its estimated volume (posterior mean).
    """
    if not packages_data:
        print("Warning: packages.json is empty. Cannot perform Bayesian estimation. Using prior mean for all items.")
        # Fallback: if no package data, return prior mean for all items
        prior_mean_volume = 5.0 # A reasonable default if no data
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
    P = np.array([pkg['total_volume'] for pkg in packages_data]).reshape(-1, 1) # Ensure P is a column vector

    # A: Design matrix (num_packages x num_unique_items)
    A = np.zeros((num_packages, num_unique_items))
    for i, pkg in enumerate(packages_data):
        for item_id in pkg['items']:
            if item_id in item_id_to_idx: # Only include items we are tracking
                A[i, item_id_to_idx[item_id]] = 1.0

    # --- Define Prior Parameters for item volumes V (mu_0, Sigma_0) ---
    # Assuming an independent Normal prior for each item's volume V_j ~ N(mu_prior, sigma_prior_squared)
    
    # Estimate a global average item volume from the package data for a more informed prior.
    avg_items_per_package = np.mean([len(p['items']) for p in packages_data]) if packages_data else 1.0
    avg_volume_per_package = np.mean(P) if P.size > 0 else 5.0 # Default if no data
    
    mu_prior = avg_volume_per_package / avg_items_per_package if avg_items_per_package > 0 else 5.0
    mu_prior = max(0.1, mu_prior) # Ensure prior mean is positive and not too small

    sigma_prior_squared = 100.0 # Large prior variance indicating high uncertainty initially

    # mu_0: Vector of prior means for V (each element is mu_prior)
    mu_0 = np.full((num_unique_items, 1), mu_prior)

    # Sigma_0: Covariance matrix for prior (diagonal matrix with sigma_prior_squared on diagonal)
    Sigma_0 = np.diag(np.full(num_unique_items, sigma_prior_squared))

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
        print("Falling back to prior mean for all items and exiting.")
        # If estimation fails, it's better to exit as results would be unreliable.
        raise RuntimeError("Failed to estimate item volumes due to linear algebra error.")
    
    # Map posterior means back to item IDs
    estimated_volumes = {}
    for i, item_id in enumerate(all_item_ids):
        # Ensure estimated volumes are non-negative, as real volumes can't be negative.
        # Clip at a small positive value to avoid issues with zero volume in calculations like value/volume ratio.
        estimated_volumes[item_id] = max(0.01, mu_N[i, 0]) 

    print("\n  Estimated Volume (Posterior Mean) for each item:")
    for item_id, vol in estimated_volumes.items():
        print(f"    {item_id}: {vol:.2f} liters")
    print(f"  (Note: Items only in items.json, not packages.json, will have volumes based primarily on the prior mean.)")
    
    return estimated_volumes

# --- Part 2: Backtracking Algorithm for Knapsack ---

# Global variables to store the best solution found
# These are modified by the recursive backtracking function.
MAX_PRICE = -1.0
BEST_SELECTION_IDS = []

def solve_knapsack_backtrack(
    items_list: list, 
    capacity: float, 
    current_index: int, 
    current_volume: float, 
    current_price: float, 
    current_selection_indices: list
):
    """
    Recursive backtracking function to solve the 0/1 Knapsack problem.
    Updates the global MAX_PRICE and BEST_SELECTION_IDS.
    
    Args:
        items_list: List of dictionaries, each with 'id', 'name', 'price', 'volume'.
                    It's assumed 'volume' has been estimated and assigned.
        capacity: The maximum volume the backpack can hold.
        current_index: The index of the item currently being considered in items_list.
        current_volume: The accumulated volume of items already selected in the current path.
        current_price: The accumulated price of items already selected in the current path.
        current_selection_indices: List of original indices of items selected in the current path.
    """
    global MAX_PRICE, BEST_SELECTION_IDS

    # Pruning Condition 1: If current volume exceeds capacity, this path is invalid.
    if current_volume > capacity:
        return

    # Base Case: All items have been considered
    if current_index == len(items_list):
        # If this complete solution is better than the best found so far, update it.
        if current_price > MAX_PRICE:
            MAX_PRICE = current_price
            # Store the IDs of the selected items for the best solution.
            BEST_SELECTION_IDS = [items_list[i]['name'] for i in current_selection_indices]
        return

    # Pruning Condition 2 (Optimization): Upper Bound Check
    # This helps prune branches that cannot lead to a better solution.
    # We calculate an optimistic upper bound by summing prices of all remaining items.
    # If the current accumulated price plus all remaining prices is not enough to beat
    # the current MAX_PRICE, then this path is not worth exploring further.
    remaining_potential_price = sum(item['price'] for item in items_list[current_index:])
    if current_price + remaining_potential_price <= MAX_PRICE:
        return # Cannot beat the current max price, so prune this branch.

    # --- Recursive Steps ---

    # Option 1: Include the current item (items_list[current_index])
    item_to_include = items_list[current_index]
    solve_knapsack_backtrack(
        items_list,
        capacity,
        current_index + 1,  # Move to the next item
        current_volume + item_to_include['volume'],
        current_price + item_to_include['price'],
        current_selection_indices + [current_index] # Add current item's index to selection
    )

    # Option 2: Exclude the current item (items_list[current_index])
    solve_knapsack_backtrack(
        items_list,
        capacity,
        current_index + 1,  # Move to the next item
        current_volume,     # Volume remains unchanged
        current_price,      # Price remains unchanged
        current_selection_indices # Selection remains unchanged
    )


# --- Main Execution ---
if __name__ == "__main__":
    # Define backpack capacity
    BACKPACK_CAPACITY = 1000.0 # liters

    # --- Step 1: Load Data ---
    packages_file = 'packages.json'
    items_file = 'items.json'

    try:
        packages_data = load_json_data(packages_file)
        items_data = load_json_data(items_file)
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError) as e:
        # Exit if any loading or initial estimation error occurs
        print(f"Exiting due to data loading or initial estimation error.")
        exit()
    except Exception as e: # Catch any other unforeseen exceptions
        print(f"An unexpected error occurred during initial setup: {e}")
        exit()


    # --- Collect all unique item IDs ---
    # We need to estimate volumes for all items that might exist, either in packages or in items.json
    all_unique_item_ids = set()
    for item in items_data:
        all_unique_item_ids.add(item['name'])
    for pkg in packages_data:
        for item_id in pkg['items']:
            all_unique_item_ids.add(item_id)
    
    # Convert to a sorted list for consistent indexing in numpy arrays
    all_unique_item_ids_list = sorted(list(all_unique_item_ids))
    print(f"\nDiscovered {len(all_unique_item_ids_list)} unique item IDs across both datasets.")

    # --- Step 2: Bayesian Estimation of Individual Item Volumes ---
    try:
        estimated_volumes_dict = estimate_individual_item_volumes_bayesian(packages_data, all_unique_item_ids_list)
    except RuntimeError as e: # Catch the specific runtime error from estimation failure
        print(f"Exiting due to Bayesian estimation error: {e}")
        exit()

    # Assign the estimated volume to each item in the items_data list.
    for item in items_data:
        item_id = item['name']
        item['volume'] = estimated_volumes_dict.get(item_id, 0.01) # Default to 0.01 if not estimated (should not happen with all_unique_item_ids_list)
        
        # Also pre-calculate value_per_volume for sorting (useful for pruning)
        if item['volume'] == 0: # Avoid division by zero, treat 0 volume items as extremely high value density
            item['value_per_volume'] = float('inf') if item['price'] > 0 else 0
        else:
            item['value_per_volume'] = item['price'] / item['volume']
    
    # Sort items by value_per_volume in descending order for better pruning efficiency
    items_data_sorted_for_knapsack = sorted(items_data, key=lambda x: x.get('value_per_volume', 0), reverse=True)


    # # --- Step 3: Solve the Knapsack Problem using Backtracking ---
    # print(f"\n--- Solving Knapsack Problem ---")
    # print(f"Backpack Capacity: {BACKPACK_CAPACITY} liters")
    # print(f"Number of items to consider (from items.json): {len(items_data_sorted_for_knapsack)}")
    #
    # # Reset global variables before starting the backtracking process
    # MAX_PRICE = -1.0
    # BEST_SELECTION_IDS = []
    #
    # # Start the backtracking process from the first item (index 0).
    # solve_knapsack_backtrack(
    #     items_list=items_data_sorted_for_knapsack,
    #     capacity=BACKPACK_CAPACITY,
    #     current_index=0,
    #     current_volume=0.0,
    #     current_price=0.0,
    #     current_selection_indices=[] # Start with an empty selection
    # )
    #
    # print(f"\n--- Best Gift Choices for Sergey's Birthday ---")
    # print(f"Maximum Total Price Achieved: {MAX_PRICE:.2f} Euros")
    # print(f"Selected Item IDs: {BEST_SELECTION_IDS}")
    #
    # # Display details of the selected items and their total volume
    # selected_items_details = []
    # total_selected_volume = 0.0
    # total_selected_price = 0.0
    # for item_id in BEST_SELECTION_IDS:
    #     # Find the item in the original (unsorted) items_data for display purposes
    #     found_item = next((item for item in items_data if item['name'] == item_id), None)
    #     if found_item:
    #         selected_items_details.append(found_item)
    #         total_selected_volume += found_item['volume']
    #         total_selected_price += found_item['price']
    #
    # print("\nDetails of Selected Items:")
    # if not selected_items_details:
    #     print("  No items selected (perhaps capacity too small or no profitable items).")
    # for item in selected_items_details:
    #     print(f"  - Name: {item['name']}, Price: {item['price']:.2f}€, Est. Volume: {item['volume']:.2f}L (ID: {item['name']})")
    # print(f"\nTotal Volume of Selected Items: {total_selected_volume:.2f} liters (Max Capacity: {BACKPACK_CAPACITY}L)")
    # print(f"Total Price of Selected Items (for verification): {total_selected_price:.2f} Euros")


    # === BENCHMARKS ===
    print(f"\n--- Benchmarking Algorithms ---")

    # # Backtracking
    # MAX_PRICE = -1.0
    # BEST_SELECTION_IDS = []
    # start_bt = time.perf_counter()
    # solve_knapsack_backtrack(
    #     items_list=items_data_sorted_for_knapsack,
    #     capacity=BACKPACK_CAPACITY,
    #     current_index=0,
    #     current_volume=0.0,
    #     current_price=0.0,
    #     current_selection_indices=[]
    # )
    # end_bt = time.perf_counter()
    # print(f"\nBacktracking:")
    # print(f"  Max Price: {MAX_PRICE:.2f} EUR")
    # print(f"  Time: {end_bt - start_bt:.3f} sec")
    # print(f"  Items: {BEST_SELECTION_IDS}")
    # volume_bt = calculate_total_volume(items_data, BEST_SELECTION_IDS)
    # print(f"  Total Volume: {volume_bt:.2f}L / {BACKPACK_CAPACITY}L")

    # Dynamic Programming with discretized volumes
    tracemalloc.start()
    start_dp = time.perf_counter()

    dp_result = solve_knapsack_dp_discretized(
        items_list=items_data_sorted_for_knapsack,
        capacity=BACKPACK_CAPACITY,
        precision=2
    )

    end_dp = time.perf_counter()
    current_mem_dp, peak_mem_dp = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nDynamic Programming (discretized):")
    print(f"  Max Price: {dp_result['total_price']:.2f} EUR")
    print(f"  Time: {end_dp - start_dp:.3f} sec")
    print(f"  Items: {dp_result['selected_items']}")
    volume_dp = calculate_total_volume(items_data, dp_result['selected_items'])
    print(f"  Total Volume: {volume_dp:.2f}L / {BACKPACK_CAPACITY}L")
    print(f"  Memory Usage: Current = {current_mem_dp / 1024:.2f} KB; Peak = {peak_mem_dp / 1024:.2f} KB")

    # Dynamic Programming with discretized volumes (memory optimized)
    tracemalloc.start()
    start_dp_opt = time.perf_counter()

    dp_opt_result = solve_knapsack_dp_discretized_memory_optimized(
        items_list=items_data_sorted_for_knapsack,
        capacity=BACKPACK_CAPACITY,
        precision=2
    )

    end_dp_opt = time.perf_counter()
    current_mem_opt, peak_mem_opt = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nDynamic Programming (optimized):")
    print(f"  Max Price: {dp_opt_result['total_price']:.2f} EUR")
    print(f"  Time: {end_dp_opt - start_dp_opt:.3f} sec")
    print(f"  Items: {dp_opt_result['selected_items']}")
    volume_dp_opt = calculate_total_volume(items_data, dp_opt_result['selected_items'])
    print(f"  Total Volume: {volume_dp_opt:.2f}L / {BACKPACK_CAPACITY}L")
    print(f"  Memory Usage: Current = {current_mem_opt / 1024:.2f} KB; Peak = {peak_mem_opt / 1024:.2f} KB")
