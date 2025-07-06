# sergeys_birthday
I wrote an algorithm, that uses baysian estimation to estimate the volumes of the individual items. 
Then we use backtracking to solve the knapsack problem. 
We assume sergey doesnt want items twice!

Below you can find the solution the algo came up with:

Discovered 60 unique item IDs across both datasets.
--- Bayesian Estimation of Individual Item Volumes ---
  Number of unique items to estimate: 60
  Number of package observations: 1000
  Assumed measurement error variance (sigma_epsilon^2): 2.0
  Prior for each item volume: N(mean=14.19, variance=50.0)

  Estimated Volume (Posterior Mean) for each item:
    A1: 22.01 liters
    A10: 14.54 liters
    A11: 22.77 liters
    A12: 14.96 liters
    A13: 21.54 liters
    A14: 29.66 liters
    A15: 11.04 liters
    A16: 15.96 liters
    A17: 6.87 liters
    A18: 8.36 liters
    A19: 15.18 liters
    A2: 14.46 liters
    A20: 22.10 liters
    A21: 26.43 liters
    A22: 25.79 liters
    A23: 7.28 liters
    A24: 25.77 liters
    A25: 9.48 liters
    A26: 20.58 liters
    A27: 7.52 liters
    A28: 6.84 liters
    A29: 20.85 liters
    A3: 8.46 liters
    A30: 19.33 liters
    A31: 6.88 liters
    A32: 0.93 liters
    A33: 5.95 liters
    A34: 13.25 liters
    A35: 4.98 liters
    A36: 25.11 liters
    A37: 12.50 liters
    A38: 9.90 liters
    A39: 1.56 liters
    A4: 23.95 liters
    A40: 10.54 liters
    A41: 23.03 liters
    A42: 11.57 liters
    A43: 5.01 liters
    A44: 4.82 liters
    A45: 22.43 liters
    A46: 17.58 liters
    A47: 26.03 liters
    A48: 7.22 liters
    A49: 11.18 liters
    A5: 16.61 liters
    A50: 11.39 liters
    A51: 26.98 liters
    A52: 23.00 liters
    A53: 5.63 liters
    A54: 11.65 liters
    A55: 16.51 liters
    A56: 10.44 liters
    A57: 18.70 liters
    A58: 25.48 liters
    A59: 16.19 liters
    A6: 0.39 liters
    A60: 12.36 liters
    A7: 8.84 liters
    A8: 3.55 liters
    A9: 0.97 liters
  (Note: Items only in items.json, not packages.json, will have volumes based primarily on the prior mean.)

--- Solving Knapsack Problem ---
Backpack Capacity: 40.0 liters
Number of items to consider (from items.json): 60

--- Best Gift Choices for Sergey's Birthday ---
Maximum Total Price Achieved: 741.00 Euros
Selected Item IDs: ['A6', 'A32', 'A9', 'A39', 'A35', 'A44', 'A8', 'A38', 'A48', 'A53']

Details of Selected Items:
  - Name: A6, Price: 109.00€, Est. Volume: 0.39L (ID: A6)
  - Name: A32, Price: 80.00€, Est. Volume: 0.93L (ID: A32)
  - Name: A9, Price: 64.00€, Est. Volume: 0.97L (ID: A9)
  - Name: A39, Price: 31.00€, Est. Volume: 1.56L (ID: A39)
  - Name: A35, Price: 86.00€, Est. Volume: 4.98L (ID: A35)
  - Name: A44, Price: 76.00€, Est. Volume: 4.82L (ID: A44)
  - Name: A8, Price: 54.00€, Est. Volume: 3.55L (ID: A8)
  - Name: A38, Price: 116.00€, Est. Volume: 9.90L (ID: A38)
  - Name: A48, Price: 81.00€, Est. Volume: 7.22L (ID: A48)
  - Name: A53, Price: 44.00€, Est. Volume: 5.63L (ID: A53)

Total Volume of Selected Items: 39.95 liters (Max Capacity: 40.0L)
Total Price of Selected Items (for verification): 741.00 Euros

---

## UPDATED

I forked this repository to explore how the solution could be improved when the number of items increases.

The backtracking algorithm works well for small and medium input sizes, but becomes too slow on large ones. So I added an alternative approach using **dynamic programming**. It's much faster, but it only works with integers — so I had to **convert decimal volumes to integers** by discretizing them (e.g., multiplying by 100).

Here’s a quick benchmark:

| Algorithm              | Max Price (EUR) | Time (sec) | Total Volume Used (L) |
|------------------------|------------------|-------------|-------------------------|
| Backtracking           | 751.00           | 70.461      | 39.97 / 40.0            |
| Dynamic Programming    | 757.00           | 0.010       | 40.00 / 40.0            |

My version does **not** change the original logic or data. I only added a separate DP implementation (`dp_solver.py`) and benchmarked both methods on the same input. This experiment is just to see how scalable the solution can become.
Total Volume: 40.00L / 40.0L
