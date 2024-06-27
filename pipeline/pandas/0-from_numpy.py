#!/user/bin/env python3
import pandas as pd
import numpy as np

def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray.

    Args:
        array: np.ndarray from which to create the DataFrame.

    Returns:
        pd.DataFrame: Newly created DataFrame with columns labeled alphabetically.
    """
    # Generate column labels: 'A', 'B', 'C', ..., 'Z'
    num_columns = array.shape[1]
    columns = [chr(i) for i in range(65, 65 + num_columns)]
    
    # Create the DataFrame
    df = pd.DataFrame(array, columns=columns)
    return df

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    A = np.random.randn(5, 8)
    print(from_numpy(A))
    B = np.random.randn(9, 3)
    print(from_numpy(B))
