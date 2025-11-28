
import numpy as np

# Define the path to your file
file_path = 'optimal_control.npy'

# Load the file
try:
    data = np.load(r"file_path")
    
    # Now 'data' is a NumPy array containing the file's contents
    print("Successfully loaded the array!")
    
    # You can inspect the array
    print("Array shape:", data.shape)
    print("Array data type:", data.dtype)
    
    # Print the first 5 elements (or the whole thing if it's small)
    #if data.size > 5:
        #print("First 5 elements:", data.flatten()[:5])
    #else:
    print("Array data:", data)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")