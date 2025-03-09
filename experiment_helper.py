import os
import re
import sys

def extract_mse_from_file(file_path):
    try:
        with open(f"{file_path}.txt", 'r') as file:
            content = file.read()
            
            # Use regex to find the MSE value
            mse_match = re.search(r'MSE:\s*(\d+\.\d+)', content)
            
            if mse_match:
                return float(mse_match.group(1))
            else:
                print(f"No MSE value found in {file_path}")
                return -1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return -1

def write_lines(model_name, mse, r2, mae, cv, count):
    lines = [
        f"Model: {model_name}\n",
        f"MSE: {mse}\n",
        f"R^2: {r2}\n",
        f"MAE: {mae}\n",
        f"CV: {cv}\n",
        f"Count: {count}"
    ]
    
    with open(f"results/{model_name}.txt", "w") as file:
        file.writelines(lines)
        
def get_count_from_file(file_path):
    """
    Extract the Count value from a file with the model metrics format.
    Returns the count as an integer if found, None otherwise.
    """
    print(file_path)
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for the Count line
        count_match = re.search(r'Count:\s*(\d+)', content)
        if count_match:
            return int(count_match.group(1))
        else:
            return -1
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return -1
    