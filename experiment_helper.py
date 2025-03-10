import os
import re
import sys

def extract_mse_from_file(file_path):
    try:
        with open(f"{file_path}", 'r') as file:
            content = file.read()
            
            # Use regex to find the MSE value
            mse_match = re.search(r'MSE:\s*(\d+\.\d+)', content)
            
            if mse_match:
                return float(mse_match.group(1))
            else:
                print(f"No MSE value found in {file_path}")
                return -10
    except Exception as e:
        #print(f"File doesn't Exist - {file_path}: {e}")
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
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for the Count line
        count_match = re.search(r'Count:\s*(\d+)', content)
        if count_match:
            return int(count_match.group(1))
        else:
            return -10
    except Exception as e:
        #print(f"File doesn't Exist - {file_path}: {e}", file=sys.stderr)
        return -1
        
def increment_count_in_file(file_path):
    try:
        # Read the file's content
        with open(file_path, 'r') as file:
            content = file.read()

        # Find the Count line using a regex that captures the prefix and the number
        count_match = re.search(r'(Count:\s*)(\d+)', content)
        if count_match:
            prefix = count_match.group(1)  # "Count:" plus any whitespace
            current_count = int(count_match.group(2))  # the current count as an integer
            new_count = current_count + 1  # increment the count

            # Create a new content by replacing the old count with the new count
            new_content = re.sub(r'(Count:\s*)(\d+)', f"{prefix}{new_count}", content)

            # Write the new content back to the file
            with open(file_path, 'w') as file:
                file.write(new_content)

            print(f"Count updated from {current_count} to {new_count}")
            return new_count
        else:
            print("No Count value found in the file.")
            return None
    except Exception as e:
        print(f"Error reading or writing file: {e}")
        return None
    