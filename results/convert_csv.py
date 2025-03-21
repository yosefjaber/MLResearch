import glob
import csv
import os

# Define the output CSV file name
output_csv = 'results.csv'

# List to store each record as a dictionary
records = []

# Use glob to find all .txt files in the current directory
for file_path in glob.glob("*.txt"):
    with open(file_path, 'r') as file:
        # Create a dictionary to store the values from the file
        record = {}
        for line in file:
            # Remove any surrounding whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue
            # Split the line into key and value using ':' as a separator
            if ': ' in line:
                key, value = line.split(': ', 1)
                record[key] = value
        # Append the record if it contains all the required keys
        if all(key in record for key in ['Model', 'MSE', 'R^2', 'MAE', 'CV', 'Count']):
            records.append(record)
        else:
            print(f"Warning: File {os.path.basename(file_path)} is missing some keys and will be skipped.")

# Define the order of columns for the CSV
fieldnames = ['Model', 'MSE', 'R^2', 'MAE', 'CV', 'Count']

# Write all records into the CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)

print(f"CSV file '{output_csv}' has been created with {len(records)} records.")
