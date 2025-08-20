#!/usr/bin/env python3
"""
Script to generate a large CSV file with approximately 2^28 rows
using the lorem Python module to generate Lorem ipsum text.
"""

import csv
import lorem

# Target number of rows: 2^28 = 268,435,456
target_rows = 2**28

print(f"Target rows: {target_rows:,}")

# Generate Lorem ipsum text by characters
print("Generating Lorem ipsum text...")
lorem_text = lorem.text()
print(f"Generated text length: {len(lorem_text)} characters")

# Convert to list of characters
lorem_chars = list(lorem_text)
lorem_length = len(lorem_chars)

print(f"Lorem ipsum text has {lorem_length} characters")

# Calculate how many complete repetitions we need
repetitions_needed = target_rows // lorem_length
remaining_chars = target_rows % lorem_length

print(f"Need {repetitions_needed:,} complete repetitions")
print(f"Plus {remaining_chars:,} additional characters")
print(
    f"Total rows will be: {repetitions_needed * lorem_length + remaining_chars:,}")

# Generate the CSV file
print("Writing CSV file...")
with open('lorem_ipsum.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header
    writer.writerow(['character'])

    # Write complete repetitions
    for rep in range(repetitions_needed):
        for char in lorem_chars:
            writer.writerow([char])

        # Progress indicator
        if rep % 1000 == 0:
            print(f"Completed {rep:,} repetitions...")

    # Write remaining characters
    for i in range(remaining_chars):
        writer.writerow([lorem_chars[i]])

    print("CSV file generation completed!")

print(f"Generated {target_rows:,} rows in lorem_ipsum.csv")
print("File size will be approximately {:.1f} MB".format(
    target_rows * 2 / 1024 / 1024))
