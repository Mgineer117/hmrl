# Sample list of dictionaries
list_of_dicts = [
    {'a': 10, 'b': 20, 'c': 30},
    {'a': 40, 'b': 50, 'c': 60},
    {'a': 70, 'b': 80, 'c': 90},
]

# Initialize a dictionary to store the sums
sums = {}
# Initialize a dictionary to store the counts
counts = {}

# Iterate over each dictionary in the list
for d in list_of_dicts:
    for key, value in d.items():
        if key in sums:
            sums[key] += value
            counts[key] += 1
        else:
            sums[key] = value
            counts[key] = 1

# Calculate the averages
averages = {key: sums[key] / counts[key] for key in sums}

print(averages)  # Output: {'a': 40.0, 'b': 50.0, 'c': 60.0}
