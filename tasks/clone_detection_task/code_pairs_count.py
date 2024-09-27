# Initialize counters
clone_1_count = 0
clone_0_count = 0

# Open and read the text file
with open("tasks/clone_detection_task/code_snippets_pairs.txt", "r") as file:
    lines = file.readlines()
    
    # Loop through lines and count occurrences of 'is clone' values
    for line in lines:
        line = line.strip()  # Remove any whitespace
        if line == "1":
            clone_1_count += 1
        elif line == "0":
            clone_0_count += 1

# Print the counts
print(f"Number of 'is clone' 1: {clone_1_count}")
print(f"Number of 'is clone' 0: {clone_0_count}")