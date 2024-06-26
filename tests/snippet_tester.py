import torch

# Example tensor, assuming it's sorted by column at index 6
classifier_batch = torch.tensor([
    [1, 2, 3, 4, 5, 6, 0],
    [1, 2, 3, 4, 5, 6, 0],
    [1, 2, 3, 4, 5, 6, 1],
    [1, 2, 3, 4, 5, 6, 2],
    [1, 2, 3, 4, 5, 6, 2],
    [1, 2, 3, 4, 5, 6, 3]
], dtype=torch.float32)

length = classifier_batch.size(0)
classifier_batch_list = []
start_index = 0
running_k = classifier_batch[0, 6].item()  # Initialize to the first element's value in column 6

# Iterate through the sorted tensor
for i in range(1, length):
    if round(classifier_batch[i, 6].item()) != running_k:
        classifier_batch_list.append(classifier_batch[start_index:i, :])
        start_index = i
        running_k = classifier_batch[i, 6].item()  # Update to the current new value

# Append the final segment
classifier_batch_list.append(classifier_batch[start_index:length, :])

# Display the results
for idx, segment in enumerate(classifier_batch_list):
    print(f"Segment {idx}:")
    print(segment)
