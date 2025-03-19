import torch

num_bodies = 5
max_contacts = 3
indices = torch.randint(0, num_bodies, (num_bodies, max_contacts))
contact_mask = torch.zeros((num_bodies, max_contacts), dtype=torch.bool)
contact_mask[1, 2] = True
contact_mask[3, 1] = True

print(f"indices: {indices}")
print(f"contact_mask: {contact_mask}")

result = indices[contact_mask]
print(f"result: {result}")

for b in range(num_bodies):
    print(f"Filtering contact indices for body {b}: {indices[b, contact_mask[b]]}")

