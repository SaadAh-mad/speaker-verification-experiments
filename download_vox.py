from datasets import load_dataset

print("Downloading dataset...")

dataset = load_dataset("s3prl/mini_voxceleb1")

print(dataset)