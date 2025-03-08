import os


# Get the current script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))

print("path:", base_dir)


parent_dir = os.path.dirname(base_dir)


model_path = os.environ.get("W2V_MODEL_PATH")


print("File path:", model_path)

word2vec_model_path = os.path.join(parent_dir, "GoogleNews-vectors-negative300.bin")

