import os
from pathlib import Path


PATH = Path(__file__).resolve().parents[0]

print(PATH)

for folder in PATH.iterdir():
    if folder.is_dir():
        (folder / ".gitkeep").touch()




# for folder in PATH.iterdir():
#     if folder.is_dir():
#         folder.rename(folder.with_name(folder.name.lower()))
#     print(folder.name)

#         # folder.name

# subdir = [
#     "01_basics", 
#     "02_layers_model", 
#     "03_datasets", 
#     "04_training",
#     "05_callbacks", 
#     "06_perprocessing",
#     "07_gpu_acceleration",
#     "08_model_saving_loading"
# ]

# for name in subdir:
#     (PATH / name).mkdir(exist_ok=True)
#     print(name)

# DL_subdirs = [
#     "fundamentals",
#     "cnns",
#     "rnns",
#     "transformers",
#     "tranfer_learning"

# ]

# for name in DL_subdirs:
#     (PATH / name).mkdir(exist_ok=True)
#     print(name)