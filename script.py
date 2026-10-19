from pathlib import Path

DIR = Path(__file__).resolve().parent

FILE_PATH = DIR / "/06_CLUSTERING"
print(FILE_PATH)

files = [
    "01_concept.md",
    "02_implementation.ipynb"
]

for folder in FILE_PATH.glob("*/"):
    for file in files:

        if folder.is_dir():
            (folder /f"{file}").touch()





