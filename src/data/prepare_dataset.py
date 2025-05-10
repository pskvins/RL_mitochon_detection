import os
import zipfile

def prepare_dataset(zip_path="data/datasets.zip", extract_to="datasets"):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"No such file found: {zip_path}")

    if not os.path.exists(extract_to):
        print(f"Unzipping: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Unzip completed: {extract_to}")
    else:
        print(f"Already unzipped: {extract_to}")

    yaml_path = os.path.join(extract_to, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No such file found: {yaml_path}")

    return yaml_path
