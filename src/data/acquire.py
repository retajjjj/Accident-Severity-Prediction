import kagglehub

def download_dataset():
    """Download the dataset from Kaggle Hub."""
    path = kagglehub.dataset_download("tsiaras/uk-road-safety-accidents-and-vehicles")
    print("Path to dataset files:", path)
    return path

if __name__ == "__main__":
    download_dataset()
