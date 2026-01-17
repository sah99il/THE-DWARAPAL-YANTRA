import requests
import zipfile
import os

def download_and_extract_models():
    """
    Downloads and extracts the InsightFace buffalo_l models.
    """
    # URL of the model zip file
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    
    # Path to the models directory
    models_dir = "models"
    
    # Path to the zip file
    zip_path = os.path.join(models_dir, "buffalo_l.zip")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    print(f"Downloading models from {url}...")
    
    try:
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Models downloaded to {zip_path}")
        
        # Extract the zip file
        print(f"Extracting models to {models_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
            
        print("Models extracted successfully.")
        
        # Clean up the zip file
        os.remove(zip_path)
        print("Cleaned up downloaded zip file.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading models: {e}")
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_and_extract_models()
