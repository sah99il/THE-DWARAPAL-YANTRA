import os
import subprocess
import sys

def run_command(command):
    """Executes a shell command and prints its output."""
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}
{e}")
        sys.exit(1)

def install_dependencies():
    """Installs the required Python packages."""
    # Ensure pip is available
    run_command([sys.executable, "-m", "ensurepip"])

    # Attempt to install onnxruntime-gpu
    print("Attempting to install onnxruntime-gpu...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu"], check=True, shell=True)
        print("onnxruntime-gpu installed successfully.")
    except subprocess.CalledProcessError:
        print("onnxruntime-gpu installation failed. Falling back to CPU version.")
        run_command([sys.executable, "-m", "pip", "install", "onnxruntime"])

    # Install other dependencies
    dependencies = [
        "insightface==0.7.3",
        "opencv-python",
        "mediapipe",
        "scipy",
        "numpy<2"
    ]
    print("Installing other dependencies...")
    run_command([sys.executable, "-m", "pip", "install"] + dependencies)

def verify_and_create_structure():
    """Verifies and creates the necessary directory and file structure."""
    print("Verifying directory structure...")
    
    # Directories to create
    dirs_to_create = [
        "models/buffalo_l",
        "database",
        "data/raw",
        "data/attacked"
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"Directory '{d}' ensured.")

    # Files to check
    files_to_check = {
        "models/buffalo_l/w600k_r50.onnx": "the face recognition model",
        "models/buffalo_l/det_10g.onnx": "the face detection model",
        "database/dwarapal.db": "the user database"
    }
    
    print("\nChecking for essential files...")
    all_files_found = True
    for file_path, description in files_to_check.items():
        if not os.path.exists(file_path):
            print(f"  [! ] Missing: {file_path} ({description})")
            all_files_found = False
        else:
            print(f"  [+] Found: {file_path}")
            
    if not all_files_found:
        print("\nWarning: Some essential files are missing. Please ensure they are downloaded or created.")
    else:
        print("\nAll essential files found.")


if __name__ == "__main__":
    install_dependencies()
    verify_and_create_structure()
    print("\nSetup complete. Please review any warnings above.")
