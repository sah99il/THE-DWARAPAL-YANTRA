import os
import subprocess
import sys

def run_command(command):
    # Run a shell command and stop if it fails
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        sys.exit(1)

def install_dependencies():
    # Make sure pip is available
    run_command([sys.executable, "-m", "ensurepip"])

    # Try GPU version of ONNX Runtime first
    print("Attempting to install onnxruntime-gpu...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "onnxruntime-gpu"],
            check=True
        )
        print("onnxruntime-gpu installed.")
    except subprocess.CalledProcessError:
        print("GPU install failed. Installing CPU version.")
        run_command([sys.executable, "-m", "pip", "install", "onnxruntime"])

    # Core project dependencies
    dependencies = [
        "insightface==0.7.3",
        "opencv-python",
        "mediapipe",
        "scipy"
    ]

    print("Installing remaining dependencies...")
    run_command([sys.executable, "-m", "pip", "install"] + dependencies)

def verify_and_create_structure():
    # Ensure required folders exist
    print("Verifying directory structure...")

    dirs_to_create = [
        "models/buffalo_l",
        "database",
        "data/raw",
        "data/attacked"
    ]

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"Ensured directory: {d}")

    # Check for important files
    files_to_check = {
        "database/dwarapal.db": "SQLite user database"
    }

    print("\nChecking essential files...")
    for path, desc in files_to_check.items():
        if os.path.exists(path):
            print(f"[+] Found: {path}")
        else:
            print(f"[!] Missing: {path} ({desc})")

if __name__ == "__main__":
    install_dependencies()
    verify_and_create_structure()
    print("\nSetup complete.")