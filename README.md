# Dwarapal Yantra: Real-Time Facial Recognition & Anti-Spoofing System

Dwarapal Yantra is a computer vision application that not only recognizes faces in real-time but also ensures the face belongs to a living person (liveness detection). I built this project to solve the common problem of facial recognition systems being easily fooled by someone holding up a printed photo or a video on a phone screen.

## Features

*   **Real-Time Identity Verification:** Uses deep learning embeddings to instantly recognize enrolled users via webcam.
*   **Anti-Spoofing / Liveness Detection:** Analyzes texture, temporal motion, and rPPG (heartbeat signals from skin color fluctuations) to detect if a physical face is in front of the camera, completely blocking photo and video replay attacks.
*   **Web Interface:** A sleek, fully functional Streamlit frontend for enrolling new users and verifying identities seamlessly.
*   **Command Line Tools:** Lightweight CLI scripts for quick enrollment and debugging.

## Project Structure

*   `core/` - The main engine containing the face identity models and liveness algorithms.
*   `ui/app.py` - The Streamlit web interface.
*   `train.py` & `train_liveness.py` - Scripts used to train the underlying neural networks.
*   `enroll.py` & `verify.py` - Command-line tools for direct webcam interaction.

## Setup & Installation

1. Clone the repository and navigate into the directory:
   ```bash
   git clone https://github.com/sah99il/THE-DWARAPAL-YANTRA.git
   cd THE-DWARAPAL-YANTRA
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Running the Web Interface (Recommended)**
```bash
python -m streamlit run ui/app.py
```
This will open a local web server where you can use your webcam to enroll your face and test the anti-spoofing recognition.

**Command Line Usage**
To enroll a new user via the terminal:
```bash
python enroll.py --name "YourName"
```
To verify identities and test liveness:
```bash
python verify.py
```
