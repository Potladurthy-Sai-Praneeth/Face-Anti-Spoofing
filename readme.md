

# Deep Learning-based Anti-Spoofing

## Introduction
With the increasing adoption of deep learning-based facial recognition systems, the need for robust anti-spoofing techniques has become crucial. These systems are vulnerable to spoofing attacks such as paper-cut masks and video replays. To counteract this, we propose a 2-stage framework inspired by the UCDCN architecture for face anti-spoofing, which utilizes auxiliary depth maps to enhance liveliness detection. The framework consists of:
- Generating a 2D depth image when a live person faces the camera.
- Authenticating a live user.

## Methodology
### Dataset Preparation
A supervised learning approach is used to build the system, requiring a [dataset](https://www.kaggle.com/datasets/tapakah68/anti-spoofing) containing labeled images of both live users and spoof attacks. The Anti-Spoofing Dataset, containing videos from multiple users, is processed into frames (images) and consolidated into a single dataset.

### Depth Map Generation
The system generates 2D depth maps from RGB images only when a live person is detected. A blank image is generated when spoof attacks (e.g., paper-cut masks, video replays) are detected. Transfer learning is employed by fine-tuning the DepthAnythingV2 model.

- Most layers in the model are frozen, while the final prediction head is trained on the created dataset.
- For live user images (label 1), ground truth depth maps are obtained from a larger version of DepthAnythingV2.
- For spoof attack images (label 0), blank images are used as ground truth to prevent depth information generation.

A classifier is trained on top of the depth map model for binary classification using two central difference convolutions. The depth map model and classifier are trained using variations of loss functions inspired by the referenced research paper.

### Face Recognition
Once depth map generation is validated, a facial recognition system authenticates users. Since multiple users may share a vehicle, a multi-user authentication system is implemented.

- User face embeddings are obtained using a One-Shot learning framework and stored.
- After verifying liveliness, the captured frame is compared against saved embeddings for authentication.

### Model Quantization
Since deployment is on a Raspberry Pi, which has limited compute and memory resources, model quantization is necessary. Directly using the PyTorch model files (~750MB) is infeasible as it requires at least 1.5GB of space including dependencies.

- Model quantization reduces floating-point weights to lower-bit precision values, significantly reducing the model size.
- The ONNX quantization library is used for layer fusion, eliminating repetitive operations and enhancing inference speed.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- PyTorch
- ONNX
- OpenCV
- NumPy
- Scikit-learn

### Installation Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/deep-anti-spoofing.git
   cd deep-anti-spoofing
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
    1) Generate facial embeddings of users. The script **authenticate.py** is designed to automate the process of user facial embedding generation. To generate embeddings we need to pass the folder containing user images as arguments to the file.

    ` python authenticate.py --generate '<path to images>' `

    The script generates the embeddings into the same folder where the 'authenticate.py' is located.

    2) Once the facial embeddings are generated run the same script without any arguments for multi-user authentication.
    
        `python authenticate.py`  
        
## Acknowledgments
This project is inspired by the research paper: "UCDCN: A Nested Architecture Based on Central Difference Convolution for Face Anti-Spoofing" published in Complex & Intelligent Systems.


