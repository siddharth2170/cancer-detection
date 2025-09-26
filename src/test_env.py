# PyTorch
import torch
print("Torch version:", torch.__version__)

# Torchvision
import torchvision
print("Torchvision version:", torchvision.__version__)

# Matplotlib
import matplotlib
print("Matplotlib version:", matplotlib.__version__)

# Scikit-learn
import sklearn
print("Scikit-learn version:", sklearn.__version__)

# OpenCV
import cv2
print("OpenCV version:", cv2.__version__)

# Jupyter (just check import)
import notebook
print("Jupyter import works!")

# Grad-CAM (local package)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
print("GradCAM import works!")
