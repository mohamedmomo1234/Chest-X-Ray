import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import cv2
import numpy as np

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    
    # one output in last layer
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

#  Grad-CAM 
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=0):
        self.model.eval()
        output = self.model(x)
            
        self.model.zero_grad()
        class_score = output[0, 0]
        class_score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()

def generate_heatmap_image(original_image_pil, cam, alpha=0.4):
    img_np = np.array(original_image_pil.resize((224, 224)))
    h, w, _ = img_np.shape
    cam_resized = cv2.resize(cam, (w, h))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img
