import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Cargar un modelo preentrenado para la segmentación panóptica
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Cargar una imagen real
image_path = 'frame_0109.jpg'  # Cambia esto a la ruta de tu imagen
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Realizar la segmentación panóptica
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    prediction = model(image_tensor)

print(prediction[0])
# Obtener las máscaras de segmentación y las clases
masks = prediction[0]['masks'].numpy()
labels = prediction[0]['labels'].numpy()
# Obtener la segmentación semántica
semantic_mask = prediction[0]['pred_semantic'].numpy()
cv2.imshow('semantic_mask', semantic_mask)
# Obtener la segmentación de instancias
instances = prediction[0]['instances']
cv2.imshow('instances ', instances )

# Renderizar las máscaras en la imagen original
for i in range(masks.shape[0]):
    mask = masks[i, 0]
    label = labels[i]
    color = np.random.rand(3) * 255  # Color aleatorio para cada clase
    mask = (mask > 0.5)  # Umbral para mejorar la visualización
    mask = np.uint8(mask * 255)  # Convertir a tipo uint8
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convertir a una imagen en escala de grises
    image = cv2.addWeighted(image, 1, mask, 0.5, 0)

# Mostrar la imagen resultante
cv2.imshow('Panoptic Segmentation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

