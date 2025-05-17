# Librerias
from ultralytics import YOLO
import cv2
import numpy as np

# Definr los nombres de cada clase
class_names = {
  
}

# función main
def main():
  # Obtener la imagen y el modelo
  model_path = ''
  img_path = ''
  
  frame = cv2.imread(img_path)
  model = YOLO(model_path)
  
  # Procesar la imagen mediante el modelo
  results = model(frame)
  
  data = {'x': [],
          'y': [],
          'class': [],
          'conf': []}

  # Para cada resultado, se va a llevar a cabo una representación visual de los resultados.

  for result in results:
    boxes = result.boxes 

    # Para cada detección
    for box in boxes:
      # Obtener coordenadas (xyxy format)
      xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)
      
      data['x'].append(float(xmax+xmin)/2)
      data['y'].append(float(ymax+ymin)/2)
      data['class'].append(class_names[int(box.cls[0])])
      data['conf'].append(box.conf[0].item())
      
      # Definir el polígono
      pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], np.int32)
      pts = pts.reshape((-1, 1, 2))  # Reshape for OpenCV polylines

      # Dibujar el poligono
      cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

      # Añadir catagorías y datos de entrenamiento
      label = f"Class {class_names[int(box.cls[0])]} {box.conf[0]:.2f}"  # Assuming box.cls contains the class and box.conf contains the confidence score
      cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar la imagen y los polígonos
    cv2.imshow("Detections", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()