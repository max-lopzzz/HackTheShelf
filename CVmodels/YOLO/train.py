# Entrenar un modelo, yolo11n en este caso

# imporotar el modelo
from ultralytics import YOLO

# definir una función maín
def main():
  # Definir el modelo
  model = YOLO('yolo11n,pt')

  # Entrenar el modelo
  model.train(data="YOLO/configs/config5.yaml", epochs=25, batch=16, device='cpu')
  
  # Guardar el modelo
  #model_path = ''
  #model.save(model_path)
  #print('Modelo guaardado en', model_path)
  
  # Metricas
  #metrics = model.val()
  
if __name__ == '__main__':
  main()