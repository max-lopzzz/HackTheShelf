from ultralytics import YOLO

# Cargamos el modelo YOLO preentrenado (puedes usar uno personalizado también)
model = YOLO("model000.pt")  # o tu modelo entrenado en productos específicos

def detect_objects(image_path):
    results = model(image_path)
    detected = []

    for result in results:
        boxes = result.boxes
        names = model.names  # Diccionario {0: 'person', 1: 'bicycle', ...}
        for box in boxes:
            class_id = int(box.cls)
            label = names[class_id]
            confidence = float(box.conf)
            detected.append({
                "label": label,
                "confidence": confidence,
                "bbox": box.xyxy.tolist()  # Coordenadas del bounding box
            })
        result.show()

    return detected