from ultralytics import YOLO
import numpy as np

model = YOLO("model005.pt")

def remove_overlapping_objects(shelf):
    """
    Dado un shelf (lista de objetos ordenados por center_x),
    elimina los objetos solapados que probablemente representan la misma detección.
    """
    i = 0
    while i < len(shelf) - 1:
        obj = shelf[i]
        next_obj = shelf[i + 1]

        # Obtener bounding box y centro del siguiente objeto
        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = next_obj["center_x"], next_obj["center_y"]

        # Verificar si el centro del siguiente está dentro del bbox actual
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            # Hay solapamiento → eliminar el de menor confianza
            if obj["confidence"] >= next_obj["confidence"]:
                # Mantener obj, borrar next_obj
                shelf.pop(i + 1)
            else:
                # Mantener next_obj, borrar obj
                shelf.pop(i)
            # No incrementamos i porque ahora i apunta al siguiente elemento
        else:
            # No hay solapamiento → avanzar al siguiente par
            i += 1

    return shelf

def detect_objects(image_path):
    results = model(image_path)
    detected = []

    for result in results:
        boxes = result.boxes
        names = model.names  # class IDs to labels

        for box in boxes:
            class_id = int(box.cls)
            label = names[class_id]
            confidence = float(box.conf)
            bbox = box.xyxy.tolist()[0]  # [x1, y1, x2, y2]

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            length_x = bbox[2] - bbox[0]
            length_y = bbox[3] - bbox[1]

            detected.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "center_x": center_x,
                "center_y": center_y,
                "length_x": length_x,
                "lenght_y": length_y
            })

        # --- Shelf grouping logic ---
        shelf_threshold = 100  # vertical distance threshold to separate shelves
        sorted_by_y = sorted(detected, key=lambda obj: obj["center_y"])

        shelves = []
        current_shelf = [sorted_by_y[0]]

        for obj in sorted_by_y[1:]:
            last_obj = current_shelf[-1]
            shelf_threshold = last_obj["lenght_y"]
            if abs(obj["center_y"] - last_obj["center_y"]) <= shelf_threshold:
                current_shelf.append(obj)
            else:
                shelves.append(current_shelf)
                current_shelf = [obj]
        shelves.append(current_shelf)  # add last shelf

        # Sort each shelf left to right and eliminate overlapping objects
        for i, shelf in enumerate(shelves):
            shelves[i] = sorted(shelf, key=lambda obj: obj["center_x"])
            # Aplicar filtro de eliminación de objetos solapados
            shelves[i] = remove_overlapping_objects(shelves[i])

        return shelves  # List of shelves, each with ordered items
    
