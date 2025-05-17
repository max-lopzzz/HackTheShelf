from ultralytics import YOLO
import numpy as np

model = YOLO("model005.pt")

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

            detected.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "center_x": center_x,
                "center_y": center_y
            })

        # --- Shelf grouping logic ---
        shelf_threshold = 100  # vertical distance threshold to separate shelves
        sorted_by_y = sorted(detected, key=lambda obj: obj["center_y"])

        shelves = []
        current_shelf = [sorted_by_y[0]]

        for obj in sorted_by_y[1:]:
            last_obj = current_shelf[-1]
            
            if abs(obj["center_y"] - last_obj["center_y"]) <= shelf_threshold:
                current_shelf.append(obj)
            else:
                shelves.append(current_shelf)
                current_shelf = [obj]
        shelves.append(current_shelf)  # add last shelf

        # Sort each shelf left to right
        for i, shelf in enumerate(shelves):
            shelves[i] = sorted(shelf, key=lambda obj: obj["center_x"])

        result.show()
        return shelves  # List of shelves, each with ordered items