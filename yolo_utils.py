from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

model = YOLO("model006.pt")

def remove_overlapping_objects(shelf):
    # Skip processing if label is 'anaquel'
    filtered_shelf = [obj for obj in shelf if obj.get("label") != "anaquel"]
    
    i = 0
    while i < len(filtered_shelf) - 1:
        obj = filtered_shelf[i]
        next_obj = filtered_shelf[i + 1]

        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = next_obj["center_x"], next_obj["center_y"]

        if x1 <= cx <= x2 and y1 <= cy <= y2:
            if obj["confidence"] >= next_obj["confidence"]:
                filtered_shelf.pop(i + 1)
            else:
                filtered_shelf.pop(i)
        else:
            i += 1

    return filtered_shelf

def detect_objects(image_path):
    results = model(image_path)
    detected = []
    annotated_img = None
    
    for result in results:
        boxes = result.boxes
        names = model.names
        
        # --- EARLY FILTERING ---
        valid_boxes = [box for box in boxes if names[int(box.cls)] != "anaquel"]
        
        # Just modify the original result (since we're not using it later)
        result.boxes = valid_boxes
        annotated_img = result.plot(font_size=25)

        # Process only valid detections
        for box in valid_boxes:
            class_id = int(box.cls)
            label = names[class_id]
            confidence = float(box.conf)
            bbox = box.xyxy.tolist()[0]

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
                "length_y": length_y 
            })
            
        # Shelf grouping logic
        if not detected:
            # Return empty shelves if no objects detected
            print("No objects detected. Returning empty shelves.")
            return annotated_img, []

        sorted_by_y = sorted(detected, key=lambda obj: obj["center_y"])
        shelves = []
        current_shelf = [sorted_by_y[0]]

        for obj in sorted_by_y[1:]:
            last_obj = current_shelf[-1]
            shelf_threshold = last_obj["length_y"]
            if abs(obj["center_y"] - last_obj["center_y"]) <= shelf_threshold:
                current_shelf.append(obj)
            else:
                shelves.append(current_shelf)
                current_shelf = [obj]
        shelves.append(current_shelf)  # add last shelf

        # Sort each shelf and remove overlapping items
        for i, shelf in enumerate(shelves):
            shelves[i] = sorted(shelf, key=lambda obj: obj["center_x"])
            shelves[i] = remove_overlapping_objects(shelves[i])

    # --- Moved outside the loop ---
    # Print shelves
    for i, shelf in enumerate(shelves):
        print(f"\nShelf {i+1}:")
        for item in shelf:
            print(f"  {item['label']}")

    # Return both annotated image and shelves
    return annotated_img, shelves  # ✅ Now returns image and detections