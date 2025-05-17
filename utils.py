import cv2
import numpy as np
import os

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    image = cv2.imread(path)
    return image

def compare_with_planogram(image_path, planogram_path):
    # Simulamos una comparación básica
    # Aquí iría la lógica real con YOLO o modelo de visión artificial

    # Ejemplo de resultado simulado
    return {
        "status": "success",
        "deviations": [
            {"product": "Coca-Cola", "type": "missing"},
            {"product": "Sabritas", "type": "misplaced", "expected_position": [100, 200], "found_position": [300, 400]},
            {"position": [500, 600], "type": "gap"}
        ],
        "summary": {
            "total_products_expected": 10,
            "products_found": 7,
            "products_missing": 2,
            "misplaced": 1,
            "gaps": 1
        }
    }