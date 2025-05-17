import cv2
import numpy as np
import os

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    image = cv2.imread(path)
    return image

# utils.py
from yolo_utils import detect_objects

def compare_with_planogram(image_real, image_planogram):
    real_objects = detect_objects(image_real)
    planogram_objects = detect_objects(image_planogram)

    # Mapeamos objetos por etiqueta
    real_dict = {}
    for obj in real_objects:
        real_dict.setdefault(obj["label"], []).append(obj["bbox"])

    planogram_dict = {}
    for obj in planogram_objects:
        planogram_dict.setdefault(obj["label"], []).append(obj["bbox"])

    matched_products = []
    misplaced_products = []
    missing_products = []
    unexpected_products = []

    # Revisamos los productos esperados en el planograma
    for label, plan_bboxes in planogram_dict.items():
        real_bboxes = real_dict.get(label, [])
        if not real_bboxes:
            missing_products.append(label)
        else:
            matched = False
            for plan_bbox in plan_bboxes:
                for real_bbox in real_bboxes:
                    if is_bbox_aligned(plan_bbox, real_bbox):
                        matched = True
                        matched_products.append(label)
                        break
                if matched:
                    break
            if not matched:
                misplaced_products.append(label)

    # Productos adicionales que no están en el planograma
    for label in real_dict:
        if label not in planogram_dict:
            unexpected_products.append(label)

    match_percentage = len(matched_products) / len(planogram_dict) if planogram_dict else 0

    return {
        "matched_products": list(set(matched_products)),
        "misplaced_products": list(set(misplaced_products)),
        "missing_products": list(set(missing_products)),
        "unexpected_products": list(set(unexpected_products)),
        "match_percentage": match_percentage
    }

def is_bbox_aligned(bbox1, bbox2, threshold=50):
    """
    Compara si dos bounding boxes están alineados.
    Acepta tanto listas simples como tensores anidados.
    """
    # Si vienen como listas anidadas (ej: [[x1,y1,x2,y2]]), tomamos el primer elemento
    if isinstance(bbox1[0], list) or isinstance(bbox1[0], np.ndarray):
        bbox1 = bbox1[0]
    if isinstance(bbox2[0], list) or isinstance(bbox2[0], np.ndarray):
        bbox2 = bbox2[0]

    x_diff = abs(bbox1[0] - bbox2[0])  # x1
    y_diff = abs(bbox1[1] - bbox2[1])  # y1

    return x_diff <= threshold and y_diff <= threshold