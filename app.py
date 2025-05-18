from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from flask_cors import CORS
from utils import compare_with_planogram
import sys
import cv2
from yolo_utils import detect_objects  # Make sure this import works

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PLAN_FOLDER = 'planograms'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLAN_FOLDER'] = PLAN_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def error_response(message, code=400):
    return jsonify({"success": False, "error": message, "code": code}), code

def success_response(data=None):
    return jsonify({"success": True, "result": data})

@app.route('/')
def index():
    """
    Ruta raíz: Muestra un mensaje de bienvenida.
    Ejemplo: GET http://localhost:5000/
    """
    return "Planomagic Backend API - Ready to validate planograms!"

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Sube una imagen real del punto de venta.
    Campo requerido en formulario: 'image'
    
    Ejemplo usando fetch:
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    });
    """
    if 'image' not in request.files:
        return error_response("No image provided", 400)

    file = request.files['image']

    if file.filename == '':
        return error_response("Empty filename", 400)

    if not allowed_file(file.filename):
        return error_response("File type not allowed. Use .jpg, .jpeg or .png", 400)

    # Usar nombre seguro y único
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(filepath)

    return success_response({
        "message": "Image uploaded successfully",
        "path": filepath
    })

@app.route('/compare', methods=['POST'])
def compare():
    """
    Compara una imagen subida contra un planograma.

    Body esperado (JSON):
    {
      "image_path": "uploads/foto.jpg",
      "planogram": "planograma1"
    }
    """
    data = request.get_json()
    image_path = data.get('image_path')
    planogram_name = data.get('planogram')

    if not image_path or not planogram_name:
        return error_response("Missing parameters: image_path and planogram are required", 400)

    planogram_path = os.path.join(app.config['PLAN_FOLDER'], planogram_name + ".jpg")

    if not os.path.exists(image_path):
        return error_response(f"Image not found at {image_path}", 404)

    if not os.path.exists(planogram_path):
        return error_response(f"Planogram '{planogram_name}' not found", 404)

    # Llamamos a tu función de comparación
    result = compare_with_planogram(real_shelves, planogram_shelves)

    return success_response(result)

@app.route('/planograms', methods=['GET'])
def list_planograms():
    """
    Devuelve una lista con los nombres de los planogramas disponibles.
    """
    plans = [f.replace(".jpg", "") for f in os.listdir(PLAN_FOLDER) if f.endswith(".jpg")]
    return success_response(plans)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PLAN_FOLDER, exist_ok=True)

    if '--test' in sys.argv:
        upload_dir = app.config['UPLOAD_FOLDER']
        plan_dir = app.config['PLAN_FOLDER']
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # --- REAL IMAGE ---
        image_files = [f for f in os.listdir(upload_dir) if allowed_file(f)]
        if not image_files:
            print("No valid images found in", upload_dir)
            exit(1)

        image_path = os.path.join(upload_dir, image_files[0])
        print(f"Using real image: {image_path}")

        # Run detection and save annotated image
        annotated_real, real_shelves = detect_objects(image_path)
        if not real_shelves:
            real_shelves = []
        real_output_path = os.path.join(output_dir, "real_" + os.path.basename(image_path))
        cv2.imwrite(real_output_path, annotated_real)
        print(f"Saved annotated real image to: {real_output_path}")

        # Print shelves
        print("=== REAL SHELVES ===")
        for i, shelf in enumerate(real_shelves):
            print(f"\nShelf {i+1}:")
            for item in shelf:
                print(f" - {item['label']} ({item['confidence']:.2f})")

        # --- PLANOGRAM IMAGE ---
        plan_files = [f for f in os.listdir(plan_dir) if allowed_file(f)]
        if not plan_files:
            print("No valid planograms found in", plan_dir)
            exit(1)

        planogram_path = os.path.join(plan_dir, plan_files[0])
        print(f"Using planogram: {planogram_path}")

        # Run detection and save annotated image
        annotated_planogram, planogram_shelves = detect_objects(planogram_path)
        if not planogram_shelves:
            planogram_shelves = []
        planogram_output_path = os.path.join(output_dir, "planogram_" + os.path.basename(planogram_path))
        cv2.imwrite(planogram_output_path, annotated_planogram)
        print(f"Saved annotated planogram to: {planogram_output_path}")

        # Print shelves
        print("=== PLANOGRAM SHELVES ===")
        for i, shelf in enumerate(planogram_shelves):
            print(f"\nShelf {i+1}:")
            for item in shelf:
                print(f" - {item['label']} ({item['confidence']:.2f})")

        # --- COMPARISON ---
        print("\n=== COMPARISON RESULT ===")
        result = compare_with_planogram(real_shelves, planogram_shelves)
        print("Test Result:", result)
    else:
        app.run(debug=True, host='127.0.0.1', port=5000)

    app.run(debug=True, host='127.0.0.1', port=5000)