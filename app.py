from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from flask_cors import CORS
from utils import compare_with_planogram, get_shelved_products

# ==============================
# Configuración inicial
# ==============================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PLAN_FOLDER = 'planograms'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLAN_FOLDER'] = PLAN_FOLDER

# ==============================
# Funciones auxiliares
# ==============================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def error_response(message, code=400):
    return jsonify({"success": False, "error": message, "code": code}), code

def success_response(data=None):
    return jsonify({"success": True, "result": data})

# ==============================
# Rutas de la API
# ==============================

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
    result = compare_with_planogram(image_path, planogram_path)

    return success_response(result)

@app.route('/planograms', methods=['GET'])
def list_planograms():
    """
    Devuelve una lista con los nombres de los planogramas disponibles.
    """
    plans = [f.replace(".jpg", "") for f in os.listdir(PLAN_FOLDER) if f.endswith(".jpg")]
    return success_response(plans)

# ==============================
# Inicio de la aplicación
# ==============================
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PLAN_FOLDER, exist_ok=True)

    import sys
    if '--test' in sys.argv:
        test_image = 'uploads/20250516_173049-1.jpg'
        test_planogram = 'planograms/20250516_172922.jpg'

        print("=== REAL SHELVES ===")
        real_shelves = get_shelved_products(test_image)
        for i, shelf in enumerate(real_shelves):
            print(f"\nShelf {i+1}:")
            for item in shelf:
                print(f" - {item['label']} ({item['confidence']:.2f})")

        print("\n=== PLANOGRAM SHELVES ===")
        planogram_shelves = get_shelved_products(test_planogram)
        for i, shelf in enumerate(planogram_shelves):
            print(f"\nShelf {i+1}:")
            for item in shelf:
                print(f" - {item['label']} ({item['confidence']:.2f})")

        print("\n=== COMPARISON RESULT ===")
        result = compare_with_planogram(test_image, test_planogram)
        print("Test Result:", result)

    app.run(debug=True, host='127.0.0.1', port=5000)