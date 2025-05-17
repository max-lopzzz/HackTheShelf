# Importamos Flask y algunas utilidades
from flask import Flask, request, jsonify, send_from_directory
from utils import load_image, compare_with_planogram
import os

# Creamos la aplicación Flask
app = Flask(__name__)

# Definimos las carpetas donde se guardarán las imágenes y los planogramas
UPLOAD_FOLDER = 'uploads'
PLAN_FOLDER = 'planograms'

# Configuramos la app para usar esas carpetas
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLAN_FOLDER'] = PLAN_FOLDER

# Ruta principal: solo devuelve un mensaje de confirmación
@app.route('/')
def index():
    return "Planomagic Backend API - Ready to validate planograms!"

# Ruta para subir una imagen desde el cliente (POST)
@app.route('/upload', methods=['POST'])
def upload_image():
    # Verificamos que se haya incluido un archivo en la solicitud
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    
    # Verificamos que el archivo tenga un nombre válido
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Guardamos la imagen en la carpeta uploads
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Devolvemos una respuesta con la ruta donde se guardó la imagen
    return jsonify({"message": "Image uploaded", "path": filepath})

# Ruta para comparar una imagen real contra un planograma (POST)
@app.route('/compare', methods=['POST'])
def compare():
    # Obtenemos los datos del cuerpo de la solicitud
    data = request.get_json()
    image_path = data.get('image_path')
    planogram_name = data.get('planogram')

    # Validamos que ambos parámetros estén presentes
    if not image_path or not planogram_name:
        return jsonify({"error": "Missing parameters"}), 400

    # Construimos la ruta completa del planograma
    planogram_path = os.path.join(app.config['PLAN_FOLDER'], planogram_name + ".jpg")

    # Verificamos que ambos archivos existan
    if not os.path.exists(image_path) or not os.path.exists(planogram_path):
        return jsonify({"error": "Image or planogram not found"}), 404

    # Llamamos a la función que compara ambas imágenes
    result = compare_with_planogram(image_path, planogram_path)

    # Devolvemos el resultado como JSON
    return jsonify(result)

# Punto de entrada del programa
if __name__ == '__main__':
    # Creamos las carpetas necesarias si no existen
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PLAN_FOLDER, exist_ok=True)

    # Iniciamos el servidor Flask en modo debug
    app.run(debug=True, host='0.0.0.0', port=5000)