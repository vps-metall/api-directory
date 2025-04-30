from deepface import DeepFace
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pickle
import numpy as np
from flask_cors import CORS
import cv2

# Настройки
KNOWN_FACES_DIR = 'img'
PICKLE_FILE = 'known_faces.pickle'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SIMILARITY_THRESHOLD = 0.9  # Порог схожести

app = Flask(__name__)
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(known_faces_dir, filename)
            try:
                result = DeepFace.represent(img_path=image_path, model_name="Facenet", detector_backend="mtcnn")
                if result:
                    encoding = normalize_embedding(result[0]["embedding"])
                    known_face_encodings.append(encoding)
                    known_face_names.append(filename.split('.')[0])
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names

def load_encodings():
    if not os.path.exists(PICKLE_FILE):
        return load_known_faces(KNOWN_FACES_DIR)
    with open(PICKLE_FILE, 'rb') as f:
        return pickle.load(f)

# Загружаем эмбеддинги при запуске
known_face_encodings, known_face_names = load_encodings()

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"type": "faceNoDetection", "userName": "unknown"}), 200

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"type": "faceNoDetection", "userName": "unknown"}), 200

    if image_file and allowed_file(image_file.filename):
        image_data = image_file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        try:
            result = DeepFace.represent(img_path=image, model_name='Facenet', detector_backend='mtcnn')
            if not result:
                print("🚫 Лицо не обнаружено.")
                return jsonify({"type": "faceNoDetection", "userName": "unknown"}), 200
            
            input_embedding = normalize_embedding(result[0]["embedding"])

            min_distance = float('inf')
            match_name = "unknown"

            print("🔍 Сравнение с базой данных:")
            for known_face_encoding, name in zip(known_face_encodings, known_face_names):
                distance = np.linalg.norm(input_embedding - known_face_encoding)
                print(f"  {name} → схожесть: {distance:.4f}")

                if distance < min_distance:
                    min_distance = distance
                    match_name = name

            print(f"✅ Лучшее совпадение: {match_name}, дистанция: {min_distance:.4f}")

            if min_distance < SIMILARITY_THRESHOLD:
                return jsonify({
                    "type": "faceDetection",
                    "userName": match_name
                }), 200
            else:
                return jsonify({
                    "type": "faceDetection",
                    "userName": "unknown"
                }), 200
        except Exception as e:
            print(f"❌ Ошибка при обработке изображения: {str(e)}")
            return jsonify({"type": "faceNoDetection", "userName": "unknown"}), 200
    else:
        return jsonify({"type": "faceNoDetection", "userName": "unknown"}), 200

@app.route('/add_image', methods=['POST'])
def add_image():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Отсутствует изображение или имя"}), 400

    image_file = request.files['image']
    name = request.form['name'].strip()

    if image_file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400

    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(f"{name}.jpg")
        save_path = os.path.join(KNOWN_FACES_DIR, filename)
        image_file.save(save_path)

        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)

        return jsonify({"status": "Изображение сохранено и массив лиц обновлён"}), 200
    else:
        return jsonify({"error": "Недопустимый формат файла"}), 400

@app.route('/delete_image', methods=['POST'])
def delete_image():
    name = request.data.decode('utf-8').strip()

    if not name:
        return jsonify({"error": "Имя не передано"}), 400

    found_file = None

    for ext in ALLOWED_EXTENSIONS:
        candidate = os.path.join(KNOWN_FACES_DIR, f"{name}.{ext}")
        if os.path.exists(candidate):
            found_file = candidate
            break

    if found_file:
        try:
            os.remove(found_file)
            global known_face_encodings, known_face_names
            known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)
            return jsonify({"status": f"Файл '{os.path.basename(found_file)}' удалён"}), 200
        except Exception as e:
            return jsonify({"error": f"Ошибка при удалении: {str(e)}"}), 500
    else:
        return jsonify({"error": f"Файл с именем '{name}' не найден"}), 404

if __name__ == '__main__':
    print(f"Загружено {len(known_face_encodings)} эмбеддингов: {', '.join(known_face_names)}")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
