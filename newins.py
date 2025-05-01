import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import insightface

# Настройки
KNOWN_FACES_DIR = 'img'
PICKLE_FILE = 'known_faces_insight.pickle'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SIMILARITY_THRESHOLD = 0.9  # Чем меньше, тем строже

app = Flask(__name__)
CORS(app)

model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

def extract_embedding(image):
    faces = model.get(image)
    if faces:
        return normalize_embedding(faces[0].embedding)
    return None

def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = cv2.imread(path)
            embedding = extract_embedding(image)
            if embedding is not None:
                known_face_encodings.append(embedding)
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"❌ Лицо не обнаружено в {filename}")

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names

def load_encodings():
    if not os.path.exists(PICKLE_FILE):
        return load_known_faces()
    with open(PICKLE_FILE, 'rb') as f:
        return pickle.load(f)

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
            input_embedding = extract_embedding(image)
            if input_embedding is None:
                print("🚫 Лицо не обнаружено.")
                return jsonify({"type": "faceNoDetection", "userName": "unknown"}), 200

            min_distance = float('inf')
            match_name = "unknown"

            print("\n📊 Результаты сравнения:")
            for known_face_encoding, name in zip(known_face_encodings, known_face_names):
                distance = np.linalg.norm(input_embedding - known_face_encoding)
                print(f"🔍 Сравнение с {name} -> расстояние: {distance:.4f}")
                if distance < min_distance:
                    min_distance = distance
                    match_name = name

            print(f"\n✅ Лучшее совпадение: {match_name}, дистанция: {min_distance:.4f}\n")

            if min_distance < SIMILARITY_THRESHOLD:
                return jsonify({"type": "faceDetection", "userName": match_name}), 200
            else:
                return jsonify({"type": "faceDetection", "userName": "unknown"}), 200

        except Exception as e:
            print(f"❌ Ошибка при обработке изображения: {str(e)}")
            return jsonify({"type": "faceNoDetection", "userName": "unknown"}), 200

    return jsonify({"type": "faceNoDetection", "userName": "unknown"}), 200


@app.route('/add_image', methods=['POST'])
def add_image():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Отсутствует изображение или имя"}), 400

    image_file = request.files['image']
    name = request.form['name'].strip()

    if image_file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400

    if allowed_file(image_file.filename):
        filename = secure_filename(f"{name}.jpg")
        save_path = os.path.join(KNOWN_FACES_DIR, filename)
        image_file.save(save_path)

        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces()

        return jsonify({"status": "Изображение добавлено и база обновлена"}), 200

    return jsonify({"error": "Неверный формат файла"}), 400

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
        os.remove(found_file)
        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces()
        return jsonify({"status": f"Файл '{os.path.basename(found_file)}' удалён"}), 200

    return jsonify({"error": f"Файл с именем '{name}' не найден"}), 404

if __name__ == '__main__':
    print(f"🔐 Загружено {len(known_face_encodings)} лиц: {', '.join(known_face_names)}")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
