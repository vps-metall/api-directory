import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from io import BytesIO
import face_recognition

# Настройки
KNOWN_FACES_DIR = 'img'
PICKLE_FILE = 'known_faces.pickle'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)  # Разрешает все источники

# Проверка формата
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Загрузка известных лиц и сохранение в .pickle
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)
            if face_encoding:
                known_face_encodings.append(face_encoding[0])
                known_face_names.append(filename.split('.')[0])
    
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    
    return known_face_encodings, known_face_names

# Загрузка из .pickle
def load_encodings():
    if not os.path.exists(PICKLE_FILE):
        return load_known_faces(KNOWN_FACES_DIR)
    with open(PICKLE_FILE, 'rb') as f:
        return pickle.load(f)

# Загружаем при старте
known_face_encodings, known_face_names = load_encodings()

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if image_file and allowed_file(image_file.filename):
        image_data = image_file.read()
        image = face_recognition.load_image_file(BytesIO(image_data))
        
        # Уменьшаем изображение для ускорения
        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        result = process_image(small_image)
        
        return jsonify(result), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

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

        # Обновляем и сохраняем эмбеддинги
        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)

        return jsonify({"status": "Изображение сохранено и массив лиц обновлён"}), 200
    else:
        return jsonify({"error": "Недопустимый формат файла"}), 400

@app.route('/delete_image', methods=['POST'])
def delete_image():
    # Получаем строку напрямую из тела запроса
    name = request.data.decode('utf-8').strip()

    if not name:
        return jsonify({"error": "Имя не передано"}), 400

    found_file = None

    # Ищем файл с любым допустимым расширением
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


def process_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        return {"type": "faceNoDetection"}
    
    unknown_face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for unknown_face_encoding in unknown_face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
        if len(distances) == 0:
            continue

        min_distance = min(distances)
        if min_distance < 0.6:
            match_index = np.argmin(distances)
            matched_face_name = known_face_names[match_index]
            return {"type": "faceDetection", "userName": matched_face_name}
        else:
            return {"type": "faceDetection", "userName": "unknown"}
    
    return {"type": "faceNoDetection"}

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
