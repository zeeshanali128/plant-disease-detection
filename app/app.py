import os
import sys
import uuid
import traceback
from flask import Flask, render_template, request, jsonify, redirect, url_for

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import *
from src.predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit


def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if not allowed_file(file.filename):
        return render_template('index.html',
                               error="Invalid file type. Please upload JPG or PNG.")

    # Save uploaded file with unique name
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    img_path = os.path.join(UPLOAD_FOLDER, unique_name)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(img_path)

    try:
        # Run prediction
        result = predict_image(img_path)

        return render_template(
            'result.html',
            crop=result['crop'],
            disease=result['disease'],
            confidence=result['confidence'],
            is_healthy=result['is_healthy'],
            display_name=result['display_name'],
            top_k=result['top_k'],
            image_url=f"uploads/{unique_name}"
        )

    except FileNotFoundError as e:
        return render_template('index.html',
                               error=str(e))
    except Exception as e:
        print(f"[ERROR] {e}")
        return render_template('index.html',
                               error="Prediction failed. Please try again.")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic access."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    img_path = os.path.join(UPLOAD_FOLDER, unique_name)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(img_path)

    try:
        result = predict_image(img_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Plant Disease Detection — Flask App")
    print("  URL: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)