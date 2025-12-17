"""
Flask API for Braille Detection System
Connects your YOLO model with the web frontend
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import base64
import io
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import os

# Import your existing modules
from models.yolo_detector import YOLODetector
from scripts.braille_converter import RobustBrailleConverter

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = Path('uploads')
RESULTS_FOLDER = Path('results')
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# Initialize model (load once on startup)
MODEL_PATH = "models/braille_detection_yolov11s/weights/best.pt"
GROQ_API_KEY = os.getenv("GROQ_API")

print("🚀 Loading YOLO model...")
detector = YOLODetector(model_name=MODEL_PATH, device='cpu', verbose=False)
print("✅ Model loaded successfully!")

# Initialize converter
converter = RobustBrailleConverter(
    line_height_threshold=50,
    word_gap_threshold=30,
    char_gap_threshold=25,
    min_confidence=0.15,
    enable_spellcheck=True,
    enable_gap_detection=True,
    bilingual=True,
    enable_llm_correction=True if GROQ_API_KEY else False,
    llm_api='groq',
    llm_api_key=GROQ_API_KEY
)
print("✅ Converter initialized!")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'llm_enabled': converter.enable_llm_correction,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/convert', methods=['POST'])
def convert_braille():
    """
    Main endpoint for braille conversion
    Expects: multipart/form-data with 'image' and 'language'
    Returns: JSON with converted text and annotated image
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        language = request.form.get('language', 'english')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Save original image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = UPLOAD_FOLDER / f"input_{timestamp}.jpg"
        cv2.imwrite(str(input_path), image)

        print(f"📸 Processing image: {input_path}")

        # Run YOLO detection
        results = detector.predict(
            source=str(input_path),
            conf=0.25,
            iou=0.45,
            imgsz=640,
            save=False,
            verbose=False
        )

        if not results or len(results) == 0:
            return jsonify({
                'error': 'No braille detected',
                'text': '',
                'detections': 0
            }), 200

        # Get detection report
        report = converter.get_detection_report(results)

        # Get annotated image
        annotated_image = results[0].plot()

        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        # Save annotated image
        output_path = RESULTS_FOLDER / f"output_{timestamp}.jpg"
        cv2.imwrite(str(output_path), annotated_image)

        print(f"✅ Conversion complete!")
        print(f"   Detections: {report['total_detections']}")
        print(f"   Text: {report['final_text'][:50]}...")

        # Prepare response
        response = {
            'success': True,
            'text': report['final_text'],
            'raw_text': report['raw_text'],
            'annotated_image': f"data:image/jpeg;base64,{annotated_base64}",
            'statistics': {
                'total_detections': int(report['total_detections']),
                'num_lines': int(report['num_lines']),
                'average_confidence': float(report['average_confidence']),
                'quality_score': float(report['quality_score']),
                'corrections_made': int(report['corrections_made']),
                'llm_corrections_made': int(report.get('llm_corrections_made', 0))
            },
            'corrections': report['corrections'],
            'llm_corrections': report.get('llm_corrections', []),
            'low_confidence_words': [
                {
                    'word': item['word'],
                    'confidence': float(item['confidence'])
                } for item in report['low_confidence_words']
            ],
            'timestamp': timestamp
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'success': False
        }), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download converted text as file"""
    try:
        file_path = RESULTS_FOLDER / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404

        return send_file(
            str(file_path),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversion history (last 10 conversions)"""
    try:
        # Get all result files
        result_files = sorted(
            RESULTS_FOLDER.glob('output_*.jpg'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:10]

        history = []
        for file_path in result_files:
            timestamp = file_path.stem.replace('output_', '')

            # Try to find corresponding text file
            text_file = RESULTS_FOLDER / f"text_{timestamp}.txt"
            text_content = ""
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()

            history.append({
                'timestamp': timestamp,
                'image_path': str(file_path.name),
                'text': text_content[:100] + '...' if len(text_content) > 100 else text_content,
                'date': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })

        return jsonify({'history': history}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get overall statistics"""
    try:
        total_conversions = len(list(RESULTS_FOLDER.glob('output_*.jpg')))

        return jsonify({
            'total_conversions': total_conversions,
            'model_name': 'YOLOv11s',
            'languages_supported': ['English', 'Filipino'],
            'llm_enabled': converter.enable_llm_correction
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 BRAILLE DETECTION API SERVER")
    print("="*60)
    print(f"📍 API running at: http://localhost:5000")
    print(f"📊 Health check: http://localhost:5000/api/health")
    print(
        f"🔧 LLM Correction: {'Enabled' if converter.enable_llm_correction else 'Disabled'}")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
