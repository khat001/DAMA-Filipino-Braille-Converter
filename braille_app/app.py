"""
Flask API for Braille Detection System
Properly integrated with existing YOLO detector and braille converter scripts
"""
from models.yolo_detector import YOLODetector
from scripts.braille_converter import RobustBrailleConverter
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pathlib import Path
import base64
import io
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import os


# Initialize Flask app with proper static and template folder configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
LOCAL_FOLDER = Path.home() / '.local' / 'share' / 'braille_app'
LOCAL_FOLDER.mkdir(parents=True, exist_ok=True)

UPLOAD_FOLDER = LOCAL_FOLDER / 'uploads'
RESULTS_FOLDER = LOCAL_FOLDER / 'results'

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# Model configuration
MODEL_PATH = "models/braille_detection_yolov11s/weights/best.pt"
GROQ_API_KEY = os.getenv("GROQ_API")

# Image standardization configuration (from predict.py defaults)
STANDARDIZE_SIZE = True
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
KEEP_ASPECT_RATIO = True


def standardize_image_size(image, target_width, target_height, keep_aspect_ratio=True):
    """
    Resize image to standard size (from predict.py lines 146-184)

    Args:
        image: Input image (numpy array)
        target_width: Target width in pixels
        target_height: Target height in pixels
        keep_aspect_ratio: If True, pad image to maintain aspect ratio

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if not keep_aspect_ratio:
        # Simple resize, may distort image
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    # Calculate scaling to fit within target dimensions
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h),
                         interpolation=cv2.INTER_LANCZOS4)

    # Create canvas with target size (black background)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate position to center the image
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    # Place resized image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


print("🚀 Loading YOLO model...")
detector = YOLODetector(model_name=MODEL_PATH, device='cpu', verbose=False)
print("✅ Model loaded successfully!")

converter = RobustBrailleConverter(
    line_height_threshold=50,
    word_gap_threshold=80,      # predict.py default
    char_gap_threshold=30,       # predict.py default
    min_confidence=0.10,         # predict.py default
    enable_spellcheck=False,     # predict.py default
    enable_gap_detection=True,
    bilingual=True,
    enable_llm_correction=True if GROQ_API_KEY else False,
    llm_api='groq',
    llm_api_key=GROQ_API_KEY,
    target_language='en'
)
print("✅ Converter initialized with default values!")


# ========== TEMPLATE ROUTES ==========

@app.route('/')
def index():
    """Serve the main homepage"""
    return render_template('index.html')


@app.route('/history')
def history():
    """Serve the history page"""
    return render_template('history.html')


@app.route('/how-it-works')
def how_it_works():
    """Serve the how it works page"""
    return render_template('how-it-works.html')


@app.route('/about')
def about():
    """Serve the about page"""
    return render_template('about.html')


# ========== API ROUTES ==========

@app.route('/api/files/<path:filename>')
def serve_file(filename):
    """Serve files from local folders"""
    try:
        # Check if file is in uploads or results
        upload_path = UPLOAD_FOLDER / filename
        result_path = RESULTS_FOLDER / filename

        if upload_path.exists():
            return send_file(str(upload_path))
        elif result_path.exists():
            return send_file(str(result_path))
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversion history with file paths"""
    try:
        result_files = sorted(
            RESULTS_FOLDER.glob('output_*.jpg'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:10]

        history = []
        for file_path in result_files:
            timestamp = file_path.stem.replace('output_', '')

            # Find corresponding files
            input_file = f"input_{timestamp}.jpg"
            text_file = RESULTS_FOLDER / f"text_{timestamp}.txt"

            text_content = ""
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()

            history.append({
                'timestamp': timestamp,
                'original_image': f"/api/files/{input_file}",
                'prediction_image': f"/api/files/{file_path.name}",
                'text': text_content[:100] + '...' if len(text_content) > 100 else text_content,
                'full_text': text_content,
                'date': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })

        return jsonify({'history': history}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'llm_enabled': converter.enable_llm_correction,
        'converter_config': {
            'line_height_threshold': converter.line_height_threshold,
            'word_gap_threshold': converter.word_gap_threshold,
            'char_gap_threshold': converter.char_gap_threshold,
            'min_confidence': converter.min_confidence,
            'enable_spellcheck': converter.enable_spellcheck,
            'enable_gap_detection': converter.enable_gap_detection,
            'bilingual': converter.bilingual,
            'target_language': converter.target_language
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/convert', methods=['POST'])
def convert_braille():
    """
    Main endpoint for braille conversion
    Expects: multipart/form-data with 'image' and optional 'language'
    Returns: JSON with converted text and annotated image
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        language = request.form.get('language', 'en')  # 'en', 'tl', or 'both'

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Get original image dimensions
        original_height, original_width = image.shape[:2]
        print(f"📐 Original image size: {original_width}x{original_height}")

        # Standardize image size (from predict.py defaults)
        if STANDARDIZE_SIZE:
            image = standardize_image_size(
                image,
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT,
                KEEP_ASPECT_RATIO
            )
            new_height, new_width = image.shape[:2]
            print(f"📏 Standardized to: {new_width}x{new_height}")

        # Save original image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = UPLOAD_FOLDER / f"input_{timestamp}.jpg"
        cv2.imwrite(str(input_path), image)

        print(f"📸 Processing image: {input_path}")

        # Update converter's target language based on request
        converter.target_language = language

        # Run YOLO detection with exact default values from predict.py (lines 40-46):
        # - imgsz=640
        # - conf=0.25
        # - iou=0.45
        # - max_det=300
        results = detector.predict(
            source=str(input_path),
            conf=0.25,
            iou=0.45,
            imgsz=640,
            max_det=300,
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

        # Save text file
        text_file = RESULTS_FOLDER / f"text_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(report['final_text'])

        print(f"✅ Conversion complete!")
        print(f"   Detections: {report['total_detections']}")
        print(f"   Text: {report['final_text'][:50]}...")

        # Prepare response
        response = {
            'success': True,
            'text': report['final_text'],
            'raw_text': report['raw_text'],
            'annotated_image': f"data:image/jpeg;base64,{annotated_base64}",
            'image_info': {
                'original_width': int(original_width),
                'original_height': int(original_height),
                'processed_width': int(new_width) if STANDARDIZE_SIZE else int(original_width),
                'processed_height': int(new_height) if STANDARDIZE_SIZE else int(original_height),
                'standardized': STANDARDIZE_SIZE
            },
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
            'timestamp': timestamp,
            'language': language
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


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get overall statistics"""
    try:
        total_conversions = len(list(RESULTS_FOLDER.glob('output_*.jpg')))

        return jsonify({
            'total_conversions': total_conversions,
            'model_name': 'YOLOv11s',
            'languages_supported': ['English (en)', 'Filipino (tl)', 'Mixed (both)'],
            'llm_enabled': converter.enable_llm_correction,
            'image_standardization': {
                'enabled': STANDARDIZE_SIZE,
                'target_width': OUTPUT_WIDTH,
                'target_height': OUTPUT_HEIGHT,
                'keep_aspect_ratio': KEEP_ASPECT_RATIO
            },
            'converter_settings': {
                'line_height_threshold': converter.line_height_threshold,
                'word_gap_threshold': converter.word_gap_threshold,
                'char_gap_threshold': converter.char_gap_threshold,
                'min_confidence': converter.min_confidence,
                'spellcheck_enabled': converter.enable_spellcheck,
                'gap_detection_enabled': converter.enable_gap_detection,
                'bilingual': converter.bilingual
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'converter': {
            'line_height_threshold': converter.line_height_threshold,
            'word_gap_threshold': converter.word_gap_threshold,
            'char_gap_threshold': converter.char_gap_threshold,
            'min_confidence': converter.min_confidence,
            'enable_spellcheck': converter.enable_spellcheck,
            'enable_gap_detection': converter.enable_gap_detection,
            'bilingual': converter.bilingual,
            'enable_llm_correction': converter.enable_llm_correction,
            'llm_api': converter.llm_api,
            'target_language': converter.target_language
        },
        'detector': {
            'model_path': MODEL_PATH,
            'device': 'cpu',
            'default_conf': 0.25,
            'default_iou': 0.45,
            'default_imgsz': 640,
            'default_max_det': 300
        },
        'image_standardization': {
            'enabled': STANDARDIZE_SIZE,
            'target_width': OUTPUT_WIDTH,
            'target_height': OUTPUT_HEIGHT,
            'keep_aspect_ratio': KEEP_ASPECT_RATIO
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 BRAILLE DETECTION API SERVER")
    print("="*60)
    print(f"📍 Web Interface: http://localhost:5000")
    print(f"📍 API Endpoint: http://localhost:5000/api")
    print(f"🏥 Health check: http://localhost:5000/api/health")
    print(f"⚙️  Configuration: http://localhost:5000/api/config")
    print(
        f"🔧 LLM Correction: {'Enabled ✅' if converter.enable_llm_correction else 'Disabled ❌'}")
    print(f"📚 Languages: English (en), Filipino (tl), Mixed (both)")
    print("\n📋 Converter Settings (predict.py Default Values):")
    print(f"   • Line Height Threshold: {converter.line_height_threshold}px")
    print(f"   • Word Gap Threshold: {converter.word_gap_threshold}px")
    print(f"   • Char Gap Threshold: {converter.char_gap_threshold}px")
    print(f"   • Min Confidence: {converter.min_confidence}")
    print(
        f"   • Spellcheck: {'Enabled' if converter.enable_spellcheck else 'Disabled'}")
    print(
        f"   • Gap Detection: {'Enabled' if converter.enable_gap_detection else 'Disabled'}")
    print(
        f"   • Bilingual: {'Enabled' if converter.bilingual else 'Disabled'}")
    print("\n📏 Image Standardization:")
    print(f"   • Enabled: {'Yes' if STANDARDIZE_SIZE else 'No'}")
    print(f"   • Target Size: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
    print(f"   • Keep Aspect Ratio: {'Yes' if KEEP_ASPECT_RATIO else 'No'}")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
