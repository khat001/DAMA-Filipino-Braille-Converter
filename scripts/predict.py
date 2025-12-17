# scripts/predict.py
"""
Robust Braille Prediction Script with Error Correction
Includes spell checking, gap detection, and intelligent corrections
"""
from scripts.braille_converter import RobustBrailleConverter
import numpy as np
import time
import yaml
import cv2
from datetime import datetime
import argparse
from models.yolo_detector import YOLODetector
from config.model_config import ModelConfig
from utils.logger import get_logger
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
GROQ_API = os.getenv("GROQ_API")
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Robust Braille Prediction with Error Correction")

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Image, directory, video, or camera (0, 1, 2...)")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "config" / "data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum detections per image")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps)")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 half-precision")
    parser.add_argument("--line-width", type=int, default=2,
                        help="Bounding box line width")
    parser.add_argument("--show-labels", action="store_true",
                        default=True, help="Show labels")
    parser.add_argument("--show-conf", action="store_true",
                        default=True, help="Show confidence")
    parser.add_argument("--hide-labels", dest="show_labels",
                        action="store_false")
    parser.add_argument("--hide-conf", dest="show_conf", action="store_false")
    parser.add_argument("--project", type=str,
                        default=str(ModelConfig.PREDICTION_DIR))
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--save", action="store_true",
                        default=True, help="Save results")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save results to txt")
    parser.add_argument("--save-conf", action="store_true",
                        help="Save confidence scores")
    parser.add_argument("--save-crop", action="store_true",
                        help="Save cropped predictions")
    parser.add_argument("--view-img", action="store_true",
                        help="Display results")
    parser.add_argument("--verbose", action="store_true",
                        default=True, help="Verbose output")
    parser.add_argument("--camera-width", type=int,
                        default=640, help="Camera width")
    parser.add_argument("--camera-height", type=int,
                        default=480, help="Camera height")
    parser.add_argument("--show-fps", action="store_true",
                        default=True, help="Show FPS")

    # Robust conversion parameters
    parser.add_argument("--line-height", type=int, default=50,
                        help="Max vertical distance for same line (pixels)")
    parser.add_argument("--word-gap", type=int, default=80,
                        help="Min horizontal distance for word spacing (pixels)")
    parser.add_argument("--char-gap", type=int, default=30,
                        help="Expected spacing between characters (pixels)")
    parser.add_argument("--min-confidence", type=float, default=0.10,
                        help="Minimum confidence to trust detection")
    parser.add_argument("--enable-spellcheck", action="store_true", default=False,
                        help="Enable spell checking corrections")
    parser.add_argument("--disable-spellcheck", dest="enable_spellcheck",
                        action="store_false")
    parser.add_argument("--enable-gap-detection", action="store_true", default=True,
                        help="Detect potential missing characters")
    parser.add_argument("--disable-gap-detection", dest="enable_gap_detection",
                        action="store_false")
    parser.add_argument("--show-text", action="store_true", default=True,
                        help="Display converted text on image")
    parser.add_argument("--save-text-file", action="store_true", default=True,
                        help="Save converted text to file")
    parser.add_argument("--save-report", action="store_true", default=True,
                        help="Save detailed correction report")
    parser.add_argument("--language", type=str, default='en',
                        help="Language for spell checking (en, es, etc.)")

    # LLM correction parameters
    parser.add_argument("--enable-llm", action="store_true", default=True,
                        help="Enable LLM-based context correction (requires API key or Ollama)")
    parser.add_argument("--llm-api", type=str, default='groq',
                        choices=['groq', 'ollama', 'together', 'huggingface'],
                        help="Which LLM API to use")
    parser.add_argument("--llm-key", type=str, default=GROQ_API,
                        help="API key for LLM service (not needed for ollama)")
    parser.add_argument("--target-language", type=str, default='en',
                        choices=['en', 'tl', 'both'],
                        help="Target language for LLM correction (en=English, tl=Filipino, both=Mixed)")

    # Image standardization parameters
    parser.add_argument("--standardize-size", action="store_true", default=True,
                        help="Resize all images to standard resolution")
    parser.add_argument("--output-width", type=int, default=1280,
                        help="Standard output width (if --standardize-size enabled)")
    parser.add_argument("--output-height", type=int, default=720,
                        help="Standard output height (if --standardize-size enabled)")
    parser.add_argument("--keep-aspect-ratio", action="store_true", default=True,
                        help="Maintain aspect ratio when resizing")
    parser.add_argument("--no-keep-aspect-ratio", dest="keep_aspect_ratio",
                        action="store_false")

    return parser.parse_args()


def is_camera_source(source):
    """Check if source is a camera index"""
    try:
        int(source)
        return True
    except ValueError:
        return False


def standardize_image_size(image, target_width, target_height, keep_aspect_ratio=True):
    """
    Resize image to standard size

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


def load_and_standardize_image(image_path, target_width, target_height, keep_aspect_ratio=True):
    """Load image and standardize its size"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return standardize_image_size(image, target_width, target_height, keep_aspect_ratio)


def draw_text_overlay(image, text, corrections=None, quality_score=None):
    """Draw converted text with quality indicators on image"""
    if not text:
        return image

    img_height, img_width = image.shape[:2]
    max_width = img_width - 40

    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 30

    # Split text into lines that fit
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        (text_width, _), _ = cv2.getTextSize(
            test_line, font, font_scale, thickness)

        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    # Add correction info line
    info_lines = []
    if corrections and len(corrections) > 0:
        info_lines.append(f"Corrections: {len(corrections)}")
    if quality_score is not None:
        color_indicator = "🟢" if quality_score > 0.8 else "🟡" if quality_score > 0.6 else "🔴"
        info_lines.append(f"Quality: {quality_score:.2f} {color_indicator}")

    # Draw background
    total_lines = len(lines) + len(info_lines)
    overlay_height = total_lines * line_height + 20
    cv2.rectangle(overlay, (10, img_height - overlay_height - 10),
                  (img_width - 10, img_height - 10), (0, 0, 0), -1)

    # Blend overlay
    alpha = 0.7
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw text
    y_position = img_height - overlay_height + 10

    # Draw info lines first
    for info_line in info_lines:
        cv2.putText(image, info_line, (20, y_position), font, 0.5,
                    (255, 255, 0), thickness)
        y_position += 25

    # Draw text lines
    for line in lines:
        cv2.putText(image, line, (20, y_position), font, font_scale,
                    (0, 255, 0), thickness)
        y_position += line_height

    return image


def predict_camera(detector, camera_idx, args, converter):
    """Handle camera/webcam prediction with live Braille to text"""
    logger = get_logger("prediction")

    logger.info("\n" + "="*60)
    logger.info("ROBUST BRAILLE CAMERA MODE")
    logger.info("="*60)
    logger.info(f"Camera index: {camera_idx}")
    logger.info(f"Spell checking: {'ON' if args.enable_spellcheck else 'OFF'}")
    logger.info(
        f"Gap detection: {'ON' if args.enable_gap_detection else 'OFF'}")
    logger.info(
        "Controls: 'q'=Quit, 's'=Save, '+'=More confident, '-'=Less confident")
    logger.info("="*60)

    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_idx}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"\n✓ Camera opened: {actual_width}x{actual_height}")
    logger.info("▶️  Starting live detection... Press 'q' to quit\n")

    frame_count = 0
    start_time = time.time()
    conf_threshold = args.conf

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Predict on current frame
            results = detector.predict(
                source=frame,
                conf=conf_threshold,
                iou=args.iou,
                imgsz=args.imgsz,
                save=False,
                verbose=False
            )

            # Get annotated frame
            if results and len(results) > 0:
                annotated_frame = results[0].plot()

                # Convert to text with metadata
                metadata = converter.convert_results_to_text(
                    results, return_metadata=True)
                converted_text = metadata['text']
                corrections = metadata['corrections']
                quality = metadata.get('average_confidence', 0.0)

                # Show text on frame
                if args.show_text and converted_text:
                    annotated_frame = draw_text_overlay(
                        annotated_frame,
                        converted_text,
                        corrections,
                        quality
                    )

                # Show detection count
                num_detections = len(results[0].boxes)
                cv2.putText(annotated_frame, f"Detections: {num_detections}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                annotated_frame = frame

            # Show FPS
            if args.show_fps:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                            (actual_width - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show confidence
            cv2.putText(annotated_frame, f"Conf: {conf_threshold:.2f}",
                        (actual_width - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display live
            cv2.imshow('Robust Braille Detection', annotated_frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_filename = f"braille_capture_{timestamp}.jpg"
                txt_filename = f"braille_text_{timestamp}.txt"

                cv2.imwrite(img_filename, annotated_frame)
                logger.info(f"📸 Saved image: {img_filename}")

                if converted_text:
                    with open(txt_filename, 'w', encoding='utf-8') as f:
                        f.write(converted_text)
                        if corrections:
                            f.write("\n\n--- Corrections Made ---\n")
                            for corr in corrections:
                                f.write(
                                    f"{corr.original} → {corr.corrected}\n")
                    logger.info(f"📝 Saved text: {txt_filename}")
                    logger.info(f"Text: {converted_text}")

            elif key == ord('+') or key == ord('='):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                logger.info(f"Confidence: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                logger.info(f"Confidence: {conf_threshold:.2f}")

    except KeyboardInterrupt:
        logger.info("\nStopped by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info(
            f"\n✅ Processed {frame_count} frames in {elapsed:.1f}s (Avg FPS: {avg_fps:.2f})")


def main():
    args = parse_args()
    logger = get_logger("prediction", ModelConfig.LOG_DIR)

    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"predictions_{timestamp}"

    logger.info("="*60)
    logger.info("Starting Robust Braille Prediction")
    logger.info("="*60)
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Confidence: {args.conf}")
    logger.info(
        f"Spell checking: {'Enabled' if args.enable_spellcheck else 'Disabled'}")
    logger.info(
        f"Gap detection: {'Enabled' if args.enable_gap_detection else 'Disabled'}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    # Initialize detector
    detector = YOLODetector(model_name=str(weights_path),
                            device=args.device, verbose=args.verbose)
    logger.info(f"Using device: {detector.device}")

    # Initialize Robust Braille converter
    converter = RobustBrailleConverter(
        line_height_threshold=args.line_height,
        word_gap_threshold=args.word_gap,
        char_gap_threshold=args.char_gap,
        min_confidence=args.min_confidence,
        enable_spellcheck=args.enable_spellcheck,
        enable_gap_detection=args.enable_gap_detection,
        bilingual=True,
        enable_llm_correction=args.enable_llm,
        llm_api=args.llm_api,
        llm_api_key=args.llm_key,
        target_language=args.target_language
    )
    logger.info(f"Robust converter initialized")
    logger.info(f"  Line height threshold: {args.line_height}px")
    logger.info(f"  Word gap threshold: {args.word_gap}px")
    logger.info(f"  Character gap threshold: {args.char_gap}px")
    logger.info(f"  Min confidence: {args.min_confidence}")

    # Check if camera
    if is_camera_source(args.source):
        camera_idx = int(args.source)
        predict_camera(detector, camera_idx, args, converter)
        return

    # Handle files/directories
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source not found: {args.source}")
        sys.exit(1)

    # Prepare source for prediction
    prediction_source = args.source
    temp_files = []  # Track temporary standardized images

    # If standardizing size and source is a file or directory
    if args.standardize_size and source_path.is_file():
        logger.info(f"\n📏 Standardizing image size...")
        try:
            standardized = load_and_standardize_image(
                source_path,
                args.output_width,
                args.output_height,
                args.keep_aspect_ratio
            )

            # Save standardized image temporarily
            temp_path = source_path.parent / \
                f"prediction_{source_path.name}"
            cv2.imwrite(str(temp_path), standardized)
            temp_files.append(temp_path)
            prediction_source = str(temp_path)

            logger.info(
                f"✓ Image standardized to {args.output_width}x{args.output_height}")
            logger.info(
                f"  Original size: {cv2.imread(str(source_path)).shape[:2]}")
            logger.info(f"  New size: {standardized.shape[:2]}")

        except Exception as e:
            logger.error(f"Failed to standardize image: {e}")
            logger.info("Continuing with original image...")

    elif args.standardize_size and source_path.is_dir():
        logger.info(f"\n📏 Standardizing all images in directory...")

        # Create temporary directory for standardized images
        temp_dir = source_path.parent / f"{source_path.name}"
        temp_dir.mkdir(exist_ok=True)

        # Process all images in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in source_path.iterdir()
                       if f.suffix.lower() in image_extensions]

        logger.info(f"Found {len(image_files)} images to standardize...")

        for img_file in image_files:
            try:
                standardized = load_and_standardize_image(
                    img_file,
                    args.output_width,
                    args.output_height,
                    args.keep_aspect_ratio
                )

                # Save to temp directory
                temp_path = temp_dir / img_file.name
                cv2.imwrite(str(temp_path), standardized)
                temp_files.append(temp_path)

            except Exception as e:
                logger.warning(f"Failed to standardize {img_file.name}: {e}")

        if temp_files:
            prediction_source = str(temp_dir)
            logger.info(
                f"✓ Standardized {len(temp_files)} images to {args.output_width}x{args.output_height}")
        else:
            logger.warning(
                "No images were standardized, using original directory")

    predict_config = ModelConfig.get_prediction_config(
        imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det,
        half=args.half, save=args.save, save_txt=args.save_txt,
        save_conf=args.save_conf, save_crop=args.save_crop,
        line_width=args.line_width, show_labels=args.show_labels,
        show_conf=args.show_conf
    )

    try:
        results = detector.predict(
            source=prediction_source, project=args.project, name=args.name, **predict_config)

        logger.info("\n" + "="*60)
        logger.info("RESULTS WITH ERROR CORRECTION")
        logger.info("="*60)

        all_reports = []

        for i, result in enumerate(results):
            # Get comprehensive report
            report = converter.get_detection_report([result])
            all_reports.append(report)

            logger.info(f"\n📄 Image {i+1}:")
            logger.info(f"  Total Detections: {report['total_detections']}")
            logger.info(f"  Lines: {report['num_lines']}")
            logger.info(
                f"  Average Confidence: {report['average_confidence']:.2f}")
            logger.info(f"  Quality Score: {report['quality_score']:.2f}")

            if report['corrections_made'] > 0:
                logger.info(
                    f"\n  🔧 Corrections Made: {report['corrections_made']}")
                for corr in report['corrections']:
                    logger.info(
                        f"    '{corr['original']}' → '{corr['corrected']}' ({corr['method']})")

            if report['low_confidence_words']:
                logger.info(f"\n  ⚠️  Low Confidence Words:")
                for lc in report['low_confidence_words'][:5]:  # Show first 5
                    logger.info(
                        f"    '{lc['word']}' (conf: {lc['confidence']:.2f})")

            logger.info(f"\n  📝 Final Text:")
            logger.info(f"  {'-'*50}")
            for line in report['final_text'].split('\n'):
                logger.info(f"  {line}")
            logger.info(f"  {'-'*50}")

            if report['raw_text'] != report['final_text']:
                logger.info(f"\n  📝 Raw Text (before corrections):")
                logger.info(f"  {'-'*50}")
                for line in report['raw_text'].split('\n'):
                    logger.info(f"  {line}")
                logger.info(f"  {'-'*50}")

            # Save text files
            save_dir = Path(args.project) / args.name
            save_dir.mkdir(parents=True, exist_ok=True)

            if hasattr(result, 'path'):
                original_name = Path(result.path).stem
            else:
                original_name = f"image_{i+1}"

            if args.save_text_file:
                text_file = save_dir / f"{original_name}_text.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(report['final_text'])
                logger.info(f"  💾 Text saved to: {text_file}")

            if args.save_report:
                report_file = save_dir / f"{original_name}_report.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== BRAILLE DETECTION REPORT ===\n\n")
                    f.write(f"Detections: {report['total_detections']}\n")
                    f.write(f"Lines: {report['num_lines']}\n")
                    f.write(
                        f"Avg Confidence: {report['average_confidence']:.2f}\n")
                    f.write(
                        f"Quality Score: {report['quality_score']:.2f}\n\n")

                    if report['corrections_made'] > 0:
                        f.write(
                            f"Corrections Made: {report['corrections_made']}\n")
                        for corr in report['corrections']:
                            f.write(
                                f"  '{corr['original']}' → '{corr['corrected']}'\n")
                        f.write("\n")

                    f.write(f"\n=== FINAL TEXT ===\n")
                    f.write(report['final_text'])

                    if report['raw_text'] != report['final_text']:
                        f.write(f"\n\n=== RAW TEXT (before corrections) ===\n")
                        f.write(report['raw_text'])

                logger.info(f"  📊 Report saved to: {report_file}")

            # Show image with text overlay
            if args.view_img and result.orig_img is not None:
                display_img = result.plot()
                if args.show_text:
                    display_img = draw_text_overlay(
                        display_img,
                        report['final_text'],
                        report['corrections'],
                        report['quality_score']
                    )
                cv2.imshow(f"Robust Braille {i+1}", display_img)
                cv2.waitKey(0)

        # Get parameter suggestions
        if results:
            logger.info("\n" + "="*60)
            logger.info("PARAMETER RECOMMENDATIONS")
            logger.info("="*60)
            suggestions = converter.suggest_parameter_adjustments(results)
            for key, msg in suggestions.items():
                logger.info(f"\n{msg}")

        if args.save and results:
            logger.info(f"\n📁 Results saved to: {results[0].save_dir}")

        logger.info("\n✅ Robust Braille prediction completed!")
        if args.view_img:
            cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Clean up temporary standardized images
        if temp_files:
            logger.info(
                f"\n🧹 Cleaning up {len(temp_files)} temporary files...")
            for temp_file in temp_files:
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        import shutil
                        shutil.rmtree(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete {temp_file}: {e}")

            # Remove temp directory if it exists and is empty
            if args.standardize_size and source_path.is_dir():
                temp_dir = source_path.parent / \
                    f"{source_path.name}"
                if temp_dir.exists():
                    try:
                        temp_dir.rmdir()
                    except:
                        pass


if __name__ == "__main__":
    main()
