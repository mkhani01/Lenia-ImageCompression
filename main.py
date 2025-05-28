from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import pickle
from scipy.ndimage import convolve
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure upload and processed directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_lenia_kernel(radius):
    """Create normalized Lenia kernel"""
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    dist = np.sqrt(x*x + y*y)
    kernel_val = np.exp(-(dist**2) / (2 * (radius/3)**2))
    kernel_val[dist > radius] = 0
    return kernel_val / np.sum(kernel_val)

def lenia_predict(image, kernel):
    """Use Lenia to predict pixel values based on neighborhood"""
    convolved = convolve(image.astype(np.float64), kernel, mode='constant', cval=0)
    return convolved

def compress_and_save_image(image_path, output_filename_base):
    """Compress image using Lenia-based predictive coding and save directly as image"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0, "Could not read image"
    
    original_size = os.path.getsize(image_path)
    original_shape = img.shape
    
    # Check if image is color or grayscale
    if len(original_shape) == 3:
        is_color = True
        channels = cv2.split(img)
    else:
        is_color = False
        channels = [img]
    
    # Create Lenia kernels for prediction
    kernels = [create_lenia_kernel(r) for r in [2, 4]]  # Reduced kernels for efficiency
    
    # Process each channel with aggressive compression
    processed_channels = []
    
    for channel in channels:
        # Convert to float
        channel_float = channel.astype(np.float64) / 255.0
        
        # More aggressive downsampling for better compression
        scale_factor = 6  # Increased from 4 to 6
        h, w = channel.shape
        small_h, small_w = max(1, h // scale_factor), max(1, w // scale_factor)
        channel_small = cv2.resize(channel, (small_w, small_h))
        
        # Upscale and predict
        channel_upscaled = cv2.resize(channel_small, (w, h)) / 255.0
        
        # Use Lenia prediction to enhance upscaled image
        predictions = []
        for kernel in kernels:
            pred = lenia_predict(channel_upscaled, kernel)
            predictions.append(pred)
        
        # Combine predictions
        enhanced = np.mean(predictions, axis=0)
        enhanced = np.clip(enhanced, 0, 1)
        
        # Blend original with enhanced prediction (lossy but much smaller)
        # Use more prediction, less original for better compression
        alpha = 0.3  # How much of original to keep
        result = alpha * channel_float + (1 - alpha) * enhanced
        result = np.clip(result, 0, 1)
        
        # Convert back to uint8
        processed_channel = (result * 255).astype(np.uint8)
        processed_channels.append(processed_channel)
    
    # Combine channels
    if is_color:
        processed_img = cv2.merge(processed_channels)
    else:
        processed_img = processed_channels[0]
    
    # Apply final enhancement
    if is_color:
        processed_img = cv2.bilateralFilter(processed_img, 5, 30, 30)
    else:
        processed_img = cv2.bilateralFilter(processed_img, 5, 30, 30)
    
    # Save directly as compressed JPEG with high compression
    compressed_filename = f"{output_filename_base}_lenia_compressed.jpg"
    compressed_path = os.path.join(app.config['PROCESSED_FOLDER'], compressed_filename)
    
    # Use aggressive JPEG compression
    cv2.imwrite(compressed_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    
    compressed_size = os.path.getsize(compressed_path)
    
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {original_size/compressed_size:.2f}:1")
    
    return compressed_path, compressed_size, None

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        filename_base = os.path.splitext(original_filename)[0]
        
        # Save the uploaded file temporarily
        temp_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(temp_upload_path)
        
        original_size = os.path.getsize(temp_upload_path)

        # Compress with Lenia and save as image
        compressed_path, compressed_size, error = compress_and_save_image(temp_upload_path, filename_base)

        # Delete the original uploaded file
        os.remove(temp_upload_path)

        if error:
            return jsonify({'error': error}), 500
        
        if compressed_path:
            processed_filename = os.path.basename(compressed_path)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            return jsonify({
                'message': 'File compressed successfully with Lenia-enhanced JPEG',
                'original_filename': original_filename,
                'processed_filename': processed_filename,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': f"{compression_ratio:.2f}:1",
                'original_size_kb': f"{original_size/1024:.1f} KB",
                'compressed_size_kb': f"{compressed_size/1024:.1f} KB"
            }), 200
        else:
            return jsonify({'error': 'Compression failed'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/download/<filename>', methods=['GET'])
def download_processed_file(filename):
    """Download the compressed image file directly"""
    safe_filename = secure_filename(filename)
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], safe_filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Since we're storing as image, just send it directly
    response = send_file(
        file_path, 
        as_attachment=True, 
        download_name=filename,
        mimetype='image/jpeg'
    )
    
    return response

@app.route('/info/<filename>', methods=['GET'])
def get_compression_info(filename):
    """Get information about compressed file"""
    safe_filename = secure_filename(filename)
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], safe_filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Read image to get dimensions
        img = cv2.imread(file_path)
        if img is not None:
            return jsonify({
                'compressed_size': os.path.getsize(file_path),
                'image_shape': img.shape,
                'file_format': 'JPEG with Lenia enhancement',
                'compression_method': 'Lenia predictive + JPEG'
            }), 200
        else:
            return jsonify({'error': 'Could not read image file'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Could not read file info: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)