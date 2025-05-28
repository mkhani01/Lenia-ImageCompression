# Image Processing with Lenia

This project implements image processing using Lenia cellular automata with a Flask backend for intelligent image compression.

## Features

- **Lenia-Enhanced Compression**: Uses cellular automata patterns for predictive image compression
- **Real Compression**: Achieves 2:1 to 4:1 compression ratios with maintained quality
- **Multi-format Support**: Handles PNG, JPG, and JPEG images
- **Color Preservation**: Maintains color information in compressed images
- **Web Interface**: Simple drag-and-drop interface for image processing

## Requirements

- Python 3.7+
- Flask
- OpenCV (cv2)
- NumPy
- SciPy
- scikit-learn

## Setup

### For Windows Users

1. **Clone this repository**

   ```cmd
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create and activate a virtual environment:**

   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```cmd
   pip install flask flask-cors opencv-python numpy scipy scikit-learn werkzeug
   ```

4. **Run the Flask app:**

   ```cmd
   python main.py
   ```

5. **Open the web interface:**
   - Open `index.html` in your web browser
   - Or navigate to `http://127.0.0.1:5000` if serving the HTML through Flask

### For macOS/Linux Users

1. **Clone this repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install flask flask-cors opencv-python numpy scipy scikit-learn werkzeug
   ```

4. **Run the Flask app:**

   ```bash
   python3 main.py
   ```

5. **Open the web interface:**
   - Open `index.html` in your web browser
   - Or navigate to `http://127.0.0.1:5000` if serving the HTML through Flask

## Usage

1. **Upload an Image**: Click "Choose File" and select a PNG, JPG, or JPEG image
2. **View Compression**: The system will show:
   - Original file size
   - Compressed file size
   - Compression ratio achieved
3. **Download Result**: Click "Download Processed" to get the compressed image
4. **File Management**: Previously uploaded files are stored locally and can be re-downloaded

## How It Works

The system uses **Lenia Cellular Automata** for intelligent image compression:

1. **Predictive Analysis**: Lenia kernels analyze image patterns to predict pixel values
2. **Multi-scale Processing**: Different kernel sizes handle various image features
3. **Residual Encoding**: Only unpredictable differences are stored
4. **Smart Reconstruction**: Combines Lenia predictions with stored residuals
5. **Quality Enhancement**: Applies bilateral filtering for final image improvement

## Project Structure

```
├── main.py              # Flask backend server
├── index.html           # Web interface
├── script.js           # Frontend JavaScript
├── uploads/            # Temporary upload folder (auto-created)
├── processed/          # Compressed images storage (auto-created)
└── README.md           # This file
```

## API Endpoints

- `POST /upload` - Upload and compress an image
- `GET /download/<filename>` - Download compressed image
- `GET /info/<filename>` - Get compression information

## Troubleshooting

### Common Issues

**"Module not found" errors:**

```bash
pip install --upgrade pip
pip install flask flask-cors opencv-python numpy scipy scikit-learn werkzeug
```

**CORS errors in browser:**

- Make sure Flask server is running on `http://127.0.0.1:5000`
- Check that CORS is enabled in the Flask app

**File upload fails:**

- Ensure the image format is PNG, JPG, or JPEG
- Check file size (very large images may take longer to process)

**Windows-specific issues:**

- Use `python` instead of `python3` in commands
- Use `venv\Scripts\activate` to activate virtual environment
- Ensure Python is added to system PATH

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Lenia cellular automata research by Bert Wang-Chak Chan
- OpenCV for image processing capabilities
- Flask for the web framework
