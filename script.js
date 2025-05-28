const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const compressedSizeDisplay = document.getElementById('compressedSize');
const fileList = document.getElementById('fileList');

const API_BASE_URL = 'http://127.0.0.1:5000'; // Base URL of your Flask API

// Load files from localStorage on page load
document.addEventListener('DOMContentLoaded', () => {
    loadFilesFromStorage();
});

imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        }
        reader.readAsDataURL(file);

        // Create FormData to send the file
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Show some loading state if you want
            compressedSizeDisplay.textContent = 'Processing...';

            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            // Show both original and compressed sizes
            compressedSizeDisplay.textContent = `Original: ${result.original_size_kb || formatBytes(result.original_size)}, Compressed: ${result.compressed_size_kb || formatBytes(result.compressed_size)} (${result.compression_ratio})`;
            addFileToList(result.original_filename, result.processed_filename, result.compressed_size);
            saveFileToStorage(result.original_filename, result.processed_filename, result.compressed_size);

        } catch (error) {
            console.error('Error uploading file:', error);
            compressedSizeDisplay.textContent = 'Error';
            alert(`Upload failed: ${error.message}`);
        } finally {
            // Clear the input for next upload
            imageUpload.value = ''; 
        }
    }
});

function addFileToList(originalFileName, processedFileName, compressedSizeBytes) {
    const listItem = document.createElement('li');
    
    const nameSpan = document.createElement('span');
    nameSpan.textContent = originalFileName; // Display original name for user clarity
    
    const infoSpan = document.createElement('span');
    infoSpan.className = 'file-info';
    infoSpan.textContent = `(Compressed: ${formatBytes(compressedSizeBytes)})`;

    const downloadButton = document.createElement('button');
    downloadButton.textContent = 'Download Processed';
    downloadButton.onclick = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/download/${processedFileName}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            
            // Create proper download filename based on original file
            const originalFileExtension = getFileExtension(originalFileName);
            const originalBaseName = getFileBaseName(originalFileName);
            const downloadFileName = `${originalBaseName}_lenia_compressed.jpg`;  // Always JPG now
            
            a.download = downloadFileName; 
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error downloading file:', error);
            alert(`Download failed: ${error.message}`);
        }
    };

    listItem.appendChild(nameSpan);
    listItem.appendChild(infoSpan);
    listItem.appendChild(downloadButton);
    fileList.appendChild(listItem);
}

function saveFileToStorage(originalFileName, processedFileName, compressedSizeBytes) {
    const storedFiles = JSON.parse(localStorage.getItem('uploadedFiles')) || [];
    // Store processedFileName for download, originalFileName for display
    if (!storedFiles.find(f => f.original_filename === originalFileName)) { // Check by original name to avoid re-adding same source
        storedFiles.push({ 
            original_filename: originalFileName, 
            processed_filename: processedFileName, 
            compressed_size: compressedSizeBytes 
        });
        localStorage.setItem('uploadedFiles', JSON.stringify(storedFiles));
    }
}

function loadFilesFromStorage() {
    const storedFiles = JSON.parse(localStorage.getItem('uploadedFiles')) || [];
    storedFiles.forEach(file => {
        addFileToList(file.original_filename, file.processed_filename, file.compressed_size);
    });
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === undefined || bytes === null || isNaN(bytes)) return 'N/A';
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Helper functions to extract file extension and base name
function getFileExtension(filename) {
    return filename.split('.').pop().toLowerCase();
}

function getFileBaseName(filename) {
    return filename.substring(0, filename.lastIndexOf('.')) || filename;
}

// Optional: Clear storage for testing during development
// localStorage.removeItem('uploadedFiles');