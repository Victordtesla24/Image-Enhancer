// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const widthSlider = document.getElementById('widthSlider');
const widthValue = document.getElementById('widthValue');
const previewContainer = document.getElementById('previewContainer');
const originalPreview = document.getElementById('originalPreview');
const enhancedImage = document.getElementById('enhancedImage');
const loadingSpinner = document.getElementById('loadingSpinner');
const toast = document.getElementById('toast');

// Event Listeners
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
uploadBtn.addEventListener('click', () => fileInput.click());
widthSlider.addEventListener('input', updateWidthValue);

// Make elements focusable
dropZone.setAttribute('tabindex', '0');
uploadBtn.setAttribute('tabindex', '0');
widthSlider.setAttribute('tabindex', '0');

// Add keyboard event listeners
dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

uploadBtn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

// Handle drag and drop events
function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processImage(file);
    } else {
        showToast('Please drop an image file', 'error');
    }
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        processImage(file);
    } else {
        showToast('Please select an image file', 'error');
    }
}

// Update width value display
function updateWidthValue() {
    widthValue.textContent = `${widthSlider.value}px`;
}

// Process and enhance image
async function processImage(file) {
    try {
        // Show loading state immediately
        previewContainer.style.display = 'grid';
        enhancedImage.style.display = 'none';
        
        // Initialize loading spinner with both classes
        loadingSpinner.className = 'loading-spinner visible';

        // Show original preview
        const reader = new FileReader();
        reader.onload = function(e) {
            originalPreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Send to API
        const response = await fetch(`/enhance?target_width=${widthSlider.value}`, {
            method: 'POST',
            body: file
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to enhance image');
        }

        // Get enhanced image blob
        const blob = await response.blob();
        const enhancedUrl = URL.createObjectURL(blob);
        
        // Display enhanced image
        enhancedImage.onload = function() {
            loadingSpinner.className = 'loading-spinner';
            enhancedImage.style.display = 'block';
        };
        enhancedImage.src = enhancedUrl;
        
        showToast('Image enhanced successfully!', 'success');
    } catch (error) {
        console.error('Error:', error);
        loadingSpinner.className = 'loading-spinner';
        showToast(error.message, 'error');
    }
}

// Toast notification
function showToast(message, type = 'success') {
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.style.display = 'block';
    
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}

// Initialize width value display
updateWidthValue();

// Set initial focus to dropZone
window.addEventListener('load', () => {
    dropZone.focus();
});

// Handle tab key for focus management
document.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
        // If shift+tab, move focus backwards
        if (e.shiftKey) {
            if (document.activeElement === dropZone) {
                e.preventDefault();
                widthSlider.focus();
            } else if (document.activeElement === uploadBtn) {
                e.preventDefault();
                dropZone.focus();
            } else if (document.activeElement === widthSlider) {
                e.preventDefault();
                uploadBtn.focus();
            }
        } else {
            // Regular tab, move focus forwards
            if (document.activeElement === dropZone) {
                e.preventDefault();
                uploadBtn.focus();
            } else if (document.activeElement === uploadBtn) {
                e.preventDefault();
                widthSlider.focus();
            } else if (document.activeElement === widthSlider) {
                e.preventDefault();
                dropZone.focus();
            }
        }
    }
});
