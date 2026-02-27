document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resetBtn = document.getElementById('reset-btn');
    const extractBtn = document.getElementById('extract-btn');
    const loadingState = document.getElementById('loading-state');
    const resultPanel = document.getElementById('result-panel');
    const resultText = document.getElementById('result-text');
    const saveLocation = document.getElementById('save-location');

    let currentFile = null;

    // Trigger file dialog
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop event listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG).');
            return;
        }

        currentFile = file;

        // Display image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            resultPanel.classList.add('hidden');
            imagePreviewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
        currentFile = null;
        imagePreview.src = '';
        fileInput.value = '';

        imagePreviewContainer.classList.add('hidden');
        resultPanel.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    extractBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI state update
        imagePreviewContainer.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultPanel.classList.add('hidden');

        // Form setup
        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            loadingState.classList.add('hidden');

            if (data.error) {
                alert('An error occurred: ' + data.error);
                resetBtn.click();
            } else {
                document.getElementById('result-ingredients').textContent = data.ingredients || 'Not found';
                document.getElementById('result-nutrition').textContent = data.nutrition || 'Not found';
                document.getElementById('result-raw').textContent = data.full_text;

                saveLocation.textContent = 'Saved to: ' + data.saved_file;
                resultPanel.classList.remove('hidden');
                imagePreviewContainer.classList.remove('hidden');
            }

        } catch (error) {
            console.error('Extraction Error:', error);
            loadingState.classList.add('hidden');
            alert('Server disconnected or failed to process the request.');
            imagePreviewContainer.classList.remove('hidden');
        }
    });
});
