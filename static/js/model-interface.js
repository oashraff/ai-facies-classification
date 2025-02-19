document.addEventListener('DOMContentLoaded', () => {
    const modelCards = document.querySelectorAll('.model-card');
    const uploadSection = document.querySelector('.upload-section');
    let selectedModel = null;

    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            // Remove selected class from all cards
            modelCards.forEach(c => c.classList.remove('selected'));
            // Add selected class to clicked card
            card.classList.add('selected');
            // Store selected model
            selectedModel = card.dataset.model;
            // Show upload section
            uploadSection.classList.remove('hidden');
        });
    });

    const uploadForm = document.getElementById('uploadForm');
    const resultsSection = document.getElementById('results');
    const visualizationContainer = document.querySelector('.visualization-container');
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay hidden';
    loadingOverlay.innerHTML = '<div class="spinner"></div><p>Analyzing seismic data...</p>';
    document.body.appendChild(loadingOverlay);

    function showLoading() {
        loadingOverlay.classList.remove('hidden');
    }

    function hideLoading() {
        loadingOverlay.classList.add('hidden');
    }

    function formatModelName(modelName) {
        switch(modelName.toLowerCase()) {
            case 'resnet50':
                return 'ResNet-50';
            case 'resnet34':
                return 'ResNet-34';
            case 'inceptionv3':
                return 'InceptionV3';
            default:
                return modelName;
        }
    }

    function displayResults(result) {
        resultsSection.classList.remove('hidden');
        visualizationContainer.innerHTML = '';

        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';

        const modelInfo = document.createElement('div');
        modelInfo.className = 'model-info';
        modelInfo.innerHTML = `
            <h3>Model Used: ${formatModelName(selectedModel)}</h3>
        `;

        const dominantFacies = Object.entries(result.prediction)
            .reduce((a, b) => a[1] > b[1] ? a : b);

        const predictionMain = document.createElement('div');
        predictionMain.className = 'prediction-main';
        predictionMain.innerHTML = `
            <h2>Dominant Facies</h2>
            <div class="dominant-facies">
                <span class="facies-name">${dominantFacies[0]}</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="--confidence: ${dominantFacies[1]}%"></div>
                    <div class="confidence-score">${dominantFacies[1].toFixed(1)}%</div>
                </div>
            </div>
        `;

        const actionButtons = document.createElement('div');
        actionButtons.className = 'result-actions';
        actionButtons.innerHTML = `
            <button class="btn btn-primary download-btn">
                <i class="fas fa-download"></i> Download PDF Report
            </button>
            <button class="btn btn-secondary reset-btn">
                <i class="fas fa-sync"></i> Start Over
            </button>
        `;

        resultCard.appendChild(modelInfo);
        resultCard.appendChild(predictionMain);
        resultCard.appendChild(actionButtons);
        visualizationContainer.appendChild(resultCard);

        actionButtons.querySelector('.download-btn').addEventListener('click', () => {
            generatePDFReport(result, selectedModel);
        });

        actionButtons.querySelector('.reset-btn').addEventListener('click', () => {
            // Reset selected model
            selectedModel = null;
            
            // Hide upload section
            uploadSection.classList.add('hidden');
            
            // Remove selected class from all model cards
            modelCards.forEach(c => c.classList.remove('selected'));
            
            // Clear results section
            resultsSection.classList.add('hidden');
            visualizationContainer.innerHTML = '';
            
            // Reset file input and submit button
            fileInput.value = '';
            submitButton.disabled = true;
            submitButton.classList.add('btn-disabled');
            
            // Remove any displayed file name
            const existingFileName = document.querySelector('.file-name');
            if (existingFileName) {
                existingFileName.remove();
            }
        });
    }

    function generatePDFReport(result, model) {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        doc.setFontSize(20);
        doc.text('Seismic Analysis Report', 20, 20);
        
        doc.setFontSize(12);
        doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 35);
        doc.text(`Model Used: ${formatModelName(model)}`, 20, 45);
        
        const dominantFacies = Object.entries(result.prediction)
            .reduce((a, b) => a[1] > b[1] ? a : b);
        
        doc.text(`Dominant Facies: ${dominantFacies[0]} (${dominantFacies[1].toFixed(1)}%)`, 20, 60);
        
        doc.text('Full Analysis:', 20, 75);
        let yPos = 85;
        Object.entries(result.prediction).forEach(([facies, confidence]) => {
            doc.text(`${facies}: ${confidence.toFixed(1)}%`, 30, yPos);
            yPos += 10;
        });
        
        doc.save('seismic-analysis-report.pdf');
    }

    function validateFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/tiff'];
        return file && validTypes.includes(file.type);
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        uploadForm.insertAdjacentElement('beforebegin', errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    const fileInput = document.getElementById('seismicData');
    const submitButton = document.querySelector('.submit-btn');
    submitButton.disabled = true;
    submitButton.classList.add('btn-disabled');

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && validateFile(file)) {
            submitButton.disabled = false;
            submitButton.classList.remove('btn-disabled');
            const fileNameDisplay = document.createElement('div');
            fileNameDisplay.className = 'file-name';
            fileNameDisplay.textContent = file.name;
            const existingFileName = document.querySelector('.file-name');
            if (existingFileName) {
                existingFileName.remove();
            }
            fileInput.parentElement.appendChild(fileNameDisplay);
        } else {
            submitButton.disabled = true;
            submitButton.classList.add('btn-disabled');
            if (file) {  // Only show error if a file was actually selected
                showError('Please select a valid image file (JPEG, PNG, or TIFF)');
            }
            const existingFileName = document.querySelector('.file-name');
            if (existingFileName) {
                existingFileName.remove();
            }
        }
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files || !fileInput.files[0]) {
            showError('Please select a file to upload.');
            return;
        }

        if (!validateFile(fileInput.files[0])) {
            showError('Please upload a valid image file (PNG, JPG, or TIFF).');
            return;
        }

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        formData.append('model', selectedModel);

        try {
            showLoading();

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            if (result.error) {
                throw new Error(result.error);
            }
            
            displayResults(result);
        } catch (error) {
            showError('An error occurred while processing your request.');
            console.error('Error:', error);
        } finally {
            hideLoading();
        }
    });
}); 