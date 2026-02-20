// DOM Elements
const elements = {
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    runBtn: document.getElementById('runBtn'),
    inputImg: document.getElementById('inputImg'),
    outputImg: document.getElementById('outputImg'),
    statusDisplay: document.getElementById('statusDisplay'),
    classesCard: document.getElementById('classesCard'),
    detectedGrid: document.getElementById('detectedGrid'),
    resultPlaceholder: document.getElementById('resultPlaceholder'),
    resultWrap: document.getElementById('resultWrap'),
    phText: document.getElementById('phText'),
    procDots: document.getElementById('procDots'),
    changeBtn: document.getElementById('changeBtn'),
    inputBadge: document.getElementById('inputBadge'),
    outputBadge: document.getElementById('outputBadge'),
    classCount: document.getElementById('classCount'),
    inferenceTime: document.getElementById('inferenceTime'),
    progressWrap: document.getElementById('progressWrap'),
    progressFill: document.getElementById('progressFill'),
    useCrf: document.getElementById('useCrf'),
    useOverlay: document.getElementById('useOverlay')
};

// State
let selectedFile = null;
let startTime = null;

// Class colors mapping
const classColors = {
    'person': 'person',
    'car': 'vehicle',
    'bus': 'vehicle',
    'truck': 'vehicle',
    'cat': 'animal',
    'dog': 'animal',
    'bird': 'animal',
    'horse': 'animal',
    'sheep': 'animal',
    'cow': 'animal'
};

// File handling
elements.dropZone.addEventListener('click', () => elements.fileInput.click());

elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('dragover');
});

elements.dropZone.addEventListener('dragleave', () => {
    elements.dropZone.classList.remove('dragover');
});

elements.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

elements.fileInput.addEventListener('change', () => {
    handleFile(elements.fileInput.files[0]);
});

elements.changeBtn.addEventListener('click', () => {
    resetToInitial();
    elements.fileInput.click();
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.inputImg.src = e.target.result;
        elements.inputImg.style.display = 'block';
        elements.dropZone.style.display = 'none';
        elements.changeBtn.style.display = 'block';
        
        // Update badge
        const name = file.name.length > 20 ? file.name.substring(0, 17) + '...' : file.name;
        elements.inputBadge.textContent = name;
        
        resetOutput();
        elements.runBtn.disabled = false;
        updateStatus('ready', 'Image ready');
    };
    reader.readAsDataURL(file);
}

function resetToInitial() {
    elements.dropZone.style.display = 'block';
    elements.inputImg.style.display = 'none';
    elements.changeBtn.style.display = 'none';
    elements.inputBadge.textContent = 'No file';
    selectedFile = null;
    elements.runBtn.disabled = true;
    resetOutput();
}

function resetOutput() {
    elements.outputImg.style.display = 'none';
    elements.resultPlaceholder.style.display = 'flex';
    elements.phText.textContent = 'Awaiting input';
    elements.procDots.style.display = 'none';
    elements.resultWrap.classList.remove('scanning');
    elements.classesCard.style.display = 'none';
    elements.detectedGrid.innerHTML = '';
    elements.outputBadge.textContent = 'Awaiting';
    elements.classCount.textContent = '0 objects';
    elements.inferenceTime.textContent = '--';
    elements.progressWrap.style.display = 'none';
    elements.progressFill.style.width = '0%';
}

function updateStatus(type, message) {
    elements.statusDisplay.className = 'status-message';
    
    if (type === 'loading') {
        elements.statusDisplay.classList.add('loading');
        elements.statusDisplay.innerHTML = '<span class="spinner"></span>' + message;
    } else {
        if (type) elements.statusDisplay.classList.add(type);
        elements.statusDisplay.textContent = message;
    }
}

function startProgress() {
    elements.progressWrap.style.display = 'block';
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        elements.progressFill.style.width = progress + '%';
    }, 200);
    return interval;
}

// Run segmentation
elements.runBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    
    elements.runBtn.disabled = true;
    startTime = Date.now();
    
    // UI updates
    elements.resultPlaceholder.style.display = 'flex';
    elements.phText.textContent = 'Processing';
    elements.procDots.style.display = 'flex';
    elements.resultWrap.classList.add('scanning');
    elements.outputImg.style.display = 'none';
    elements.outputBadge.textContent = 'Processing';
    
    updateStatus('loading', 'Running inference');
    const progressInterval = startProgress();
    
    const form = new FormData();
    form.append('image', selectedFile);
    form.append('use_crf', elements.useCrf.checked ? 'true' : 'false');
    
    fetch('/segment', { method: 'POST', body: form })
        .then(res => res.json())
        .then(data => {
            clearInterval(progressInterval);
            elements.progressFill.style.width = '100%';
            
            if (data.error) throw new Error(data.error);
            
            const elapsed = Date.now() - startTime;
            elements.inferenceTime.textContent = elapsed;
            
            elements.outputImg.onload = () => {
                elements.resultPlaceholder.style.display = 'none';
                elements.resultWrap.classList.remove('scanning');
                elements.procDots.style.display = 'none';
                elements.outputImg.style.display = 'block';
                elements.outputBadge.textContent = 'Complete';
                
                // Success animation
                elements.outputBadge.style.transition = 'all 0.3s ease';
                elements.outputBadge.style.backgroundColor = 'rgba(5, 150, 105, 0.1)';
                elements.outputBadge.style.borderColor = '#059669';
                elements.outputBadge.style.color = '#059669';
                setTimeout(() => {
                    elements.outputBadge.style.backgroundColor = '';
                    elements.outputBadge.style.borderColor = '';
                    elements.outputBadge.style.color = '';
                }, 500);
            };
            
            elements.outputImg.src = data.result_url + '?t=' + Date.now();
            
            updateStatus('success', `Done in ${elapsed}ms`);
            
            // Show detected classes
            if (data.classes && data.classes.length > 0) {
                elements.classesCard.style.display = 'block';
                elements.classCount.textContent = data.classes.length + ' object' + 
                    (data.classes.length > 1 ? 's' : '');
                
                elements.detectedGrid.innerHTML = '';
                data.classes.forEach((cls, i) => {
                    const chip = document.createElement('div');
                    const colorClass = classColors[cls] || 'default';
                    chip.className = `class-chip ${colorClass}`;
                    chip.style.animationDelay = `${i * 0.05}s`;
                    chip.innerHTML = `
                        <span class="chip-dot"></span>
                        ${cls}
                    `;
                    elements.detectedGrid.appendChild(chip);
                });
            }
        })
        .catch(error => {
            clearInterval(progressInterval);
            elements.progressFill.style.width = '0%';
            updateStatus('error', `Error: ${error.message}`);
            elements.phText.textContent = 'Failed';
            elements.procDots.style.display = 'none';
            elements.resultWrap.classList.remove('scanning');
            elements.outputBadge.textContent = 'Failed';
        })
        .finally(() => {
            elements.runBtn.disabled = false;
            setTimeout(() => {
                elements.progressWrap.style.display = 'none';
            }, 1000);
        });
});

// Overlay toggle
elements.useOverlay.addEventListener('change', function() {
    if (elements.outputImg.style.display === 'block') {
        elements.outputImg.style.opacity = this.checked ? '0.6' : '1';
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !elements.runBtn.disabled && selectedFile) {
        elements.runBtn.click();
    }
    if (e.key === 'Escape' && selectedFile) {
        resetToInitial();
    }
});

// Initial animation
document.querySelectorAll('.stat-card, .card, .controls-panel, .classes-section, .info-bar')
    .forEach(el => el.classList.add('fade-slide-up'));