var dropZone       = document.getElementById('dropZone');
var fileInput      = document.getElementById('fileInput');
var runBtn         = document.getElementById('runBtn');
var inputImg       = document.getElementById('inputImg');
var outputImg      = document.getElementById('outputImg');
var statusDisplay  = document.getElementById('statusDisplay');
var classesCard    = document.getElementById('classesCard');
var detectedGrid   = document.getElementById('detectedGrid');
var resultPlaceholder = document.getElementById('resultPlaceholder');
var resultWrap     = document.getElementById('resultWrap');
var phText         = document.getElementById('phText');
var procDots       = document.getElementById('procDots');
var changeBtn      = document.getElementById('changeBtn');
var inputBadge     = document.getElementById('inputBadge');
var outputBadge    = document.getElementById('outputBadge');
var classCount     = document.getElementById('classCount');
var inferenceTime  = document.getElementById('inferenceTime');
var progressWrap   = document.getElementById('progressWrap');
var progressFill   = document.getElementById('progressFill');
var selectedFile   = null;
var startTime      = null;

// Class colors
var classColors = {
    'person':   'person',
    'car':      'car',
    'cat':      'cat',
    'dog':      'dog',
    'bird':     'bird'
};

// Drop zone
dropZone.addEventListener('click', function() { fileInput.click(); });
dropZone.addEventListener('dragover', function(e) { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', function() { dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', function() { handleFile(fileInput.files[0]); });

changeBtn.addEventListener('click', function() {
    dropZone.style.display = 'block';
    inputImg.style.display = 'none';
    changeBtn.style.display = 'none';
    inputBadge.textContent = 'No file';
    selectedFile = null;
    runBtn.disabled = true;
    resetOutput();
    fileInput.click();
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    selectedFile = file;

    var reader = new FileReader();
    reader.onload = function(e) {
        inputImg.src = e.target.result;
        inputImg.style.display = 'block';
        dropZone.style.display = 'none';
        changeBtn.style.display = 'block';

        // Update badge with filename
        var name = file.name.length > 18 ? file.name.substring(0,15) + '...' : file.name;
        inputBadge.textContent = name;

        resetOutput();
        runBtn.disabled = false;
        setStatus('', 'Image ready — click Run');
    };
    reader.readAsDataURL(file);
}

function resetOutput() {
    outputImg.style.display = 'none';
    resultPlaceholder.style.display = 'flex';
    phText.textContent = 'Ready to segment';
    procDots.style.display = 'none';
    resultWrap.classList.remove('scanning');
    classesCard.style.display = 'none';
    detectedGrid.innerHTML = '';
    outputBadge.textContent = 'Awaiting';
    classCount.textContent = '0 objects';
    inferenceTime.textContent = '--';
    progressWrap.style.display = 'none';
    progressFill.style.width = '0%';
}

function setStatus(type, msg) {
    statusDisplay.className = 'status-display';
    if (type) statusDisplay.classList.add(type);
    if (type === 'loading') {
        statusDisplay.innerHTML = '<div class="spinner"></div>' + msg;
    } else {
        statusDisplay.textContent = msg;
    }
}

function animateProgress() {
    progressWrap.style.display = 'block';
    var pct = 0;
    var interval = setInterval(function() {
        pct += Math.random() * 12;
        if (pct > 90) pct = 90;
        progressFill.style.width = pct + '%';
    }, 300);
    return interval;
}

runBtn.addEventListener('click', function() {
    if (!selectedFile) return;
    runBtn.disabled = true;
    startTime = Date.now();

    // Show scanning state
    resultPlaceholder.style.display = 'flex';
    phText.textContent = 'Processing';
    procDots.style.display = 'flex';
    resultWrap.classList.add('scanning');
    outputImg.style.display = 'none';
    outputBadge.textContent = 'Processing...';

    setStatus('loading', 'Running inference...');
    var progressInterval = animateProgress();

    var form = new FormData();
    form.append('image', selectedFile);
    form.append('use_crf', document.getElementById('useCrf').checked ? 'true' : 'false');

    fetch('/segment', { method: 'POST', body: form })
        .then(function(res) { return res.json(); })
        .then(function(data) {
            clearInterval(progressInterval);
            progressFill.style.width = '100%';

            if (data.error) throw new Error(data.error);

            var elapsed = Date.now() - startTime;
            inferenceTime.textContent = elapsed;

            outputImg.onload = function() {
                resultPlaceholder.style.display = 'none';
                resultWrap.classList.remove('scanning');
                procDots.style.display = 'none';
                outputImg.style.display = 'block';
                outputBadge.textContent = 'Complete';
            };
            outputImg.src = data.result_url + '?t=' + Date.now();

            setStatus('success', 'Done in ' + elapsed + 'ms');

            // Show detected classes
            if (data.classes && data.classes.length > 0) {
                classesCard.style.display = 'block';
                classCount.textContent = data.classes.length + ' object' + (data.classes.length > 1 ? 's' : '');
                detectedGrid.innerHTML = '';
                data.classes.forEach(function(cls, i) {
                    var chip = document.createElement('div');
                    var colorClass = classColors[cls] || 'default';
                    chip.className = 'det-chip ' + colorClass;
                    chip.style.animationDelay = (i * 0.06) + 's';
                    chip.innerHTML = '<div class="chip-dot"></div>' + cls;
                    chip.title = 'VOC class: ' + cls;
                    detectedGrid.appendChild(chip);
                });
            }
        })
        .catch(function(e) {
            clearInterval(progressInterval);
            progressFill.style.width = '0%';
            setStatus('error', 'Error: ' + e.message);
            phText.textContent = 'Failed';
            procDots.style.display = 'none';
            resultWrap.classList.remove('scanning');
            outputBadge.textContent = 'Failed';
        })
        .finally(function() {
            runBtn.disabled = false;
            setTimeout(function() { progressWrap.style.display = 'none'; }, 1500);
        });
});

// CRF toggle visual feedback
document.getElementById('useCrf').addEventListener('change', function() {
    if (this.checked) {
        setStatus('', 'DenseCRF enabled — sharper boundaries');
    } else {
        setStatus('', 'DenseCRF disabled — raw CNN output');
    }
});

// Overlay toggle
document.getElementById('useOverlay').addEventListener('change', function() {
    if (outputImg.style.display === 'block') {
        outputImg.style.opacity = this.checked ? '0.7' : '1';
    }
});

// Keyboard shortcut: Enter to run
document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !runBtn.disabled) runBtn.click();
});