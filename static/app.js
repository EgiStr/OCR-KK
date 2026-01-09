/**
 * KK-OCR v2 - Modern Web Application
 * Feature-rich client for Indonesian Family Card OCR
 */

// ============================================
// Application State
// ============================================
const AppState = {
    // Configuration
    config: {
        apiBaseUrl: localStorage.getItem('apiBaseUrl') || 'http://localhost:8000',
        pollingInterval: parseInt(localStorage.getItem('pollingInterval')) || 2000,
        autoSwitchResults: localStorage.getItem('autoSwitchResults') !== 'false',
        theme: localStorage.getItem('theme') || 'system'
    },
    
    // Current view state
    currentView: 'upload',
    currentTab: 'single',
    pipelineMode: 'yolo_vlm',
    
    // File management
    singleFile: null,
    batchFiles: [],
    
    // Jobs tracking
    activeJobs: {},
    pollingIntervals: {},
    
    // Results storage
    results: JSON.parse(localStorage.getItem('results') || '[]'),
    currentResult: null
};

// ============================================
// DOM Elements
// ============================================
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNavigation();
    initPipelineSelector();
    initUploadTabs();
    initSingleUpload();
    initBatchUpload();
    initProcessing();
    initJobs();
    initResults();
    initSettings();
    initModal();
    loadSavedResults();
    updateJobBadge();
});

// ============================================
// Theme Management
// ============================================
function initTheme() {
    const theme = AppState.config.theme;
    applyTheme(theme);
    
    $('#themeToggle').addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const newTheme = current === 'dark' ? 'light' : 'dark';
        applyTheme(newTheme);
        AppState.config.theme = newTheme;
        localStorage.setItem('theme', newTheme);
        $('#themeSelect').value = newTheme;
    });
}

function applyTheme(theme) {
    if (theme === 'system') {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
    } else {
        document.documentElement.setAttribute('data-theme', theme);
    }
}

// ============================================
// Navigation
// ============================================
function initNavigation() {
    $$('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const view = item.dataset.view;
            switchView(view);
        });
    });
}

function switchView(viewName) {
    // Update nav items
    $$('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });
    
    // Update views
    $$('.view').forEach(view => {
        view.classList.toggle('active', view.id === `${viewName}View`);
    });
    
    AppState.currentView = viewName;
}

// ============================================
// Pipeline Selector
// ============================================
function initPipelineSelector() {
    $$('.pipeline-option').forEach(option => {
        option.addEventListener('click', () => {
            $$('.pipeline-option').forEach(o => o.classList.remove('active'));
            option.classList.add('active');
            AppState.pipelineMode = option.dataset.mode;
            
            // Update processing steps visibility based on mode
            updateProcessingSteps();
        });
    });
}

function updateProcessingSteps() {
    const yoloStep = $('[data-step="yolo"]');
    const unetStep = $('[data-step="unet"]');
    
    if (AppState.pipelineMode === 'vlm_only') {
        yoloStep.style.display = 'none';
        unetStep.style.display = 'none';
    } else if (AppState.pipelineMode === 'yolo_vlm') {
        yoloStep.style.display = 'flex';
        unetStep.style.display = 'none';
    } else {
        yoloStep.style.display = 'flex';
        unetStep.style.display = 'flex';
    }
}

// ============================================
// Upload Tabs
// ============================================
function initUploadTabs() {
    $$('.upload-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;
            switchUploadTab(tabName);
        });
    });
}

function switchUploadTab(tabName) {
    $$('.upload-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    $$('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}Tab`);
    });
    
    AppState.currentTab = tabName;
}

// ============================================
// Single File Upload
// ============================================
function initSingleUpload() {
    const uploadZone = $('#singleUploadZone');
    const fileInput = $('#singleFileInput');
    const preview = $('#singlePreview');
    
    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSingleFile(e.target.files[0]);
        }
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleSingleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Remove file
    $('#removeSingleFile').addEventListener('click', () => {
        AppState.singleFile = null;
        fileInput.value = '';
        preview.style.display = 'none';
        uploadZone.style.display = 'block';
    });
    
    // Process button
    $('#processSingleBtn').addEventListener('click', processSingleFile);
}

function handleSingleFile(file) {
    if (!validateFile(file)) return;
    
    AppState.singleFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        if (file.type.startsWith('image/')) {
            $('#singlePreviewImg').src = e.target.result;
        } else {
            // PDF placeholder
            $('#singlePreviewImg').src = createPDFPlaceholder();
        }
    };
    reader.readAsDataURL(file);
    
    $('#singleFileName').textContent = file.name;
    $('#singleFileSize').textContent = formatFileSize(file.size);
    
    $('#singleUploadZone').style.display = 'none';
    $('#singlePreview').style.display = 'block';
}

// ============================================
// Batch Upload
// ============================================
function initBatchUpload() {
    const uploadZone = $('#batchUploadZone');
    const fileInput = $('#batchFileInput');
    
    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            addBatchFiles(Array.from(e.target.files));
        }
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            addBatchFiles(Array.from(e.dataTransfer.files));
        }
    });
    
    // Add more files
    $('#addMoreFiles').addEventListener('click', () => fileInput.click());
    
    // Clear all
    $('#clearAllFiles').addEventListener('click', clearBatchFiles);
    
    // Process button
    $('#processBatchBtn').addEventListener('click', processBatchFiles);
}

function addBatchFiles(files) {
    const validFiles = files.filter(validateFile);
    
    // Limit to 20 files
    const remaining = 20 - AppState.batchFiles.length;
    const filesToAdd = validFiles.slice(0, remaining);
    
    if (validFiles.length > remaining) {
        showToast(`Only ${remaining} more files can be added (max 20)`, 'warning');
    }
    
    AppState.batchFiles = [...AppState.batchFiles, ...filesToAdd];
    updateBatchFileList();
}

function updateBatchFileList() {
    const list = $('#batchFileList');
    const grid = $('#fileGrid');
    const count = $('#batchFileCount');
    const uploadZone = $('#batchUploadZone');
    
    if (AppState.batchFiles.length === 0) {
        list.style.display = 'none';
        uploadZone.style.display = 'block';
        return;
    }
    
    list.style.display = 'block';
    if (AppState.batchFiles.length < 20) {
        uploadZone.style.display = 'block';
    } else {
        uploadZone.style.display = 'none';
    }
    
    count.textContent = AppState.batchFiles.length;
    
    grid.innerHTML = AppState.batchFiles.map((file, index) => `
        <div class="file-card" data-index="${index}">
            <img class="file-card-thumbnail" src="${file.type.startsWith('image/') ? URL.createObjectURL(file) : createPDFPlaceholder()}" alt="${file.name}">
            <div class="file-card-name">${file.name}</div>
            <button class="file-card-remove" onclick="removeFileFromBatch(${index})">×</button>
        </div>
    `).join('');
}

function removeFileFromBatch(index) {
    AppState.batchFiles.splice(index, 1);
    updateBatchFileList();
}

function clearBatchFiles() {
    AppState.batchFiles = [];
    $('#batchFileInput').value = '';
    updateBatchFileList();
}

// ============================================
// Processing
// ============================================
function initProcessing() {
    updateProcessingSteps();
}

async function processSingleFile() {
    if (!AppState.singleFile) {
        showToast('No file selected', 'error');
        return;
    }
    
    showProcessing();
    updateProgress(10, 'Uploading file...');
    
    try {
        const formData = new FormData();
        formData.append('file', AppState.singleFile);
        
        // Show YOLO step
        updateProgress(30, 'Detecting fields...');
        setStepActive('yolo');
        
        const response = await fetch(`${AppState.config.apiBaseUrl}/v2/extract/kk`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }
        
        // Show enhancement step
        updateProgress(60, 'Enhancing image...');
        setStepCompleted('yolo');
        setStepActive('unet');
        
        await delay(500);
        
        // Show VLM step
        updateProgress(80, 'Extracting data...');
        setStepCompleted('unet');
        setStepActive('vlm');
        
        const result = await response.json();
        
        // Complete
        updateProgress(100, 'Complete!');
        setStepCompleted('vlm');
        
        await delay(500);
        hideProcessing();
        
        // Save result
        saveResult(result);
        
        // Reset upload
        AppState.singleFile = null;
        $('#singleFileInput').value = '';
        $('#singlePreview').style.display = 'none';
        $('#singleUploadZone').style.display = 'block';
        
        // Show result
        if (AppState.config.autoSwitchResults) {
            switchView('results');
        }
        
        showToast('Extraction completed successfully!', 'success');
        
    } catch (error) {
        console.error('Processing error:', error);
        hideProcessing();
        showToast(error.message || 'Processing failed', 'error');
    }
}

async function processBatchFiles() {
    if (AppState.batchFiles.length === 0) {
        showToast('No files selected', 'error');
        return;
    }
    
    const useAsync = $('#asyncMode').checked;
    
    if (useAsync) {
        await processBatchAsync();
    } else {
        await processBatchSync();
    }
}

async function processBatchSync() {
    showProcessing();
    updateProgress(0, 'Starting batch processing...');
    
    try {
        const formData = new FormData();
        AppState.batchFiles.forEach(file => {
            formData.append('files', file);
        });
        
        updateProgress(20, `Processing ${AppState.batchFiles.length} files...`);
        setStepActive('yolo');
        
        const response = await fetch(`${AppState.config.apiBaseUrl}/v2/extract/kk/batch`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Batch processing failed');
        }
        
        updateProgress(80, 'Finalizing...');
        setStepCompleted('yolo');
        setStepCompleted('unet');
        setStepActive('vlm');
        
        const result = await response.json();
        
        updateProgress(100, 'Complete!');
        setStepCompleted('vlm');
        
        await delay(500);
        hideProcessing();
        
        // Save results
        if (result.results) {
            result.results.forEach(r => {
                if (r.data) saveResult(r.data);
            });
        }
        
        // Clear batch files
        clearBatchFiles();
        
        if (AppState.config.autoSwitchResults) {
            switchView('results');
        }
        
        const summary = result.summary || {};
        showToast(`Batch completed: ${summary.successful || 0} successful, ${summary.failed || 0} failed`, 'success');
        
    } catch (error) {
        console.error('Batch processing error:', error);
        hideProcessing();
        showToast(error.message || 'Batch processing failed', 'error');
    }
}

async function processBatchAsync() {
    showProcessing();
    updateProgress(10, 'Starting async batch job...');
    
    try {
        const formData = new FormData();
        AppState.batchFiles.forEach(file => {
            formData.append('files', file);
        });
        
        const response = await fetch(`${AppState.config.apiBaseUrl}/v2/extract/kk/batch/async`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start async job');
        }
        
        const jobStatus = await response.json();
        
        hideProcessing();
        
        // Add job to tracking
        AppState.activeJobs[jobStatus.job_id] = jobStatus;
        startJobPolling(jobStatus.job_id);
        updateJobsView();
        updateJobBadge();
        
        // Clear batch files
        clearBatchFiles();
        
        // Switch to jobs view
        switchView('jobs');
        
        showToast('Batch job started! Tracking in Jobs view.', 'info');
        
    } catch (error) {
        console.error('Async batch error:', error);
        hideProcessing();
        showToast(error.message || 'Failed to start async job', 'error');
    }
}

function showProcessing() {
    $('#processingOverlay').style.display = 'flex';
    resetProcessingSteps();
}

function hideProcessing() {
    $('#processingOverlay').style.display = 'none';
}

function updateProgress(percent, status) {
    $('#progressFill').style.width = `${percent}%`;
    $('#progressText').textContent = `${percent}%`;
    $('#processingStatus').textContent = status;
}

function resetProcessingSteps() {
    $$('.processing-steps .step').forEach(step => {
        step.classList.remove('active', 'completed');
    });
}

function setStepActive(stepName) {
    const step = $(`[data-step="${stepName}"]`);
    if (step) {
        step.classList.remove('completed');
        step.classList.add('active');
    }
}

function setStepCompleted(stepName) {
    const step = $(`[data-step="${stepName}"]`);
    if (step) {
        step.classList.remove('active');
        step.classList.add('completed');
    }
}

// ============================================
// Jobs Management
// ============================================
function initJobs() {
    // Initialize from any persisted jobs
    const savedJobs = JSON.parse(localStorage.getItem('activeJobs') || '{}');
    AppState.activeJobs = savedJobs;
    
    // Resume polling for active jobs
    Object.keys(AppState.activeJobs).forEach(jobId => {
        const job = AppState.activeJobs[jobId];
        if (job.status === 'pending' || job.status === 'processing') {
            startJobPolling(jobId);
        }
    });
    
    updateJobsView();
}

function startJobPolling(jobId) {
    if (AppState.pollingIntervals[jobId]) return;
    
    AppState.pollingIntervals[jobId] = setInterval(async () => {
        try {
            const response = await fetch(`${AppState.config.apiBaseUrl}/v2/extract/kk/batch/${jobId}`);
            
            if (!response.ok) {
                throw new Error('Failed to get job status');
            }
            
            const jobStatus = await response.json();
            AppState.activeJobs[jobId] = jobStatus;
            updateJobsView();
            saveActiveJobs();
            
            // Check if job is complete
            if (jobStatus.status === 'completed' || jobStatus.status === 'failed') {
                stopJobPolling(jobId);
                
                if (jobStatus.status === 'completed') {
                    // Save results
                    if (jobStatus.results) {
                        jobStatus.results.forEach(r => {
                            if (r.data) saveResult(r.data);
                        });
                    }
                    showToast(`Job ${jobId.substring(0, 8)}... completed!`, 'success');
                } else {
                    showToast(`Job ${jobId.substring(0, 8)}... failed`, 'error');
                }
                
                updateJobBadge();
            }
            
        } catch (error) {
            console.error('Job polling error:', error);
        }
    }, AppState.config.pollingInterval);
}

function stopJobPolling(jobId) {
    if (AppState.pollingIntervals[jobId]) {
        clearInterval(AppState.pollingIntervals[jobId]);
        delete AppState.pollingIntervals[jobId];
    }
}

function updateJobsView() {
    const container = $('#jobsContainer');
    const jobs = Object.values(AppState.activeJobs);
    
    if (jobs.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📋</span>
                <h3>No Active Jobs</h3>
                <p>Start a batch processing job to see it here</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = jobs.map(job => {
        const progress = job.progress || 0;
        const statusIcon = {
            'pending': '⏳',
            'processing': '🔄',
            'completed': '✅',
            'failed': '❌'
        }[job.status] || '❓';
        
        return `
            <div class="job-card ${job.status}">
                <div class="job-icon">${statusIcon}</div>
                <div class="job-info">
                    <div class="job-id">${job.job_id}</div>
                    <div class="job-status">${job.status.toUpperCase()}</div>
                </div>
                <div class="job-progress">
                    <div class="job-progress-bar">
                        <div class="job-progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <div class="job-progress-text">${progress}% - ${job.processed || 0}/${job.total || 0} files</div>
                </div>
                <div class="job-actions">
                    ${job.status === 'completed' || job.status === 'failed' ? 
                        `<button class="btn-danger btn-sm" onclick="deleteJob('${job.job_id}')">Delete</button>` : 
                        ''
                    }
                </div>
            </div>
        `;
    }).join('');
}

function updateJobBadge() {
    const activeCount = Object.values(AppState.activeJobs).filter(
        j => j.status === 'pending' || j.status === 'processing'
    ).length;
    
    const badge = $('#activeJobCount');
    if (activeCount > 0) {
        badge.textContent = activeCount;
        badge.style.display = 'block';
    } else {
        badge.style.display = 'none';
    }
}

async function deleteJob(jobId) {
    try {
        await fetch(`${AppState.config.apiBaseUrl}/v2/extract/kk/batch/${jobId}`, {
            method: 'DELETE'
        });
        
        stopJobPolling(jobId);
        delete AppState.activeJobs[jobId];
        saveActiveJobs();
        updateJobsView();
        updateJobBadge();
        
        showToast('Job deleted', 'info');
    } catch (error) {
        console.error('Delete job error:', error);
        showToast('Failed to delete job', 'error');
    }
}

function saveActiveJobs() {
    localStorage.setItem('activeJobs', JSON.stringify(AppState.activeJobs));
}

// ============================================
// Results Management
// ============================================
function initResults() {
    $('#exportAllJson').addEventListener('click', exportAllJson);
    $('#exportAllExcel').addEventListener('click', exportAllExcel);
    $('#clearResults').addEventListener('click', clearResults);
}

function saveResult(data) {
    // Add timestamp and ID
    const result = {
        id: `result_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString(),
        data: data
    };
    
    AppState.results.unshift(result);
    
    // Keep only last 50 results
    if (AppState.results.length > 50) {
        AppState.results = AppState.results.slice(0, 50);
    }
    
    localStorage.setItem('results', JSON.stringify(AppState.results));
    updateResultsView();
}

function loadSavedResults() {
    updateResultsView();
}

function updateResultsView() {
    const container = $('#resultsContainer');
    const toolbar = $('#resultsToolbar');
    const countEl = $('#resultsCount');
    
    if (AppState.results.length === 0) {
        toolbar.style.display = 'none';
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📊</span>
                <h3>No Results Yet</h3>
                <p>Process documents to see extraction results here</p>
            </div>
        `;
        return;
    }
    
    toolbar.style.display = 'flex';
    countEl.textContent = AppState.results.length;
    
    container.innerHTML = AppState.results.map(result => {
        const data = result.data;
        const header = data.header || {};
        const members = data.anggota_keluarga || [];
        
        return `
            <div class="result-card" onclick="showResultDetail('${result.id}')">
                <div class="result-card-header">
                    <h4>KK ${header.no_kk || 'Unknown'}</h4>
                    <span>${new Date(result.timestamp).toLocaleString('id-ID')}</span>
                </div>
                <div class="result-card-body">
                    <div class="result-item">
                        <span class="result-label">Kepala Keluarga</span>
                        <span class="result-value">${header.kepala_keluarga || '-'}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Anggota</span>
                        <span class="result-value">${members.length} orang</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Kecamatan</span>
                        <span class="result-value">${header.kecamatan || '-'}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function showResultDetail(resultId) {
    const result = AppState.results.find(r => r.id === resultId);
    if (!result) return;
    
    AppState.currentResult = result;
    renderResultModal(result.data);
    $('#resultModal').classList.add('active');
}

function renderResultModal(data) {
    const body = $('#modalBody');
    const header = data.header || {};
    const members = data.anggota_keluarga || [];
    const metadata = data.metadata || {};
    
    body.innerHTML = `
        <!-- Metadata -->
        <div class="data-section">
            <h4>ℹ️ Metadata</h4>
            <div class="data-grid">
                <div class="data-item">
                    <div class="label">Waktu Proses</div>
                    <div class="value">${metadata.processing_timestamp || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">Model YOLO</div>
                    <div class="value">${metadata.model_version_yolo || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">Model VLM</div>
                    <div class="value">${metadata.model_version_vlm || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">File</div>
                    <div class="value">${metadata.source_file || '-'}</div>
                </div>
            </div>
        </div>
        
        <!-- Header KK -->
        <div class="data-section">
            <h4>📋 Informasi Kartu Keluarga</h4>
            <div class="data-grid">
                <div class="data-item">
                    <div class="label">No. KK</div>
                    <div class="value">${header.no_kk || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">Kepala Keluarga</div>
                    <div class="value">${header.kepala_keluarga || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">Alamat</div>
                    <div class="value">${header.alamat || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">RT/RW</div>
                    <div class="value">${header.rt || '-'}/${header.rw || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">Desa/Kelurahan</div>
                    <div class="value">${header.desa || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">Kecamatan</div>
                    <div class="value">${header.kecamatan || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">Kabupaten/Kota</div>
                    <div class="value">${header.kabupaten_kota || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">Provinsi</div>
                    <div class="value">${header.provinsi || '-'}</div>
                </div>
                <div class="data-item">
                    <div class="label">Kode Pos</div>
                    <div class="value">${header.kode_pos || '-'}</div>
                </div>
            </div>
        </div>
        
        <!-- Family Members -->
        <div class="data-section">
            <h4>👥 Anggota Keluarga (${members.length})</h4>
            ${members.map((member, index) => `
                <div class="member-section">
                    <div class="member-header">
                        <span class="member-name">${member.nama_lengkap || 'Tidak Ada Nama'}</span>
                        <span class="member-badge">Anggota ${index + 1}</span>
                    </div>
                    <div class="data-grid">
                        <div class="data-item">
                            <div class="label">NIK</div>
                            <div class="value">${member.nik || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Jenis Kelamin</div>
                            <div class="value">${member.jenis_kelamin || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Tempat Lahir</div>
                            <div class="value">${member.tempat_lahir || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Tanggal Lahir</div>
                            <div class="value">${member.tanggal_lahir || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Agama</div>
                            <div class="value">${member.agama || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Pendidikan</div>
                            <div class="value">${member.pendidikan || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Pekerjaan</div>
                            <div class="value">${member.jenis_pekerjaan || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Status Perkawinan</div>
                            <div class="value">${member.status_perkawinan || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Status Keluarga</div>
                            <div class="value">${member.status_keluarga || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Kewarganegaraan</div>
                            <div class="value">${member.kewarganegaraan || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Nama Ayah</div>
                            <div class="value">${member.nama_ayah || '-'}</div>
                        </div>
                        <div class="data-item">
                            <div class="label">Nama Ibu</div>
                            <div class="value">${member.nama_ibu || '-'}</div>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function clearResults() {
    if (!confirm('Are you sure you want to clear all results?')) return;
    
    AppState.results = [];
    localStorage.removeItem('results');
    updateResultsView();
    showToast('All results cleared', 'info');
}

function exportAllJson() {
    if (AppState.results.length === 0) {
        showToast('No results to export', 'warning');
        return;
    }
    
    const data = AppState.results.map(r => r.data);
    downloadJson(data, `kk_results_${Date.now()}.json`);
    showToast('JSON exported successfully', 'success');
}

function exportAllExcel() {
    if (AppState.results.length === 0) {
        showToast('No results to export', 'warning');
        return;
    }
    
    try {
        const wb = XLSX.utils.book_new();
        
        // Summary sheet
        const summaryData = AppState.results.map((r, i) => {
            const h = r.data.header || {};
            const m = r.data.anggota_keluarga || [];
            return [
                i + 1,
                h.no_kk || '-',
                h.kepala_keluarga || '-',
                h.kecamatan || '-',
                h.kabupaten_kota || '-',
                m.length,
                new Date(r.timestamp).toLocaleString('id-ID')
            ];
        });
        
        summaryData.unshift(['No', 'No. KK', 'Kepala Keluarga', 'Kecamatan', 'Kabupaten/Kota', 'Jumlah Anggota', 'Waktu Proses']);
        
        const wsSummary = XLSX.utils.aoa_to_sheet(summaryData);
        wsSummary['!cols'] = [
            { wch: 5 }, { wch: 18 }, { wch: 25 }, { wch: 15 }, { wch: 20 }, { wch: 15 }, { wch: 20 }
        ];
        XLSX.utils.book_append_sheet(wb, wsSummary, 'Summary');
        
        // All members sheet
        const allMembers = [];
        allMembers.push([
            'No. KK', 'No', 'NIK', 'Nama Lengkap', 'Jenis Kelamin', 'Tempat Lahir', 'Tanggal Lahir',
            'Agama', 'Pendidikan', 'Pekerjaan', 'Status Perkawinan', 'Status Keluarga',
            'Kewarganegaraan', 'Nama Ayah', 'Nama Ibu'
        ]);
        
        AppState.results.forEach(result => {
            const noKK = result.data.header?.no_kk || '-';
            const members = result.data.anggota_keluarga || [];
            
            members.forEach((m, i) => {
                allMembers.push([
                    noKK, i + 1, m.nik || '-', m.nama_lengkap || '-', m.jenis_kelamin || '-',
                    m.tempat_lahir || '-', m.tanggal_lahir || '-', m.agama || '-',
                    m.pendidikan || '-', m.jenis_pekerjaan || '-', m.status_perkawinan || '-',
                    m.status_keluarga || '-', m.kewarganegaraan || '-', m.nama_ayah || '-', m.nama_ibu || '-'
                ]);
            });
        });
        
        const wsMembers = XLSX.utils.aoa_to_sheet(allMembers);
        wsMembers['!cols'] = [
            { wch: 18 }, { wch: 5 }, { wch: 18 }, { wch: 25 }, { wch: 12 }, { wch: 15 }, { wch: 12 },
            { wch: 10 }, { wch: 15 }, { wch: 20 }, { wch: 15 }, { wch: 15 }, { wch: 12 }, { wch: 25 }, { wch: 25 }
        ];
        XLSX.utils.book_append_sheet(wb, wsMembers, 'All Members');
        
        XLSX.writeFile(wb, `kk_results_${Date.now()}.xlsx`);
        showToast('Excel exported successfully', 'success');
        
    } catch (error) {
        console.error('Excel export error:', error);
        showToast('Failed to export Excel', 'error');
    }
}

// ============================================
// Modal
// ============================================
function initModal() {
    $('#closeModal').addEventListener('click', closeModal);
    $('.modal-overlay').addEventListener('click', closeModal);
    
    $('#modalDownloadJson').addEventListener('click', () => {
        if (AppState.currentResult) {
            const noKK = AppState.currentResult.data.header?.no_kk || 'unknown';
            downloadJson(AppState.currentResult.data, `kk_${noKK}.json`);
        }
    });
    
    $('#modalExportExcel').addEventListener('click', () => {
        if (AppState.currentResult) {
            exportSingleExcel(AppState.currentResult.data);
        }
    });
    
    $('#modalCopyJson').addEventListener('click', () => {
        if (AppState.currentResult) {
            navigator.clipboard.writeText(JSON.stringify(AppState.currentResult.data, null, 2))
                .then(() => showToast('JSON copied to clipboard', 'success'))
                .catch(() => showToast('Failed to copy', 'error'));
        }
    });
    
    // Close on escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

function closeModal() {
    $('#resultModal').classList.remove('active');
    AppState.currentResult = null;
}

function exportSingleExcel(data) {
    try {
        const wb = XLSX.utils.book_new();
        const header = data.header || {};
        const members = data.anggota_keluarga || [];
        
        // Header sheet
        const headerData = [
            ['Informasi Kartu Keluarga', ''],
            ['No. KK', header.no_kk || '-'],
            ['Kepala Keluarga', header.kepala_keluarga || '-'],
            ['Alamat', header.alamat || '-'],
            ['RT/RW', `${header.rt || '-'}/${header.rw || '-'}`],
            ['Desa/Kelurahan', header.desa || '-'],
            ['Kecamatan', header.kecamatan || '-'],
            ['Kabupaten/Kota', header.kabupaten_kota || '-'],
            ['Provinsi', header.provinsi || '-'],
            ['Kode Pos', header.kode_pos || '-']
        ];
        
        const wsHeader = XLSX.utils.aoa_to_sheet(headerData);
        wsHeader['!cols'] = [{ wch: 20 }, { wch: 40 }];
        XLSX.utils.book_append_sheet(wb, wsHeader, 'Header');
        
        // Members sheet
        const memberData = [
            ['No', 'NIK', 'Nama Lengkap', 'Jenis Kelamin', 'Tempat Lahir', 'Tanggal Lahir',
             'Agama', 'Pendidikan', 'Pekerjaan', 'Status Perkawinan', 'Status Keluarga',
             'Kewarganegaraan', 'Nama Ayah', 'Nama Ibu']
        ];
        
        members.forEach((m, i) => {
            memberData.push([
                i + 1, m.nik || '-', m.nama_lengkap || '-', m.jenis_kelamin || '-',
                m.tempat_lahir || '-', m.tanggal_lahir || '-', m.agama || '-',
                m.pendidikan || '-', m.jenis_pekerjaan || '-', m.status_perkawinan || '-',
                m.status_keluarga || '-', m.kewarganegaraan || '-', m.nama_ayah || '-', m.nama_ibu || '-'
            ]);
        });
        
        const wsMembers = XLSX.utils.aoa_to_sheet(memberData);
        wsMembers['!cols'] = [
            { wch: 5 }, { wch: 18 }, { wch: 25 }, { wch: 12 }, { wch: 15 }, { wch: 12 },
            { wch: 10 }, { wch: 15 }, { wch: 20 }, { wch: 15 }, { wch: 15 }, { wch: 12 }, { wch: 25 }, { wch: 25 }
        ];
        XLSX.utils.book_append_sheet(wb, wsMembers, 'Anggota Keluarga');
        
        const noKK = header.no_kk || 'unknown';
        XLSX.writeFile(wb, `kk_${noKK}.xlsx`);
        showToast('Excel exported successfully', 'success');
        
    } catch (error) {
        console.error('Excel export error:', error);
        showToast('Failed to export Excel', 'error');
    }
}

// ============================================
// Settings
// ============================================
function initSettings() {
    const apiBaseUrl = $('#apiBaseUrl');
    const pollingInterval = $('#pollingInterval');
    const themeSelect = $('#themeSelect');
    const autoSwitch = $('#autoSwitchResults');
    
    // Load saved values
    apiBaseUrl.value = AppState.config.apiBaseUrl;
    pollingInterval.value = AppState.config.pollingInterval;
    themeSelect.value = AppState.config.theme;
    autoSwitch.checked = AppState.config.autoSwitchResults;
    
    // Event handlers
    apiBaseUrl.addEventListener('change', (e) => {
        AppState.config.apiBaseUrl = e.target.value;
        localStorage.setItem('apiBaseUrl', e.target.value);
        showToast('API URL updated', 'success');
    });
    
    pollingInterval.addEventListener('change', (e) => {
        const value = Math.max(1000, Math.min(10000, parseInt(e.target.value)));
        e.target.value = value;
        AppState.config.pollingInterval = value;
        localStorage.setItem('pollingInterval', value);
        showToast('Polling interval updated', 'success');
    });
    
    themeSelect.addEventListener('change', (e) => {
        AppState.config.theme = e.target.value;
        localStorage.setItem('theme', e.target.value);
        applyTheme(e.target.value);
    });
    
    autoSwitch.addEventListener('change', (e) => {
        AppState.config.autoSwitchResults = e.target.checked;
        localStorage.setItem('autoSwitchResults', e.target.checked);
    });
}

// ============================================
// Utility Functions
// ============================================
function validateFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
        showToast(`Invalid file type: ${file.name}`, 'error');
        return false;
    }
    
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showToast(`File too large: ${file.name} (max 10MB)`, 'error');
        return false;
    }
    
    return true;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function createPDFPlaceholder() {
    return 'data:image/svg+xml;base64,' + btoa(`
        <svg width="150" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect width="150" height="100" fill="#f1f5f9" rx="8"/>
            <text x="50%" y="45%" font-family="Arial" font-size="24" text-anchor="middle" fill="#64748b">📄</text>
            <text x="50%" y="70%" font-family="Arial" font-size="12" text-anchor="middle" fill="#64748b">PDF</text>
        </svg>
    `);
}

function downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// Toast Notifications
// ============================================
function showToast(message, type = 'info') {
    const container = $('#toastContainer');
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type]}</span>
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }
    }, 4000);
}

// Make functions globally available
window.removeFileFromBatch = removeFileFromBatch;
window.showResultDetail = showResultDetail;
window.deleteJob = deleteJob;
