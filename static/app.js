// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINT = `${API_BASE_URL}/v2/extract/kk`;

// Global state
let currentFile = null;
let resultData = null;

// Helper function: pick first available key from object
function pick(obj, keys, fallback = '-') {
    if (!obj) return fallback;
    for (const key of keys) {
        if (obj.hasOwnProperty(key) && obj[key] !== null && obj[key] !== undefined) {
            return obj[key];
        }
    }
    return fallback;
}

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const selectFileBtn = document.getElementById('selectFileBtn');
const previewSection = document.getElementById('previewSection');
const previewImg = document.getElementById('previewImg');
const fileDetails = document.getElementById('fileDetails');
const changeFileBtn = document.getElementById('changeFileBtn');
const processBtn = document.getElementById('processBtn');
const processSection = document.getElementById('processSection');
const processingStatus = document.getElementById('processingStatus');
const progressBar = document.getElementById('progressBar');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');
const downloadJsonBtn = document.getElementById('downloadJsonBtn');
const copyJsonBtn = document.getElementById('copyJsonBtn');
const newUploadBtn = document.getElementById('newUploadBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    selectFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Buttons
    changeFileBtn.addEventListener('click', resetUpload);
    processBtn.addEventListener('click', processDocument);
    retryBtn.addEventListener('click', () => {
        hideSection(errorSection);
        showSection(previewSection);
    });
    newUploadBtn.addEventListener('click', resetUpload);
    downloadJsonBtn.addEventListener('click', downloadJson);
    copyJsonBtn.addEventListener('click', copyJson);
    
    // Export Excel button
    const exportExcelBtn = document.getElementById('exportExcelBtn');
    if (exportExcelBtn) {
        exportExcelBtn.addEventListener('click', exportToExcel);
    }
}

// File Selection Handlers
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

// File Validation and Preview
function validateAndPreviewFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
        showError('Format file tidak didukung. Gunakan JPEG, PNG, atau PDF.');
        return;
    }

    // Validate file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('Ukuran file terlalu besar. Maksimal 10MB.');
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        if (file.type.startsWith('image/')) {
            previewImg.src = e.target.result;
        } else {
            // For PDF, show placeholder
            previewImg.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgZmlsbD0iI0Y5RkFGQiIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iNDgiIGZpbGw9IiM2QjcyODAiIHRleHQtYW5jaG9yPSJtaWRkbGUiPvCfk4Q8L3RleHQ+PHRleHQgeD0iNTAlIiB5PSI3MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iIzZCNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+UERGIEZpbGU8L3RleHQ+PC9zdmc+';
        }

        // Show file details
        fileDetails.innerHTML = `
            <div><strong>Nama File:</strong> ${file.name}</div>
            <div><strong>Ukuran:</strong> ${formatFileSize(file.size)}</div>
            <div><strong>Tipe:</strong> ${file.type}</div>
        `;

        // Hide upload area, show preview
        hideSection(uploadArea);
        showSection(previewSection);
    };

    reader.readAsDataURL(file);
}

// Process Document
async function processDocument() {
    if (!currentFile) {
        showError('Tidak ada file yang dipilih.');
        return;
    }

    // Hide preview, show processing
    hideSection(previewSection);
    showSection(processSection);

    // Reset processing UI
    updateProcessingStatus('Mengunggah file...', 10);
    resetProcessingSteps();

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', currentFile);

        // Update status
        updateProcessingStatus('Melakukan YOLO detection...', 30);
        setStepActive('step1');

        // Make API request
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Gagal memproses dokumen');
        }

        // Update status
        updateProcessingStatus('Melakukan enhancement dengan U-Net...', 60);
        setStepCompleted('step1');
        setStepActive('step2');

        // Simulate delay for better UX
        await new Promise(resolve => setTimeout(resolve, 500));

        // Update status
        updateProcessingStatus('Melakukan ekstraksi data dengan VLM...', 80);
        setStepCompleted('step2');
        setStepActive('step3');

        // Get result
        const result = await response.json();
        resultData = result;

        // Update status
        updateProcessingStatus('Selesai!', 100);
        setStepCompleted('step3');

        // Wait a moment then show results
        await new Promise(resolve => setTimeout(resolve, 500));

        // Hide processing, show results
        hideSection(processSection);
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        hideSection(processSection);
        showError(error.message || 'Terjadi kesalahan saat memproses dokumen');
    }
}

// Display Results
function displayResults(data) {
    // Metadata
        // store globally
        resultData = data;

        // Metadata
        const metadataGrid = document.getElementById('metadataGrid');
        const md = data.metadata || {};
        const sourceFile = pick(md, ['source_filename', 'source_file', 'sourceFile', 'sourceFileName'], '-');
        const processingTime = pick(md, ['processing_timestamp', 'processing_time', 'timestamp'], '-');
        const yoloVer = pick(md, ['model_version_yolo', 'model_version_yolo'], '-');
        const unetVer = pick(md, ['model_version_unet', 'model_version_unet'], '-');
        const vlmVer = pick(md, ['model_version_vlm', 'model_version_vlm', 'model_version_vlm_local'], '-');

        metadataGrid.innerHTML = `
            <div class="data-item">
                <div class="data-label">Nama File</div>
                <div class="data-value">${sourceFile}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Waktu Proses</div>
                <div class="data-value">${processingTime}</div>
            </div>
            <div class="data-item">
                <div class="data-label">YOLO Version</div>
                <div class="data-value">${yoloVer}</div>
            </div>
            <div class="data-item">
                <div class="data-label">VLM Version</div>
                <div class="data-value">${vlmVer}</div>
            </div>
        `;

    // Header
    const headerGrid = document.getElementById('headerGrid');
        const header = data.header || {};
        const desa = pick(header, ['desa', 'desa_kelurahan', 'desa_kel'], '-');
        headerGrid.innerHTML = `
            <div class="data-item">
                <div class="data-label">No. KK</div>
                <div class="data-value">${pick(header, ['no_kk', 'noKK'], '-')}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Kepala Keluarga</div>
                <div class="data-value">${pick(header, ['kepala_keluarga', 'kepala_keluarga'])}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Alamat</div>
                <div class="data-value">${pick(header, ['alamat', 'address'])}</div>
            </div>
            <div class="data-item">
                <div class="data-label">RT/RW</div>
                <div class="data-value">${pick(header, ['rt'])}/${pick(header, ['rw'])}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Desa/Kelurahan</div>
                <div class="data-value">${desa}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Kecamatan</div>
                <div class="data-value">${pick(header, ['kecamatan'])}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Kabupaten/Kota</div>
                <div class="data-value">${pick(header, ['kabupaten_kota', 'kabupaten'])}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Provinsi</div>
                <div class="data-value">${pick(header, ['provinsi'])}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Kode Pos</div>
                <div class="data-value">${pick(header, ['kode_pos', 'kodepos', 'postal_code'])}</div>
            </div>
        `;

    // Family Members
    const memberCount = document.getElementById('memberCount');
    const membersContainer = document.getElementById('membersContainer');
        const members = data.anggota_keluarga || data.anggota || [];
        memberCount.textContent = members.length;

        membersContainer.innerHTML = members.map((member, index) => {
            const pekerjaan = pick(member, ['jenis_pekerjaan', 'pekerjaan', 'jenis_pekerjaan'], '-');
            return `
            <div class="member-card">
                <div class="member-header">
                    <div class="member-name">${pick(member, ['nama_lengkap', 'name'], 'Tidak Ada Nama')}</div>
                    <div class="member-badge">Anggota ${index + 1}</div>
                </div>
                <div class="member-details">
                    <div class="data-item"><div class="data-label">NIK</div><div class="data-value">${pick(member, ['nik'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Jenis Kelamin</div><div class="data-value">${pick(member, ['jenis_kelamin', 'gender'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Tempat Lahir</div><div class="data-value">${pick(member, ['tempat_lahir', 'birth_place'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Tanggal Lahir</div><div class="data-value">${pick(member, ['tanggal_lahir', 'birth_date'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Agama</div><div class="data-value">${pick(member, ['agama'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Pendidikan</div><div class="data-value">${pick(member, ['pendidikan'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Pekerjaan</div><div class="data-value">${pekerjaan}</div></div>
                    <div class="data-item"><div class="data-label">Status Perkawinan</div><div class="data-value">${pick(member, ['status_perkawinan'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Status Keluarga</div><div class="data-value">${pick(member, ['status_keluarga'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Kewarganegaraan</div><div class="data-value">${pick(member, ['kewarganegaraan'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Nama Ayah</div><div class="data-value">${pick(member, ['nama_ayah'], '-')}</div></div>
                    <div class="data-item"><div class="data-label">Nama Ibu</div><div class="data-value">${pick(member, ['nama_ibu'], '-')}</div></div>
                </div>
            </div>`;
        }).join('');

    // Footer
    const footerGrid = document.getElementById('footerGrid');
    const footer = data.footer;
        const tanda_kepala = pick(footer, ['tanda_tangan_kepala_keluarga', 'tanda_tangan_kepala'], {});
        const tanda_pejabat = pick(footer, ['tanda_tangan_pejabat', 'tanda_tangan_pejabat'], {});
        const kepalaDinas = pick(footer, ['kepala_dinas'], '-');

        footerGrid.innerHTML = `
            <div class="data-item">
                <div class="data-label">Tanggal Pembuatan</div>
                <div class="data-value">${pick(header, ['tanggal_pembuatan'], '-')}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Tanda Tangan Kepala Keluarga</div>
                <div class="data-value">${(typeof tanda_kepala === 'object' ? (tanda_kepala.text || '-') : tanda_kepala) || '-'}</div>
            </div>
            <div class="data-item">
                <div class="data-label">Tanda Tangan Pejabat</div>
                <div class="data-value">${(typeof tanda_pejabat === 'object' ? (tanda_pejabat.text || '-') : tanda_pejabat) || '-'}</div>
            </div>
        `;

    // Raw JSON
    const rawJson = document.getElementById('rawJson');
    rawJson.textContent = JSON.stringify(data, null, 2);

    // Show results section
    showSection(resultsSection);
}

// Processing UI Helpers
function updateProcessingStatus(status, progress) {
    processingStatus.textContent = status;
    progressBar.style.width = `${progress}%`;
}

function resetProcessingSteps() {
    ['step1', 'step2', 'step3'].forEach(id => {
        const step = document.getElementById(id);
        step.classList.remove('active', 'completed');
    });
}

function setStepActive(stepId) {
    const step = document.getElementById(stepId);
    step.classList.add('active');
}

function setStepCompleted(stepId) {
    const step = document.getElementById(stepId);
    step.classList.remove('active');
    step.classList.add('completed');
}

// Download and Copy Functions
function downloadJson() {
    if (!resultData) return;

    const blob = new Blob([JSON.stringify(resultData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `kk_extraction_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast('JSON berhasil diunduh!');
}

function copyJson() {
    if (!resultData) return;

    const jsonText = JSON.stringify(resultData, null, 2);
    navigator.clipboard.writeText(jsonText).then(() => {
        showToast('JSON berhasil disalin ke clipboard!');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showToast('Gagal menyalin JSON');
    });
}

// Reset Upload
function resetUpload() {
    currentFile = null;
    resultData = null;
    fileInput.value = '';
    
    hideSection(previewSection);
    hideSection(processSection);
    hideSection(resultsSection);
    hideSection(errorSection);
    showSection(uploadArea);
}

// Error Handling
function showError(message) {
    errorMessage.textContent = message;
    hideSection(uploadArea);
    hideSection(previewSection);
    hideSection(processSection);
    hideSection(resultsSection);
    showSection(errorSection);
}

// UI Helpers
function showSection(element) {
    element.style.display = 'block';
}

function hideSection(element) {
    element.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function showToast(message) {
    // Simple toast notification
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: #10B981;
        color: white;
        padding: 15px 25px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => document.body.removeChild(toast), 300);
    }, 3000);
}

// Add CSS for toast animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Export to Excel Function
function exportToExcel() {
    if (!resultData) {
        showToast('❌ Tidak ada data untuk diexport', 'error');
        return;
    }

    try {
        // Create workbook
        const wb = XLSX.utils.book_new();

        // Sheet 1: Header KK (use pick() for robust keys)

        const hdr = resultData.header || {};
        const headerData = [
            ['Informasi Kartu Keluarga', ''],
            ['No. KK', pick(hdr, ['no_kk', 'noKK'], '-')],
            ['Kepala Keluarga', pick(hdr, ['kepala_keluarga', 'kepala_keluarga'], '-')],
            ['Alamat', pick(hdr, ['alamat', 'address'], '-')],
            ['RT/RW', `${pick(hdr, ['rt'], '-')}/${pick(hdr, ['rw'], '-')}`],
            ['Desa/Kelurahan', pick(hdr, ['desa', 'desa_kelurahan', 'desa_kel'], '-')],
            ['Kecamatan', pick(hdr, ['kecamatan'], '-')],
            ['Kabupaten/Kota', pick(hdr, ['kabupaten_kota', 'kabupaten'], '-')],
            ['Provinsi', pick(hdr, ['provinsi'], '-')],
            ['Kode Pos', pick(hdr, ['kode_pos', 'kodepos', 'postal_code'], '-')],
            ['Tanggal Pembuatan', pick(hdr, ['tanggal_pembuatan'], '-')]
        ];
        const wsHeader = XLSX.utils.aoa_to_sheet(headerData);
        
        // Set column widths
        wsHeader['!cols'] = [
            { wch: 25 },
            { wch: 40 }
        ];
        
        XLSX.utils.book_append_sheet(wb, wsHeader, 'Header KK');

        // Sheet 2: Anggota Keluarga
        if ((resultData.anggota_keluarga && resultData.anggota_keluarga.length > 0) || (resultData.anggota && resultData.anggota.length > 0)) {
            const members = resultData.anggota_keluarga || resultData.anggota || [];
            
            // Create headers
            const memberHeaders = [
                'No',
                'NIK',
                'Nama Lengkap',
                'Jenis Kelamin',
                'Tempat Lahir',
                'Tanggal Lahir',
                'Agama',
                'Pendidikan',
                'Pekerjaan',
                'Status Perkawinan',
                'Status Keluarga',
                'Kewarganegaraan',
                'Nama Ayah',
                'Nama Ibu',
                'No. Paspor',
                'No. KITAP'
            ];

            // Create data rows
            const memberRows = members.map((member, index) => [
                index + 1,
                member.nik || '-',
                member.nama_lengkap || '-',
                member.jenis_kelamin || '-',
                member.tempat_lahir || '-',
                member.tanggal_lahir || '-',
                member.agama || '-',
                member.pendidikan || '-',
                pick(member, ['jenis_pekerjaan', 'pekerjaan', 'jenis_pekerjaan'], '-'),
                member.status_perkawinan || '-',
                member.status_keluarga || '-',
                member.kewarganegaraan || '-',
                member.nama_ayah || '-',
                member.nama_ibu || '-',
                member.no_paspor || '-',
                member.no_KITAP || '-'
            ]);

            // Combine headers and data
            const memberData = [memberHeaders, ...memberRows];
            const wsMembers = XLSX.utils.aoa_to_sheet(memberData);
            
            // Set column widths
            wsMembers['!cols'] = [
                { wch: 5 },   // No
                { wch: 18 },  // NIK
                { wch: 25 },  // Nama
                { wch: 12 },  // Gender
                { wch: 20 },  // Tempat Lahir
                { wch: 12 },  // Tanggal Lahir
                { wch: 12 },  // Agama
                { wch: 20 },  // Pendidikan
                { wch: 25 },  // Pekerjaan
                { wch: 15 },  // Status Kawin
                { wch: 18 },  // Status Keluarga
                { wch: 15 },  // Kewarganegaraan
                { wch: 25 },  // Ayah
                { wch: 25 },  // Ibu
                { wch: 12 },  // Paspor
                { wch: 12 }   // KITAP
            ];
            
            XLSX.utils.book_append_sheet(wb, wsMembers, 'Anggota Keluarga');
        }

        // Sheet 3: Metadata
        const md = resultData.metadata || {};
        const metadataData = [
            ['Metadata Ekstraksi', ''],
            ['Waktu Proses', pick(md, ['processing_timestamp', 'processing_time', 'timestamp'], '-')],
            ['Model YOLO', pick(md, ['model_version_yolo'], '-')],
            ['Model U-Net', pick(md, ['model_version_unet'], '-')],
            ['Model VLM', pick(md, ['model_version_vlm', 'model_version_vlm_local'], '-')],
            ['File Sumber', pick(md, ['source_file', 'source_filename'], '-')]
        ];
        const wsMetadata = XLSX.utils.aoa_to_sheet(metadataData);
        wsMetadata['!cols'] = [
            { wch: 25 },
            { wch: 50 }
        ];
        XLSX.utils.book_append_sheet(wb, wsMetadata, 'Metadata');

        // Generate filename
        const noKK = resultData.header?.no_kk || 'KK';
        const timestamp = new Date().toISOString().slice(0, 10);
        const filename = `KK_${noKK}_${timestamp}.xlsx`;

        // Save file
        XLSX.writeFile(wb, filename);
        
        showToast('✅ Excel berhasil diexport!', 'success');
        
    } catch (error) {
        console.error('Export Excel error:', error);
        showToast('❌ Gagal export Excel: ' + error.message, 'error');
    }
}

// Download JSON Function (existing)
function downloadJson() {
    if (!resultData) {
        showToast('❌ Tidak ada data untuk didownload', 'error');
        return;
    }

    try {
        const dataStr = JSON.stringify(resultData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        
        const noKK = resultData.header?.no_kk || 'KK';
        const timestamp = new Date().toISOString().slice(0, 10);
        link.href = url;
        link.download = `KK_${noKK}_${timestamp}.json`;
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        showToast('✅ JSON berhasil didownload!', 'success');
    } catch (error) {
        showToast('❌ Gagal download JSON', 'error');
    }
}

// Copy JSON Function (existing)
function copyJson() {
    if (!resultData) {
        showToast('❌ Tidak ada data untuk dicopy', 'error');
        return;
    }

    try {
        const dataStr = JSON.stringify(resultData, null, 2);
        navigator.clipboard.writeText(dataStr).then(() => {
            showToast('✅ JSON berhasil dicopy ke clipboard!', 'success');
        }).catch(() => {
            showToast('❌ Gagal copy JSON', 'error');
        });
    } catch (error) {
        showToast('❌ Gagal copy JSON', 'error');
    }
}
