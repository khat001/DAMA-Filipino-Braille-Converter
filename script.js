document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const selectBtn = document.getElementById('select-btn');
    const convertBtn = document.getElementById('convert-btn');
    const previewDisplay = document.getElementById('preview-display');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const previewImage = document.getElementById('preview-image');
    const outputText = document.getElementById('output-text');
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    // Handle file selection via button
    selectBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    // Process uploaded files
    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        if (!file.type.match('image.*')) {
            alert('Please upload an image file.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewPlaceholder.classList.add('hidden');
            previewImage.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
    
    // Convert button handler (simulated conversion)
    convertBtn.addEventListener('click', function() {
        if (!previewImage.src || previewImage.classList.contains('hidden')) {
            showWarningOverlay();
            return;
        }
        
        const loadingOverlay = document.getElementById('loading-overlay');
        loadingOverlay.classList.remove('hidden');
        convertBtn.disabled = true;
        
        // Simulate API call with timeout
        setTimeout(() => {
            // This would be replaced with actual braille recognition API call
            const sampleText = "This is a simulated braille conversion result.\n\nThe quick brown fox jumps over the lazy dog.\n\nBraille is a tactile writing system used by people who are visually impaired.";
            
            outputText.innerHTML = `<p class="text-gray-800 whitespace-pre-line">${sampleText}</p>`;
            loadingOverlay.classList.add('hidden');
            convertBtn.disabled = false;
            
            // Enable buttons
            copyBtn.disabled = false;
            saveBtn.disabled = false;
            downloadBtn.disabled = false;
            copyBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            saveBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            downloadBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            
            // Add to history
            addToHistory(sampleText.substring(0, 50) + '...');
        }, 1500);
    });
    
    // Add converted item to history
    function addToHistory(text) {
        const historyList = document.getElementById('history-list');
        const todaySection = historyList.querySelector('div:first-child');
        
        const historyItem = document.createElement('div');
        historyItem.className = 'bg-primary-50 rounded-lg p-3 mt-2 cursor-pointer hover:bg-primary-100 transition-colors';
        historyItem.innerHTML = `
            <p class="text-gray-800 font-medium truncate">${text}</p>
            <p class="text-xs text-gray-500 mt-1">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</p>
        `;
        
        if (todaySection) {
            todaySection.appendChild(historyItem);
        } else {
            const newDaySection = document.createElement('div');
            newDaySection.className = 'border-b border-gray-100 pb-4';
            newDaySection.innerHTML = `<p class="text-sm text-gray-500">Today</p>`;
            newDaySection.appendChild(historyItem);
            historyList.prepend(newDaySection);
        }
    }
    
    // Button elements
    const copyBtn = document.getElementById('copy-btn');
    const saveBtn = document.getElementById('save-records-btn');
    const downloadBtn = document.getElementById('download-btn');
    
    // Disable buttons initially
    copyBtn.disabled = true;
    saveBtn.disabled = true;
    downloadBtn.disabled = true;
    copyBtn.classList.add('opacity-50', 'cursor-not-allowed');
    saveBtn.classList.add('opacity-50', 'cursor-not-allowed');
    downloadBtn.classList.add('opacity-50', 'cursor-not-allowed');
    
    // Copy text functionality
    copyBtn.addEventListener('click', function() {
        const textContent = outputText.textContent;
        if (textContent && textContent !== 'Your converted braille text will appear here...') {
            navigator.clipboard.writeText(textContent).then(() => {
                showSuccessPopup('Text copied successfully!');
            });
        }
    });
    
    // Save functionality
    saveBtn.addEventListener('click', function() {
        const textContent = outputText.textContent;
        if (textContent && textContent !== 'Your converted braille text will appear here...') {
            showSaveConfirm();
        }
    });
    
    // Save confirmation popup
    function showSaveConfirm() {
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        overlay.innerHTML = `
            <div class="bg-white rounded-xl p-8 text-center max-w-md mx-4">
                <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i data-feather="save" class="w-8 h-8 text-blue-500"></i>
                </div>
                <h3 class="text-lg font-semibold text-gray-800 mb-2">Save File</h3>
                <p class="text-gray-600 mb-6">Are you sure you want to save this file?</p>
                <div class="flex gap-3 justify-center">
                    <button id="cancel-save" class="bg-gray-300 hover:bg-gray-400 text-gray-800 font-medium py-2 px-6 rounded-lg transition-colors">
                        Cancel
                    </button>
                    <button id="confirm-save" class="bg-primary-500 hover:bg-primary-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                        Yes, Save
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        feather.replace();
        
        document.getElementById('cancel-save').addEventListener('click', () => overlay.remove());
        document.getElementById('confirm-save').addEventListener('click', () => {
            overlay.remove();
            showSavingProgress();
        });
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.remove();
        });
    }
    
    // Saving progress popup
    function showSavingProgress() {
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        overlay.innerHTML = `
            <div class="bg-white rounded-xl p-8 text-center max-w-md mx-4">
                <div class="animate-spin rounded-full h-16 w-16 border-b-2 border-primary-500 mx-auto mb-4"></div>
                <h3 class="text-lg font-semibold text-gray-800 mb-2">Saving...</h3>
                <p class="text-gray-600">Please wait</p>
            </div>
        `;
        document.body.appendChild(overlay);
        
        setTimeout(() => {
            overlay.remove();
            showSuccessPopup('Record saved successfully!');
        }, 1500);
    }
    
    // Download text functionality
    downloadBtn.addEventListener('click', function() {
        const textContent = outputText.textContent;
        if (textContent && textContent !== 'Your converted braille text will appear here...') {
            showDownloadConfirm(textContent);
        }
    });
    
    // Success popup
    function showSuccessPopup(message) {
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        overlay.innerHTML = `
            <div class="bg-white rounded-xl p-8 text-center max-w-md mx-4">
                <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i data-feather="check-circle" class="w-8 h-8 text-green-500"></i>
                </div>
                <h3 class="text-lg font-semibold text-gray-800 mb-2">Success!</h3>
                <p class="text-gray-600 mb-4">${message}</p>
                <button class="bg-primary-500 hover:bg-primary-700 text-white font-medium py-2 px-6 rounded-lg transition-colors" onclick="this.closest('.fixed').remove()">
                    OK
                </button>
            </div>
        `;
        document.body.appendChild(overlay);
        feather.replace();
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.remove();
        });
    }
    
    // Download confirmation popup
    function showDownloadConfirm(textContent) {
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        overlay.innerHTML = `
            <div class="bg-white rounded-xl p-8 text-center max-w-md mx-4">
                <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i data-feather="download" class="w-8 h-8 text-blue-500"></i>
                </div>
                <h3 class="text-lg font-semibold text-gray-800 mb-2">Download File</h3>
                <p class="text-gray-600 mb-6">Are you sure you want to download this file?</p>
                <div class="flex gap-3 justify-center">
                    <button id="cancel-download" class="bg-gray-300 hover:bg-gray-400 text-gray-800 font-medium py-2 px-6 rounded-lg transition-colors">
                        Cancel
                    </button>
                    <button id="confirm-download" class="bg-primary-500 hover:bg-primary-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                        Yes, Download
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        feather.replace();
        
        document.getElementById('cancel-download').addEventListener('click', () => overlay.remove());
        document.getElementById('confirm-download').addEventListener('click', () => {
            const blob = new Blob([textContent], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'braille-conversion.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            overlay.remove();
        });
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.remove();
        });
    }
    
    // Show warning overlay
    function showWarningOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        overlay.innerHTML = `
            <div class="bg-white rounded-xl p-8 text-center max-w-md mx-4">
                <div class="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i data-feather="alert-circle" class="w-8 h-8 text-red-500"></i>
                </div>
                <h3 class="text-lg font-semibold text-gray-800 mb-2">No Image Uploaded</h3>
                <p class="text-gray-600 mb-4">Please upload an image first before converting.</p>
                <button class="bg-primary-500 hover:bg-primary-700 text-white font-medium py-2 px-6 rounded-lg transition-colors" onclick="this.closest('.fixed').remove()">
                    OK
                </button>
            </div>
        `;
        document.body.appendChild(overlay);
        feather.replace();
        
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                overlay.remove();
            }
        });
    }
    
    // Sidebar toggle functionality
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');
    const sidebarClose = document.getElementById('sidebar-close');
    const sidebarOverlay = document.getElementById('sidebar-overlay');
    const mainContent = document.getElementById('main-content');
    
    function openSidebar() {
        sidebar.classList.remove('-translate-x-full');
        sidebarOverlay.classList.remove('hidden');
        feather.replace();
    }
    
    function closeSidebar() {
        sidebar.classList.add('-translate-x-full');
        sidebarOverlay.classList.add('hidden');
    }
    
    menuToggle.addEventListener('click', openSidebar);
    sidebarClose.addEventListener('click', closeSidebar);
    sidebarOverlay.addEventListener('click', closeSidebar);
    
    // Initialize tooltips for icons
    feather.replace();
});