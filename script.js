// Updated script.js - Connected to your Braille Detection API

const API_BASE_URL = "http://localhost:5000/api";

document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("file-input");
  const selectBtn = document.getElementById("select-btn");
  const convertBtn = document.getElementById("convert-btn");
  const previewDisplay = document.getElementById("preview-display");
  const previewPlaceholder = document.getElementById("preview-placeholder");
  const previewImage = document.getElementById("preview-image");
  const outputText = document.getElementById("output-text");
  const detectionPreview = document.getElementById("detection-preview");
  const loadingOverlay = document.getElementById("loading-overlay");

  let currentFile = null;
  let currentLanguage = "english";

  // Language selection
  document
    .getElementById("english-radio")
    .addEventListener("change", function () {
      if (this.checked) currentLanguage = "english";
    });

  document
    .getElementById("filipino-radio")
    .addEventListener("change", function () {
      if (this.checked) currentLanguage = "filipino";
    });

  // Check API health on load
  checkAPIHealth();

  async function checkAPIHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      console.log("✅ API Status:", data);
    } catch (error) {
      console.error("❌ API not responding:", error);
      showErrorPopup(
        "Backend API is not running. Please start the Flask server."
      );
    }
  }

  // Prevent default drag behaviors
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  // Highlight drop area
  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(
      eventName,
      () => dropArea.classList.add("highlight"),
      false
    );
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(
      eventName,
      () => dropArea.classList.remove("highlight"),
      false
    );
  });

  // Handle dropped files
  dropArea.addEventListener(
    "drop",
    function (e) {
      const files = e.dataTransfer.files;
      handleFiles(files);
    },
    false
  );

  // Handle file selection via button
  selectBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", function () {
    handleFiles(this.files);
  });

  // Process uploaded files
  function handleFiles(files) {
    if (files.length === 0) return;

    const file = files[0];
    if (!file.type.match("image.*")) {
      showErrorPopup("Please upload an image file.");
      return;
    }

    currentFile = file;

    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      previewPlaceholder.classList.add("hidden");
      previewImage.classList.remove("hidden");

      // Reset output
      detectionPreview.innerHTML =
        '<p class="text-gray-400 italic text-sm">Detection preview will appear here</p>';
      outputText.innerHTML =
        '<p class="text-gray-500 italic">Your converted braille text will appear here...</p>';
    };
    reader.readAsDataURL(file);
  }

  // Convert button handler - REAL API CALL
  convertBtn.addEventListener("click", async function () {
    if (!currentFile) {
      showErrorPopup("Please upload an image first.");
      return;
    }

    // Show loading
    loadingOverlay.classList.remove("hidden");
    convertBtn.disabled = true;

    try {
      // Prepare form data
      const formData = new FormData();
      formData.append("image", currentFile);
      formData.append("language", currentLanguage);

      console.log("📤 Sending request to API...");

      // Call your backend API
      const response = await fetch(`${API_BASE_URL}/convert`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      console.log("✅ Response received:", result);

      if (result.success) {
        // Display results
        displayResults(result);
        showSuccessPopup("Conversion completed successfully!");
      } else {
        throw new Error(result.error || "Conversion failed");
      }
    } catch (error) {
      console.error("❌ Error:", error);
      showErrorPopup(`Conversion failed: ${error.message}`);
    } finally {
      loadingOverlay.classList.add("hidden");
      convertBtn.disabled = false;
    }
  });

  // Display conversion results
  function displayResults(result) {
    // Show annotated image with detections
    if (result.annotated_image) {
      detectionPreview.innerHTML = `
      <img src="${result.annotated_image}" 
           class="max-w-full max-h-full object-contain rounded-lg shadow-lg" 
           alt="Detection Preview">
    `;
    }

    // Show converted text (simplified)
    outputText.innerHTML = `
    <div class="space-y-4">
      <div class="bg-white rounded-lg p-4 border border-gray-300">
        <h4 class="font-semibold text-gray-800 mb-3">Converted Text:</h4>
        <div class="text-gray-800 whitespace-pre-wrap leading-relaxed">
          ${escapeHtml(result.text)}
        </div>
      </div>
    </div>
  `;

    feather.replace();

    // Store result for later use
    window.currentResult = result;
  }

  // Helper function to escape HTML
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Copy button
  document.getElementById("copy-btn").addEventListener("click", function () {
    if (window.currentResult && window.currentResult.text) {
      navigator.clipboard
        .writeText(window.currentResult.text)
        .then(() => {
          showSuccessPopup("Text copied to clipboard!");
        })
        .catch((err) => {
          showErrorPopup("Failed to copy text");
        });
    }
  });

  // Save button
  document
    .getElementById("save-records-btn")
    .addEventListener("click", function () {
      if (window.currentResult) {
        // Save to localStorage
        saveToHistory(window.currentResult);
        showSuccessPopup("Record saved to history!");
      }
    });

  // Download button
  document
    .getElementById("download-btn")
    .addEventListener("click", function () {
      if (window.currentResult && window.currentResult.text) {
        const blob = new Blob([window.currentResult.text], {
          type: "text/plain",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `braille-conversion-${new Date()
          .toISOString()
          .slice(0, 10)}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showSuccessPopup("File downloaded successfully!");
      }
    });

  // Save to localStorage
  function saveToHistory(result) {
    let history = JSON.parse(localStorage.getItem("brailleHistory") || "[]");

    history.unshift({
      timestamp: result.timestamp || new Date().toISOString(),
      text: result.text.substring(0, 200),
      fullText: result.text,
      stats: result.statistics,
      date: new Date().toLocaleString(),
    });

    // Keep only last 50 records
    history = history.slice(0, 50);

    localStorage.setItem("brailleHistory", JSON.stringify(history));
  }

  // Success popup
  function showSuccessPopup(message) {
    const overlay = document.createElement("div");
    overlay.className =
      "fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50";
    overlay.innerHTML = `
            <div class="bg-white rounded-xl p-8 text-center max-w-md mx-4 animate-fade-in">
                <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i data-feather="check-circle" class="w-8 h-8 text-green-500"></i>
                </div>
                <h3 class="text-lg font-semibold text-gray-800 mb-2">Success!</h3>
                <p class="text-gray-600 mb-4">${message}</p>
                <button class="bg-primary-500 hover:bg-primary-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                    OK
                </button>
            </div>
        `;
    document.body.appendChild(overlay);
    feather.replace();

    overlay
      .querySelector("button")
      .addEventListener("click", () => overlay.remove());
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });

    // Auto-remove after 3 seconds
    setTimeout(() => overlay.remove(), 3000);
  }

  // Error popup
  function showErrorPopup(message) {
    const overlay = document.createElement("div");
    overlay.className =
      "fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50";
    overlay.innerHTML = `
            <div class="bg-white rounded-xl p-8 text-center max-w-md mx-4">
                <div class="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i data-feather="alert-circle" class="w-8 h-8 text-red-500"></i>
                </div>
                <h3 class="text-lg font-semibold text-gray-800 mb-2">Error</h3>
                <p class="text-gray-600 mb-4">${message}</p>
                <button class="bg-primary-500 hover:bg-primary-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                    OK
                </button>
            </div>
        `;
    document.body.appendChild(overlay);
    feather.replace();

    overlay
      .querySelector("button")
      .addEventListener("click", () => overlay.remove());
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.remove();
    });
  }

  // Sidebar functionality
  const menuToggle = document.getElementById("menu-toggle");
  const sidebar = document.getElementById("sidebar");
  const sidebarClose = document.getElementById("sidebar-close");
  const sidebarOverlay = document.getElementById("sidebar-overlay");

  menuToggle.addEventListener("click", () => {
    sidebar.classList.remove("-translate-x-full");
    sidebarOverlay.classList.remove("hidden");
    feather.replace();
  });

  function closeSidebar() {
    sidebar.classList.add("-translate-x-full");
    sidebarOverlay.classList.add("hidden");
  }

  sidebarClose.addEventListener("click", closeSidebar);
  sidebarOverlay.addEventListener("click", closeSidebar);

  // Initialize
  feather.replace();
});
