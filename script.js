// Power BI Embedding Configuration
let powerbiEmbed = null;

// DOM Elements
const configPanel = document.getElementById('configPanel');
const viewerContainer = document.getElementById('viewerContainer');
const embedUrlInput = document.getElementById('embedUrl');
const accessTokenInput = document.getElementById('accessToken');
const reportIdInput = document.getElementById('reportId');
const embedTypeSelect = document.getElementById('embedType');
const loadReportBtn = document.getElementById('loadReport');
const clearReportBtn = document.getElementById('clearReport');
const backToConfigBtn = document.getElementById('backToConfig');
const powerbiContainer = document.getElementById('powerbi-container');
const errorMessage = document.getElementById('errorMessage');

// Event Listeners
loadReportBtn.addEventListener('click', loadPowerBIReport);
clearReportBtn.addEventListener('click', clearReport);
backToConfigBtn.addEventListener('click', showConfigPanel);

// Load Power BI Report
function loadPowerBIReport() {
    const embedUrl = embedUrlInput.value.trim();
    const accessToken = accessTokenInput.value.trim();
    const reportId = reportIdInput.value.trim();
    const embedType = embedTypeSelect.value;

    // Validation
    if (!embedUrl) {
        showError('Please provide an Embed URL');
        return;
    }

    if (!accessToken) {
        showError('Please provide an Access Token');
        return;
    }

    // Hide error message
    hideError();

    // Show loading state
    loadReportBtn.disabled = true;
    loadReportBtn.innerHTML = '<span class="loading"></span>Loading...';

    try {
        // Extract report ID from embed URL if not provided
        let finalReportId = reportId;
        if (!finalReportId && embedUrl.includes('reportId=')) {
            const match = embedUrl.match(/reportId=([^&]+)/);
            if (match) {
                finalReportId = match[1];
            }
        }

        // Clear previous embed
        if (powerbiEmbed) {
            powerbiEmbed.reset();
        }
        powerbiContainer.innerHTML = '';

        // Configure embed settings
        const config = {
            type: embedType,
            tokenType: models.TokenType.Embed,
            accessToken: accessToken,
            embedUrl: embedUrl,
            id: finalReportId || undefined,
            settings: {
                panes: {
                    filters: {
                        expanded: false,
                        visible: true
                    },
                    pageNavigation: {
                        visible: true
                    }
                },
                background: models.BackgroundType.Transparent,
                layoutType: models.LayoutType.Custom,
                customLayout: {
                    displayOption: models.DisplayOption.FitToPage
                }
            }
        };

        // Embed the report
        powerbiEmbed = powerbi.embed(powerbiContainer, config);

        // Handle events
        powerbiEmbed.on('loaded', () => {
            console.log('Power BI report loaded successfully');
            loadReportBtn.disabled = false;
            loadReportBtn.innerHTML = 'Load Power BI Map';
            showViewer();
        });

        powerbiEmbed.on('error', (event) => {
            console.error('Power BI embed error:', event);
            showError('Failed to load Power BI report. Please check your credentials and try again.');
            loadReportBtn.disabled = false;
            loadReportBtn.innerHTML = 'Load Power BI Map';
        });

        powerbiEmbed.on('rendered', () => {
            console.log('Power BI report rendered');
        });

    } catch (error) {
        console.error('Error embedding Power BI:', error);
        showError('An error occurred while loading the report: ' + error.message);
        loadReportBtn.disabled = false;
        loadReportBtn.innerHTML = 'Load Power BI Map';
    }
}

// Clear Report
function clearReport() {
    if (powerbiEmbed) {
        powerbiEmbed.reset();
        powerbiEmbed = null;
    }
    powerbiContainer.innerHTML = '';
    embedUrlInput.value = '';
    accessTokenInput.value = '';
    reportIdInput.value = '';
    hideError();
    showConfigPanel();
}

// Show Viewer
function showViewer() {
    configPanel.style.display = 'none';
    viewerContainer.style.display = 'block';
    // Scroll to viewer
    viewerContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show Config Panel
function showConfigPanel() {
    viewerContainer.style.display = 'none';
    configPanel.style.display = 'block';
    configPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show Error Message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'flex';
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide Error Message
function hideError() {
    errorMessage.style.display = 'none';
}

// Handle Enter key in input fields
[embedUrlInput, accessTokenInput, reportIdInput].forEach(input => {
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            loadPowerBIReport();
        }
    });
});

// Check if Power BI SDK is loaded
if (typeof powerbi === 'undefined') {
    console.error('Power BI SDK not loaded. Check your internet connection.');
    showError('Power BI SDK failed to load. Please check your internet connection.');
}
