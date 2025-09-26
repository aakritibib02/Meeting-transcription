/**
 * Popup JavaScript - Handles UI interactions and settings management
 */

class PopupManager {
    constructor() {
        this.isActive = false;
        this.init();
    }

    async init() {
        await this.loadSettings();
        this.setupEventListeners();
        this.updateUI();
    }

    /**
     * Load saved settings from Chrome storage
     */
    async loadSettings() {
        try {
            const settings = await chrome.storage.sync.get({
                inputLanguage: 'auto',
                outputLanguage: 'en',
                transcriptionEnabled: true,
                translationEnabled: true,
                taskDetectionEnabled: true,
                summaryEnabled: true,
                userEmail: '',
                autoEmailEnabled: false,
                isActive: false
            });

            // Apply settings to UI
            document.getElementById('inputLanguage').value = settings.inputLanguage;
            document.getElementById('outputLanguage').value = settings.outputLanguage;
            document.getElementById('transcriptionEnabled').checked = settings.transcriptionEnabled;
            document.getElementById('translationEnabled').checked = settings.translationEnabled;
            document.getElementById('taskDetectionEnabled').checked = settings.taskDetectionEnabled;
            document.getElementById('summaryEnabled').checked = settings.summaryEnabled;
            document.getElementById('userEmail').value = settings.userEmail;
            document.getElementById('autoEmailEnabled').checked = settings.autoEmailEnabled;
            
            this.isActive = settings.isActive;
            this.updateStatusDisplay();
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    /**
     * Save settings to Chrome storage
     */
    async saveSettings() {
        try {
            const settings = {
                inputLanguage: document.getElementById('inputLanguage').value,
                outputLanguage: document.getElementById('outputLanguage').value,
                transcriptionEnabled: document.getElementById('transcriptionEnabled').checked,
                translationEnabled: document.getElementById('translationEnabled').checked,
                taskDetectionEnabled: document.getElementById('taskDetectionEnabled').checked,
                summaryEnabled: document.getElementById('summaryEnabled').checked,
                userEmail: document.getElementById('userEmail').value,
                autoEmailEnabled: document.getElementById('autoEmailEnabled').checked,
                isActive: this.isActive
            };

            await chrome.storage.sync.set(settings);
            
            // Notify background script of settings change
            await chrome.runtime.sendMessage({
                action: 'settingsUpdated',
                settings: settings
            });
        } catch (error) {
            console.error('Error saving settings:', error);
        }
    }

    /**
     * Setup event listeners for all interactive elements
     */
    setupEventListeners() {
        // Settings change listeners
        const settingsInputs = [
            'inputLanguage', 'outputLanguage', 'transcriptionEnabled',
            'translationEnabled', 'taskDetectionEnabled', 'summaryEnabled',
            'userEmail', 'autoEmailEnabled'
        ];

        settingsInputs.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => this.saveSettings());
            }
        });

        // Control buttons
        document.getElementById('startBtn').addEventListener('click', () => this.startAssistant());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopAssistant());

        // Quick action buttons
        document.getElementById('exportSummary').addEventListener('click', () => this.exportSummary());
        document.getElementById('viewTasks').addEventListener('click', () => this.viewTasks());
        document.getElementById('clearHistory').addEventListener('click', () => this.clearHistory());

        // Listen for messages from background script
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
        });
    }

    /**
     * Start the meeting assistant
     */
    async startAssistant() {
        try {
            // Check if we're on a supported meeting platform
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const isMeetingPlatform = this.isSupportedPlatform(tab.url);

            if (!isMeetingPlatform) {
                this.showNotification('Please open a supported meeting platform (Zoom, Google Meet, Teams, etc.)');
                return;
            }

            // Send start command to background script
            await chrome.runtime.sendMessage({ action: 'startAssistant', tabId: tab.id });
            
            this.isActive = true;
            await this.saveSettings();
            this.updateUI();
            
            this.showNotification('Meeting Assistant started!');
        } catch (error) {
            console.error('Error starting assistant:', error);
            this.showNotification('Error starting assistant. Please try again.');
        }
    }

    /**
     * Stop the meeting assistant
     */
    async stopAssistant() {
        try {
            await chrome.runtime.sendMessage({ action: 'stopAssistant' });
            
            this.isActive = false;
            await this.saveSettings();
            this.updateUI();
            
            this.showNotification('Meeting Assistant stopped.');
        } catch (error) {
            console.error('Error stopping assistant:', error);
        }
    }

    /**
     * Export meeting summary
     */
    async exportSummary() {
        try {
            const response = await chrome.runtime.sendMessage({ action: 'exportSummary' });
            
            if (response.success && response.summary) {
                this.downloadFile(response.summary, `meeting-summary-${new Date().toISOString().split('T')[0]}.txt`);
                this.showNotification('Summary exported successfully!');
            } else {
                this.showNotification('No summary available to export.');
            }
        } catch (error) {
            console.error('Error exporting summary:', error);
            this.showNotification('Error exporting summary.');
        }
    }

    /**
     * View detected tasks
     */
    async viewTasks() {
        try {
            const response = await chrome.runtime.sendMessage({ action: 'getTasks' });
            
            if (response.success && response.tasks && response.tasks.length > 0) {
                this.showTasksModal(response.tasks);
            } else {
                this.showNotification('No tasks detected yet.');
            }
        } catch (error) {
            console.error('Error getting tasks:', error);
        }
    }

    /**
     * Clear history and stored data
     */
    async clearHistory() {
        if (confirm('Are you sure you want to clear all meeting history and data?')) {
            try {
                await chrome.runtime.sendMessage({ action: 'clearHistory' });
                this.showNotification('History cleared successfully!');
            } catch (error) {
                console.error('Error clearing history:', error);
                this.showNotification('Error clearing history.');
            }
        }
    }

    /**
     * Handle messages from background script
     */
    handleMessage(message, sender, sendResponse) {
        switch (message.action) {
            case 'statusUpdate':
                this.updateStatus(message.status);
                break;
            case 'notification':
                this.showNotification(message.message);
                break;
        }
    }

    /**
     * Update UI based on current state
     */
    updateUI() {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (this.isActive) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }

        this.updateStatusDisplay();
    }

    /**
     * Update status display
     */
    updateStatusDisplay() {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');

        if (this.isActive) {
            statusDot.className = 'status-dot processing';
            statusText.textContent = 'Active';
        } else {
            statusDot.className = 'status-dot';
            statusText.textContent = 'Ready';
        }
    }

    /**
     * Update status from background script
     */
    updateStatus(status) {
        const statusText = document.getElementById('statusText');
        const statusDot = document.getElementById('statusDot');

        statusText.textContent = status;
        
        if (status === 'Listening' || status === 'Processing') {
            statusDot.className = 'status-dot processing';
        } else if (status === 'Active') {
            statusDot.className = 'status-dot';
        } else {
            statusDot.className = 'status-dot inactive';
        }
    }

    /**
     * Check if URL is a supported meeting platform
     */
    isSupportedPlatform(url) {
        const supportedDomains = [
            'zoom.us', 'meet.google.com', 'teams.microsoft.com',
            'webex.com', 'gotomeeting.com', 'skype.com'
        ];

        return supportedDomains.some(domain => url.includes(domain));
    }

    /**
     * Show notification to user
     */
    showNotification(message) {
        // Create temporary notification element
        const notification = document.createElement('div');
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s;
        `;

        document.body.appendChild(notification);
        
        // Fade in
        requestAnimationFrame(() => {
            notification.style.opacity = '1';
        });

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    /**
     * Download file with given content
     */
    downloadFile(content, filename) {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }

    /**
     * Show tasks in a modal-like interface
     */
    showTasksModal(tasks) {
        // Simple tasks display - in a real app, this could be a proper modal
        let taskText = 'Detected Tasks:\\n\\n';
        tasks.forEach((task, index) => {
            taskText += `${index + 1}. ${task.description}\\n`;
            taskText += `   Assignee: ${task.assignee || 'Unassigned'}\\n`;
            taskText += `   Priority: ${task.priority || 'Normal'}\\n\\n`;
        });

        alert(taskText);
    }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PopupManager();
});