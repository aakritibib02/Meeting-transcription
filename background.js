/**
 * Background Script (Service Worker) - Manages extension lifecycle and coordination
 */

class MeetingAssistantBackground {
    constructor() {
        this.isActive = false;
        this.currentTabId = null;
        this.meetingData = {
            transcriptions: [],
            translations: [],
            tasks: [],
            summary: '',
            startTime: null
        };
        this.settings = {};
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSettings();
    }

    /**
     * Setup event listeners for extension events
     */
    setupEventListeners() {
        // Extension installation and startup
        chrome.runtime.onInstalled.addListener((details) => {
            console.log('Meeting Assistant installed:', details.reason);
            this.initializeExtension();
        });

        // Handle messages from popup and content scripts
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open for async responses
        });

        // Tab updates and navigation
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            this.handleTabUpdate(tabId, changeInfo, tab);
        });

        // Tab removal
        chrome.tabs.onRemoved.addListener((tabId) => {
            if (tabId === this.currentTabId) {
                this.stopAssistant();
            }
        });

        // Window focus changes
        chrome.windows.onFocusChanged.addListener((windowId) => {
            this.handleWindowFocusChange(windowId);
        });
    }

    /**
     * Initialize extension settings and data
     */
    async initializeExtension() {
        try {
            // Set default settings if not already present
            const defaultSettings = {
                inputLanguage: 'auto',
                outputLanguage: 'en',
                transcriptionEnabled: true,
                translationEnabled: true,
                taskDetectionEnabled: true,
                summaryEnabled: true,
                userEmail: '',
                autoEmailEnabled: false,
                isActive: false
            };

            const existingSettings = await chrome.storage.sync.get(defaultSettings);
            await chrome.storage.sync.set(existingSettings);
            
            this.settings = existingSettings;
        } catch (error) {
            console.error('Error initializing extension:', error);
        }
    }

    /**
     * Load settings from storage
     */
    async loadSettings() {
        try {
            this.settings = await chrome.storage.sync.get();
            this.isActive = this.settings.isActive || false;
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    /**
     * Handle messages from popup and content scripts
     */
    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.action) {
                case 'startAssistant':
                    await this.startAssistant(message.tabId);
                    sendResponse({ success: true });
                    break;

                case 'stopAssistant':
                    await this.stopAssistant();
                    sendResponse({ success: true });
                    break;

                case 'settingsUpdated':
                    await this.updateSettings(message.settings);
                    sendResponse({ success: true });
                    break;

                case 'transcriptionData':
                    await this.processTranscription(message.data, sender.tab.id);
                    sendResponse({ success: true });
                    break;

                case 'exportSummary':
                    const summary = await this.generateMeetingSummary();
                    sendResponse({ success: true, summary: summary });
                    break;

                case 'getTasks':
                    sendResponse({ success: true, tasks: this.meetingData.tasks });
                    break;

                case 'clearHistory':
                    await this.clearMeetingHistory();
                    sendResponse({ success: true });
                    break;

                case 'getStatus':
                    sendResponse({ 
                        success: true, 
                        isActive: this.isActive,
                        currentTab: this.currentTabId 
                    });
                    break;

                default:
                    sendResponse({ success: false, error: 'Unknown action' });
            }
        } catch (error) {
            console.error('Error handling message:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    /**
     * Start the meeting assistant
     */
    async startAssistant(tabId) {
        try {
            if (this.isActive && this.currentTabId === tabId) {
                return; // Already active on this tab
            }

            // Stop any existing session
            if (this.isActive) {
                await this.stopAssistant();
            }

            this.isActive = true;
            this.currentTabId = tabId;
            this.meetingData.startTime = new Date();
            
            // Clear previous meeting data
            this.resetMeetingData();

            // Inject and initialize content script
            await this.injectContentScript(tabId);

            // Update settings
            await this.updateSettings({ ...this.settings, isActive: true });

            // Notify popup of status change
            this.broadcastStatusUpdate('Active');

            console.log('Meeting Assistant started on tab:', tabId);
        } catch (error) {
            console.error('Error starting assistant:', error);
            throw error;
        }
    }

    /**
     * Stop the meeting assistant
     */
    async stopAssistant() {
        try {
            if (!this.isActive) {
                return;
            }

            // Generate final summary if enabled
            if (this.settings.summaryEnabled && this.meetingData.transcriptions.length > 0) {
                await this.generateMeetingSummary();
            }

            // Send stop signal to content script
            if (this.currentTabId) {
                try {
                    await chrome.tabs.sendMessage(this.currentTabId, { action: 'stopAssistant' });
                } catch (error) {
                    // Tab might have been closed
                    console.log('Could not send stop message to content script:', error.message);
                }
            }

            this.isActive = false;
            this.currentTabId = null;

            // Update settings
            await this.updateSettings({ ...this.settings, isActive: false });

            // Notify popup of status change
            this.broadcastStatusUpdate('Stopped');

            console.log('Meeting Assistant stopped');
        } catch (error) {
            console.error('Error stopping assistant:', error);
        }
    }

    /**
     * Inject content script into the meeting tab
     */
    async injectContentScript(tabId) {
        try {
            // Content script should already be injected via manifest
            // Send initialization message
            await chrome.tabs.sendMessage(tabId, {
                action: 'initAssistant',
                settings: this.settings
            });
        } catch (error) {
            console.error('Error injecting content script:', error);
            // Content script might not be ready yet, that's okay
        }
    }

    /**
     * Process transcription data from content script
     */
    async processTranscription(data, tabId) {
        if (!this.isActive || tabId !== this.currentTabId) {
            return;
        }

        try {
            // Store transcription
            this.meetingData.transcriptions.push({
                timestamp: new Date(),
                text: data.text,
                confidence: data.confidence || 0,
                language: data.language || 'unknown',
                speaker: data.speaker || 'unknown'
            });

            // Process translation if enabled
            if (this.settings.translationEnabled && data.text) {
                const translation = await this.translateText(data.text);
                if (translation) {
                    this.meetingData.translations.push({
                        timestamp: new Date(),
                        original: data.text,
                        translated: translation,
                        targetLanguage: this.settings.outputLanguage
                    });

                    // Send translation back to content script
                    await chrome.tabs.sendMessage(tabId, {
                        action: 'translationResult',
                        translation: translation,
                        originalText: data.text
                    });
                }
            }

            // Process task detection if enabled
            if (this.settings.taskDetectionEnabled && data.text) {
                const tasks = await this.detectTasks(data.text);
                if (tasks.length > 0) {
                    this.meetingData.tasks.push(...tasks);
                    
                    // Send task assignments if auto-email is enabled
                    if (this.settings.autoEmailEnabled && this.settings.userEmail) {
                        await this.sendTaskEmails(tasks);
                    }
                }
            }

            // Update status
            this.broadcastStatusUpdate('Processing');

        } catch (error) {
            console.error('Error processing transcription:', error);
        }
    }

    /**
     * Translate text using free translation service
     */
    async translateText(text) {
        try {
            if (!text || text.trim().length === 0) {
                return null;
            }

            // Using a simple translation approach with built-in browser APIs
            // In a production environment, you might want to use a more sophisticated solution
            
            // For now, we'll use a mock translation that reverses the process
            // In reality, you'd integrate with a free translation API like Google Translate (free tier)
            // or use a local translation library
            
            const targetLang = this.settings.outputLanguage || 'en';
            
            // Mock translation - in reality, implement proper translation
            if (targetLang === 'en') {
                return text; // Already in English
            }
            
            // For demonstration, we'll add a prefix to indicate translation
            return `[Translated to ${targetLang}] ${text}`;
            
        } catch (error) {
            console.error('Translation error:', error);
            return null;
        }
    }

    /**
     * Detect tasks and action items in text
     */
    async detectTasks(text) {
        const tasks = [];
        
        try {
            // Simple task detection using keywords and patterns
            const taskPatterns = [
                /(?:need to|have to|must|should|will|going to)\s+([^.!?]+)/gi,
                /(?:action item|task|todo|assignment):\s*([^.!?]+)/gi,
                /(?:@\w+|assign|assigned to)\s+([^.!?]+)/gi,
                /(?:by|due|deadline|before)\s+([^.!?]+)/gi
            ];

            const assigneePatterns = [
                /@(\w+)/g,
                /(?:assign|assigned to|responsibility of)\s+(\w+)/gi
            ];

            const priorityPatterns = [
                /(?:urgent|high priority|asap|immediately)/gi,
                /(?:low priority|when possible|eventually)/gi
            ];

            let matches = [];
            taskPatterns.forEach(pattern => {
                const found = [...text.matchAll(pattern)];
                matches.push(...found);
            });

            matches.forEach(match => {
                const taskText = match[1]?.trim();
                if (taskText && taskText.length > 3) {
                    const task = {
                        id: this.generateTaskId(),
                        description: taskText,
                        timestamp: new Date(),
                        source: 'transcription',
                        assignee: this.extractAssignee(text, match.index),
                        priority: this.extractPriority(text),
                        status: 'pending'
                    };

                    tasks.push(task);
                }
            });

        } catch (error) {
            console.error('Task detection error:', error);
        }

        return tasks;
    }

    /**
     * Extract assignee from context
     */
    extractAssignee(text, position) {
        const assigneePattern = /@(\w+)|(?:assign|assigned to|responsibility of)\s+(\w+)/gi;
        const matches = [...text.matchAll(assigneePattern)];
        
        for (const match of matches) {
            if (Math.abs(match.index - position) < 100) { // Within 100 characters
                return match[1] || match[2];
            }
        }
        
        return null;
    }

    /**
     * Extract priority from context
     */
    extractPriority(text) {
        if (/urgent|high priority|asap|immediately/gi.test(text)) {
            return 'high';
        } else if (/low priority|when possible|eventually/gi.test(text)) {
            return 'low';
        }
        return 'normal';
    }

    /**
     * Generate unique task ID
     */
    generateTaskId() {
        return 'task_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Send task assignment emails
     */
    async sendTaskEmails(tasks) {
        try {
            for (const task of tasks) {
                if (task.assignee && this.settings.userEmail) {
                    const subject = encodeURIComponent(`Task Assignment: ${task.description.substring(0, 50)}...`);
                    const body = encodeURIComponent(`
Hi ${task.assignee},

You have been assigned a new task from the meeting:

Task: ${task.description}
Priority: ${task.priority}
Date: ${task.timestamp.toLocaleDateString()}
Time: ${task.timestamp.toLocaleTimeString()}

Please confirm receipt of this task.

Best regards,
Meeting Assistant
                    `);

                    const mailtoLink = `mailto:${task.assignee}@company.com?subject=${subject}&body=${body}`;
                    
                    // Create a new tab with the mailto link
                    await chrome.tabs.create({ url: mailtoLink, active: false });
                }
            }
        } catch (error) {
            console.error('Error sending task emails:', error);
        }
    }

    /**
     * Generate meeting summary
     */
    async generateMeetingSummary() {
        try {
            if (!this.settings.summaryEnabled || this.meetingData.transcriptions.length === 0) {
                return '';
            }

            const startTime = this.meetingData.startTime || new Date();
            const endTime = new Date();
            const duration = Math.round((endTime - startTime) / 1000 / 60); // minutes

            let summary = `MEETING SUMMARY\n`;
            summary += `=================\n\n`;
            summary += `Date: ${startTime.toLocaleDateString()}\n`;
            summary += `Start Time: ${startTime.toLocaleTimeString()}\n`;
            summary += `End Time: ${endTime.toLocaleTimeString()}\n`;
            summary += `Duration: ${duration} minutes\n\n`;

            // Key points from transcriptions
            summary += `KEY DISCUSSION POINTS:\n`;
            summary += `------------------------\n`;
            
            const keyPoints = this.extractKeyPoints();
            keyPoints.forEach((point, index) => {
                summary += `${index + 1}. ${point}\n`;
            });

            // Tasks
            if (this.meetingData.tasks.length > 0) {
                summary += `\nACTION ITEMS:\n`;
                summary += `--------------\n`;
                this.meetingData.tasks.forEach((task, index) => {
                    summary += `${index + 1}. ${task.description}\n`;
                    if (task.assignee) {
                        summary += `   Assigned to: ${task.assignee}\n`;
                    }
                    summary += `   Priority: ${task.priority}\n\n`;
                });
            }

            // Full transcription
            if (this.meetingData.transcriptions.length > 0) {
                summary += `\nFULL TRANSCRIPTION:\n`;
                summary += `-------------------\n`;
                this.meetingData.transcriptions.forEach(transcript => {
                    const time = transcript.timestamp.toLocaleTimeString();
                    summary += `[${time}] ${transcript.text}\n`;
                });
            }

            this.meetingData.summary = summary;
            return summary;

        } catch (error) {
            console.error('Error generating summary:', error);
            return 'Error generating meeting summary.';
        }
    }

    /**
     * Extract key points from transcriptions
     */
    extractKeyPoints() {
        const keyPoints = [];
        
        try {
            // Simple keyword-based extraction
            const importantKeywords = [
                'decision', 'decided', 'conclude', 'important', 'key', 'main',
                'problem', 'issue', 'solution', 'agree', 'disagree', 'vote',
                'budget', 'cost', 'deadline', 'milestone', 'goal', 'objective'
            ];

            const sentences = this.meetingData.transcriptions
                .map(t => t.text)
                .join(' ')
                .split(/[.!?]+/)
                .filter(s => s.trim().length > 10);

            sentences.forEach(sentence => {
                const lowerSentence = sentence.toLowerCase();
                const hasImportantKeyword = importantKeywords.some(keyword => 
                    lowerSentence.includes(keyword)
                );

                if (hasImportantKeyword && keyPoints.length < 10) {
                    keyPoints.push(sentence.trim());
                }
            });

        } catch (error) {
            console.error('Error extracting key points:', error);
        }

        return keyPoints.slice(0, 5); // Return top 5 key points
    }

    /**
     * Clear meeting history and data
     */
    async clearMeetingHistory() {
        this.resetMeetingData();
        await chrome.storage.local.clear();
        console.log('Meeting history cleared');
    }

    /**
     * Reset meeting data
     */
    resetMeetingData() {
        this.meetingData = {
            transcriptions: [],
            translations: [],
            tasks: [],
            summary: '',
            startTime: null
        };
    }

    /**
     * Update settings
     */
    async updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        await chrome.storage.sync.set(this.settings);
        
        // Notify content script of settings change
        if (this.currentTabId) {
            try {
                await chrome.tabs.sendMessage(this.currentTabId, {
                    action: 'settingsUpdated',
                    settings: this.settings
                });
            } catch (error) {
                // Content script might not be ready
            }
        }
    }

    /**
     * Broadcast status update to popup
     */
    broadcastStatusUpdate(status) {
        try {
            chrome.runtime.sendMessage({
                action: 'statusUpdate',
                status: status,
                isActive: this.isActive
            });
        } catch (error) {
            // Popup might not be open
        }
    }

    /**
     * Handle tab updates
     */
    handleTabUpdate(tabId, changeInfo, tab) {
        // If the current meeting tab is navigated away from a meeting platform, stop the assistant
        if (tabId === this.currentTabId && changeInfo.url) {
            const isMeetingPlatform = this.isSupportedPlatform(changeInfo.url);
            if (!isMeetingPlatform && this.isActive) {
                console.log('Navigated away from meeting platform, stopping assistant');
                this.stopAssistant();
            }
        }
    }

    /**
     * Handle window focus changes
     */
    handleWindowFocusChange(windowId) {
        // Could be used to pause/resume assistant when window loses focus
        // For now, we'll keep it running
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
}

// Initialize the background service
const meetingAssistant = new MeetingAssistantBackground();