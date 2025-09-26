/**
 * Content Script - Injects into meeting platforms and handles speech recognition
 */

class MeetingAssistantContent {
    constructor() {
        this.isInitialized = false;
        this.isListening = false;
        this.recognition = null;
        this.overlay = null;
        this.settings = {};
        this.transcriptionBuffer = [];
        this.platform = this.detectPlatform();
        
        this.init();
    }

    async init() {
        // Wait for page to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            await this.setup();
        }
    }

    async setup() {
        try {
            console.log('Meeting Assistant Content Script initializing on', this.platform);
            
            this.setupMessageListeners();
            await this.loadSettings();
            this.setupSpeechRecognition();
            
            this.isInitialized = true;
            console.log('Meeting Assistant Content Script ready');
        } catch (error) {
            console.error('Error initializing content script:', error);
        }
    }

    /**
     * Detect which meeting platform we're on
     */
    detectPlatform() {
        const hostname = window.location.hostname.toLowerCase();
        
        if (hostname.includes('zoom.us')) {
            return 'zoom';
        } else if (hostname.includes('meet.google.com')) {
            return 'googlemeet';
        } else if (hostname.includes('teams.microsoft.com')) {
            return 'teams';
        } else if (hostname.includes('webex.com')) {
            return 'webex';
        } else if (hostname.includes('gotomeeting.com')) {
            return 'gotomeeting';
        } else if (hostname.includes('skype.com')) {
            return 'skype';
        } else {
            return 'unknown';
        }
    }

    /**
     * Setup message listeners for communication with background script
     */
    setupMessageListeners() {
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open
        });
    }

    /**
     * Handle messages from background script and popup
     */
    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.action) {
                case 'initAssistant':
                    await this.initializeAssistant(message.settings);
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

                case 'translationResult':
                    this.displayTranslation(message.translation, message.originalText);
                    sendResponse({ success: true });
                    break;

                case 'getStatus':
                    sendResponse({
                        success: true,
                        isListening: this.isListening,
                        platform: this.platform,
                        isInitialized: this.isInitialized
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
     * Load settings from storage
     */
    async loadSettings() {
        try {
            this.settings = await chrome.storage.sync.get();
        } catch (error) {
            console.error('Error loading settings:', error);
            this.settings = {
                inputLanguage: 'auto',
                outputLanguage: 'en',
                transcriptionEnabled: true,
                translationEnabled: true,
                taskDetectionEnabled: true,
                summaryEnabled: true
            };
        }
    }

    /**
     * Initialize the assistant with given settings
     */
    async initializeAssistant(settings) {
        if (settings) {
            this.settings = settings;
        }

        await this.createOverlay();
        await this.startListening();
    }

    /**
     * Stop the assistant
     */
    async stopAssistant() {
        await this.stopListening();
        this.removeOverlay();
    }

    /**
     * Update settings
     */
    async updateSettings(settings) {
        this.settings = { ...this.settings, ...settings };
        
        // Restart recognition if language changed
        if (this.isListening && this.recognition) {
            await this.stopListening();
            await this.startListening();
        }

        // Update overlay if it exists
        if (this.overlay) {
            this.updateOverlaySettings();
        }
    }

    /**
     * Setup speech recognition
     */
    setupSpeechRecognition() {
        try {
            // Check if Speech Recognition is available
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            
            if (!SpeechRecognition) {
                console.warn('Speech Recognition API not available in this browser');
                this.showNotification('Speech recognition not supported in this browser');
                return false;
            }

            this.recognition = new SpeechRecognition();
            
            // Configure recognition
            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            this.recognition.maxAlternatives = 1;
            
            // Set language
            const inputLang = this.settings.inputLanguage || 'auto';
            if (inputLang !== 'auto') {
                this.recognition.lang = inputLang;
            }

            // Event listeners
            this.recognition.onstart = () => {
                console.log('Speech recognition started');
                this.isListening = true;
                this.updateOverlayStatus('Listening...');
            };

            this.recognition.onresult = (event) => {
                this.handleSpeechResult(event);
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.handleSpeechError(event);
            };

            this.recognition.onend = () => {
                console.log('Speech recognition ended');
                this.isListening = false;
                this.updateOverlayStatus('Stopped');
                
                // Restart if we should still be listening
                if (this.settings.transcriptionEnabled && this.isInitialized) {
                    setTimeout(() => {
                        if (!this.isListening && this.recognition) {
                            this.startListening();
                        }
                    }, 1000);
                }
            };

            return true;
        } catch (error) {
            console.error('Error setting up speech recognition:', error);
            return false;
        }
    }

    /**
     * Start listening for speech
     */
    async startListening() {
        try {
            if (!this.settings.transcriptionEnabled || !this.recognition) {
                return;
            }

            if (this.isListening) {
                return; // Already listening
            }

            // Request microphone permission first
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                // Close the stream immediately, we just needed permission
                stream.getTracks().forEach(track => track.stop());
            } catch (error) {
                console.error('Microphone permission denied:', error);
                this.showNotification('Microphone permission required for transcription');
                return;
            }

            this.recognition.start();
            console.log('Started speech recognition');
        } catch (error) {
            console.error('Error starting speech recognition:', error);
            this.showNotification('Error starting speech recognition');
        }
    }

    /**
     * Stop listening for speech
     */
    async stopListening() {
        try {
            if (this.recognition && this.isListening) {
                this.recognition.stop();
                this.isListening = false;
                console.log('Stopped speech recognition');
            }
        } catch (error) {
            console.error('Error stopping speech recognition:', error);
        }
    }

    /**
     * Handle speech recognition results
     */
    handleSpeechResult(event) {
        try {
            let finalTranscript = '';
            let interimTranscript = '';

            // Process all results
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                const transcript = result[0].transcript;

                if (result.isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            // Update overlay with current transcript
            if (finalTranscript || interimTranscript) {
                this.displayTranscription(finalTranscript, interimTranscript);
            }

            // Send final transcript to background script for processing
            if (finalTranscript.trim().length > 0) {
                this.sendTranscriptionToBackground({
                    text: finalTranscript.trim(),
                    confidence: event.results[event.results.length - 1][0].confidence,
                    language: this.recognition.lang || 'auto',
                    timestamp: new Date().toISOString(),
                    platform: this.platform
                });
            }

        } catch (error) {
            console.error('Error handling speech result:', error);
        }
    }

    /**
     * Handle speech recognition errors
     */
    handleSpeechError(event) {
        const errorMessages = {
            'no-speech': 'No speech detected',
            'audio-capture': 'Audio capture failed',
            'not-allowed': 'Microphone permission denied',
            'network': 'Network error during recognition',
            'language-not-supported': 'Language not supported',
            'service-not-allowed': 'Speech service not allowed'
        };

        const message = errorMessages[event.error] || `Speech recognition error: ${event.error}`;
        console.error(message);
        
        // Show error in overlay
        this.updateOverlayStatus(`Error: ${event.error}`);

        // Try to recover from certain errors
        if (event.error === 'no-speech' || event.error === 'network') {
            // These are recoverable, just restart
            setTimeout(() => {
                if (!this.isListening && this.settings.transcriptionEnabled) {
                    this.startListening();
                }
            }, 2000);
        } else if (event.error === 'not-allowed') {
            this.showNotification('Microphone access denied. Please allow microphone access and try again.');
        }
    }

    /**
     * Send transcription data to background script
     */
    async sendTranscriptionToBackground(data) {
        try {
            await chrome.runtime.sendMessage({
                action: 'transcriptionData',
                data: data
            });
        } catch (error) {
            console.error('Error sending transcription to background:', error);
        }
    }

    /**
     * Create the overlay interface
     */
    async createOverlay() {
        if (this.overlay) {
            return; // Already created
        }

        try {
            // Create overlay container
            this.overlay = document.createElement('div');
            this.overlay.id = 'meeting-assistant-overlay';
            this.overlay.innerHTML = `
                <div class="ma-overlay-header">
                    <div class="ma-title">🎙️ Meeting Assistant</div>
                    <div class="ma-status" id="ma-status">Ready</div>
                    <div class="ma-controls">
                        <button id="ma-minimize" class="ma-btn ma-btn-small" title="Minimize">−</button>
                        <button id="ma-close" class="ma-btn ma-btn-small" title="Close">×</button>
                    </div>
                </div>
                <div class="ma-content" id="ma-content">
                    <div class="ma-section" id="ma-transcription">
                        <div class="ma-section-title">Live Transcription</div>
                        <div class="ma-transcript" id="ma-transcript">
                            <div class="ma-transcript-placeholder">Transcription will appear here...</div>
                        </div>
                    </div>
                    <div class="ma-section" id="ma-translation" style="display: none;">
                        <div class="ma-section-title">Translation</div>
                        <div class="ma-translated-text" id="ma-translated-text"></div>
                    </div>
                    <div class="ma-section" id="ma-tasks" style="display: none;">
                        <div class="ma-section-title">Detected Tasks</div>
                        <div class="ma-task-list" id="ma-task-list"></div>
                    </div>
                </div>
            `;

            // Position overlay appropriately for each platform
            this.positionOverlayForPlatform();

            // Add event listeners
            this.setupOverlayControls();

            // Insert into page
            document.body.appendChild(this.overlay);

            console.log('Overlay created successfully');
        } catch (error) {
            console.error('Error creating overlay:', error);
        }
    }

    /**
     * Position overlay based on platform
     */
    positionOverlayForPlatform() {
        const baseStyles = `
            position: fixed;
            z-index: 10000;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            width: 320px;
            max-height: 400px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            resize: both;
            overflow: auto;
        `;

        // Platform-specific positioning
        let position = 'top: 20px; right: 20px;'; // default

        switch (this.platform) {
            case 'googlemeet':
                position = 'top: 70px; right: 20px;'; // Below Google Meet's top bar
                break;
            case 'zoom':
                position = 'bottom: 100px; right: 20px;'; // Above Zoom's bottom controls
                break;
            case 'teams':
                position = 'top: 60px; right: 20px;'; // Below Teams' top bar
                break;
            case 'webex':
                position = 'top: 80px; right: 20px;';
                break;
        }

        this.overlay.style.cssText = baseStyles + position;
    }

    /**
     * Setup overlay control handlers
     */
    setupOverlayControls() {
        const minimizeBtn = this.overlay.querySelector('#ma-minimize');
        const closeBtn = this.overlay.querySelector('#ma-close');
        const content = this.overlay.querySelector('#ma-content');

        minimizeBtn.addEventListener('click', () => {
            const isMinimized = content.style.display === 'none';
            content.style.display = isMinimized ? 'block' : 'none';
            minimizeBtn.textContent = isMinimized ? '−' : '+';
            minimizeBtn.title = isMinimized ? 'Minimize' : 'Expand';
        });

        closeBtn.addEventListener('click', async () => {
            await chrome.runtime.sendMessage({ action: 'stopAssistant' });
        });

        // Make overlay draggable
        let isDragging = false;
        let currentX, currentY, initialX, initialY, xOffset = 0, yOffset = 0;

        const header = this.overlay.querySelector('.ma-overlay-header');
        header.addEventListener('mousedown', (e) => {
            if (e.target.tagName === 'BUTTON') return;
            
            isDragging = true;
            initialX = e.clientX - xOffset;
            initialY = e.clientY - yOffset;
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            e.preventDefault();
            currentX = e.clientX - initialX;
            currentY = e.clientY - initialY;

            xOffset = currentX;
            yOffset = currentY;

            this.overlay.style.transform = `translate3d(${currentX}px, ${currentY}px, 0)`;
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
    }

    /**
     * Remove overlay from page
     */
    removeOverlay() {
        if (this.overlay && this.overlay.parentNode) {
            this.overlay.parentNode.removeChild(this.overlay);
            this.overlay = null;
        }
    }

    /**
     * Update overlay status
     */
    updateOverlayStatus(status) {
        if (this.overlay) {
            const statusElement = this.overlay.querySelector('#ma-status');
            if (statusElement) {
                statusElement.textContent = status;
                
                // Add visual indicator
                statusElement.className = 'ma-status';
                if (status === 'Listening...') {
                    statusElement.classList.add('ma-status-active');
                } else if (status.includes('Error')) {
                    statusElement.classList.add('ma-status-error');
                }
            }
        }
    }

    /**
     * Display transcription in overlay
     */
    displayTranscription(finalText, interimText = '') {
        if (!this.overlay || !this.settings.transcriptionEnabled) return;

        const transcriptElement = this.overlay.querySelector('#ma-transcript');
        if (!transcriptElement) return;

        // Update transcript display
        let content = '';
        
        // Add recent final transcripts from buffer
        const recentTranscripts = this.transcriptionBuffer.slice(-3);
        recentTranscripts.forEach(transcript => {
            content += `<div class="ma-transcript-line ma-final">${transcript}</div>`;
        });

        // Add current final text
        if (finalText) {
            content += `<div class="ma-transcript-line ma-final ma-current">${finalText}</div>`;
            // Add to buffer
            this.transcriptionBuffer.push(finalText);
            if (this.transcriptionBuffer.length > 10) {
                this.transcriptionBuffer.shift(); // Keep only recent items
            }
        }

        // Add interim text
        if (interimText) {
            content += `<div class="ma-transcript-line ma-interim">${interimText}</div>`;
        }

        if (!content) {
            content = '<div class="ma-transcript-placeholder">Transcription will appear here...</div>';
        }

        transcriptElement.innerHTML = content;
        
        // Scroll to bottom
        transcriptElement.scrollTop = transcriptElement.scrollHeight;
    }

    /**
     * Display translation in overlay
     */
    displayTranslation(translatedText, originalText) {
        if (!this.overlay || !this.settings.translationEnabled) return;

        const translationSection = this.overlay.querySelector('#ma-translation');
        const translatedTextElement = this.overlay.querySelector('#ma-translated-text');
        
        if (!translationSection || !translatedTextElement) return;

        // Show translation section
        translationSection.style.display = 'block';
        
        // Add new translation
        const translationDiv = document.createElement('div');
        translationDiv.className = 'ma-translation-item';
        translationDiv.innerHTML = `
            <div class="ma-original-text">${originalText}</div>
            <div class="ma-arrow">↓</div>
            <div class="ma-translated">${translatedText}</div>
        `;

        translatedTextElement.insertBefore(translationDiv, translatedTextElement.firstChild);

        // Keep only recent translations
        const translations = translatedTextElement.querySelectorAll('.ma-translation-item');
        if (translations.length > 5) {
            translations[translations.length - 1].remove();
        }
    }

    /**
     * Update overlay settings visibility
     */
    updateOverlaySettings() {
        if (!this.overlay) return;

        const transcriptionSection = this.overlay.querySelector('#ma-transcription');
        const translationSection = this.overlay.querySelector('#ma-translation');
        const tasksSection = this.overlay.querySelector('#ma-tasks');

        if (transcriptionSection) {
            transcriptionSection.style.display = this.settings.transcriptionEnabled ? 'block' : 'none';
        }
        if (translationSection) {
            translationSection.style.display = this.settings.translationEnabled ? 'block' : 'none';
        }
        if (tasksSection) {
            tasksSection.style.display = this.settings.taskDetectionEnabled ? 'block' : 'none';
        }
    }

    /**
     * Show notification to user
     */
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `ma-notification ma-notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${type === 'error' ? '#ff4757' : '#1e90ff'};
            color: white;
            padding: 12px 24px;
            border-radius: 6px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 10001;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;

        document.body.appendChild(notification);

        // Fade in
        requestAnimationFrame(() => {
            notification.style.opacity = '1';
        });

        // Remove after delay
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }
}

// Initialize content script when page loads
if (typeof window !== 'undefined') {
    const meetingAssistant = new MeetingAssistantContent();
}