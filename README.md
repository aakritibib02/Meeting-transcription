# Meeting Transcription & Translation Assistant

A powerful Chrome extension that provides real-time transcription, translation, and task detection for online meetings across multiple platforms including Zoom, Google Meet, Microsoft Teams, and more.

## 🌟 Features

- **Live Transcription**: Real-time speech-to-text conversion using Web Speech API
- **Multi-language Translation**: Translate conversations in real-time to your preferred language
- **Smart Task Detection**: Automatically identify and extract action items and assignments
- **Meeting Summaries**: Generate comprehensive meeting reports with key points and decisions
- **Email Integration**: Automatic task assignment emails via mailto links
- **Multi-platform Support**: Works on Zoom, Google Meet, Microsoft Teams, WebEx, and more
- **Privacy-focused**: All processing happens locally - no data sent to external servers
- **Lightweight**: CPU-friendly design with minimal performance impact

## 🛠 Installation

### Method 1: Load as Unpacked Extension (Recommended for Development)

1. **Download or Clone** this repository to your local machine
2. **Open Chrome** and navigate to `chrome://extensions/`
3. **Enable Developer Mode** by toggling the switch in the top-right corner
4. **Click "Load unpacked"** and select the `meeting-transcription-extension` folder
5. **Pin the extension** to your toolbar for easy access

### Method 2: Build and Install

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd meeting-transcription-extension
   ```

2. The extension is ready to use - no build process required!

3. Follow Method 1 steps 2-5 above.

## 🚀 Quick Start

1. **Install the extension** using the instructions above
2. **Open a meeting** in any supported platform (Zoom, Google Meet, Teams, etc.)
3. **Click the extension icon** in the toolbar to open the settings panel
4. **Configure your preferences**:
   - Select input language (or use auto-detect)
   - Choose translation target language
   - Enable/disable features as needed
   - Add your email for task assignments
5. **Click "Start Assistant"** to begin transcription
6. **Grant microphone permission** when prompted
7. **Watch the overlay** for real-time transcription and translation

## ⚙️ Configuration

### Language Settings

- **Input Language**: Select the primary language of the meeting or use auto-detect
- **Translation Language**: Choose your preferred language for translations

### Features

- **Live Transcription**: Enable/disable real-time speech-to-text
- **Real-time Translation**: Toggle automatic translation of spoken content
- **Task Detection**: Automatically identify action items and assignments
- **Meeting Summary**: Generate comprehensive meeting reports

### Email Settings

- **Your Email**: Required for receiving task assignments and summaries
- **Auto-send Task Assignments**: Automatically open mailto links for detected tasks

## 🎯 Usage Guide

### Starting a Meeting Session

1. Join your meeting on any supported platform
2. Click the Meeting Assistant extension icon
3. Verify your settings are correct
4. Click "Start Assistant"
5. Allow microphone access when prompted
6. The overlay will appear showing live transcription

### Using the Overlay

The overlay provides real-time information and can be:
- **Moved**: Drag by the header to reposition
- **Minimized**: Click the "−" button to collapse
- **Closed**: Click the "×" button to stop the session

### Viewing Results

- **Live Transcription**: View real-time speech-to-text in the overlay
- **Translations**: See translated text below original transcription
- **Tasks**: Detected action items appear in the tasks section
- **Summary**: Export comprehensive meeting summary via the popup

## 🔧 Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Google Meet | ✅ Fully Supported | Optimal positioning |
| Zoom | ✅ Fully Supported | Positioned above controls |
| Microsoft Teams | ✅ Fully Supported | Below top navigation |
| WebEx | ✅ Supported | Standard positioning |
| GoToMeeting | ✅ Supported | Standard positioning |
| Skype | ✅ Supported | Basic support |

## 🛡️ Privacy & Security

- **Local Processing**: All speech recognition and processing happens locally
- **No Data Collection**: No personal data is sent to external servers
- **Microphone Access**: Only used for live transcription during active sessions
- **Storage**: Settings and meeting data stored locally in Chrome

## 🔍 Troubleshooting

### Common Issues

**"Microphone permission denied"**
- Solution: Click the microphone icon in the address bar and allow access
- Alternative: Go to Chrome Settings > Privacy and Security > Site Settings > Microphone

**"Speech recognition not supported"**
- Solution: Ensure you're using Chrome (Speech API not available in all browsers)
- Workaround: Update to the latest Chrome version

**"No transcription appearing"**
- Check: Microphone is working and not muted
- Check: Extension has microphone permission
- Check: You're speaking clearly and audibly
- Try: Restart the assistant and check settings

**"Translation not working"**
- Check: Translation is enabled in settings
- Check: Internet connection for translation services
- Try: Different target language or restart session

**"Overlay not appearing"**
- Check: You're on a supported meeting platform
- Try: Refresh the page and restart the assistant
- Check: No other extensions are conflicting

### Performance Issues

**High CPU usage**
- Disable unused features (translation, task detection) if not needed
- Close other tabs/applications during meetings
- Restart Chrome if memory usage is high

**Delayed transcription**
- Check internet connection speed
- Reduce background applications
- Ensure Chrome is up to date

## 🎨 Customization

### Modifying Languages

Add new languages by editing `translation-worker.js`:

```javascript
// Add to language mappings
'your-lang': {
    'hello': 'your-translation',
    // ... more translations
}
```

### Styling the Overlay

Modify `overlay.css` to customize appearance:

```css
#meeting-assistant-overlay {
    /* Customize position, colors, size, etc. */
}
```

### Task Detection Patterns

Enhance task detection in `background.js`:

```javascript
const taskPatterns = [
    /your-custom-pattern/gi,
    // ... existing patterns
];
```

## 🔧 Development

### Project Structure

```
meeting-transcription-extension/
├── manifest.json          # Extension configuration
├── popup.html             # Extension popup interface
├── popup.css              # Popup styling
├── popup.js               # Popup functionality
├── background.js          # Service worker
├── content-script.js      # Injected into meeting pages
├── overlay.css            # Overlay interface styling
├── translation-worker.js  # Translation functionality
└── README.md              # This file
```

### Key Components

1. **Manifest**: Defines permissions and content script injection
2. **Background Script**: Coordinates between popup and content scripts
3. **Content Script**: Handles speech recognition and UI overlay
4. **Popup**: User interface for settings and controls
5. **Translation Worker**: Handles multi-service translation

### Adding New Features

1. Update `manifest.json` if new permissions are needed
2. Modify relevant scripts (background, content, popup)
3. Update UI components as necessary
4. Test across different meeting platforms

## 📝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and test thoroughly
4. Commit changes: `git commit -am 'Add new feature'`
5. Push to branch: `git push origin feature/new-feature`
6. Submit a pull request

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues, questions, or feature requests:

1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed description
4. Include browser version, OS, and meeting platform

## 🔮 Roadmap

### Planned Features

- **Speaker Identification**: Distinguish between different speakers
- **Advanced Translation**: Support for more languages and better accuracy
- **Meeting Analytics**: Participation metrics and speaking time analysis
- **Calendar Integration**: Automatic meeting detection and scheduling
- **Custom Vocabulary**: Industry-specific terminology support
- **Export Formats**: PDF, DOCX, and other summary formats

### Known Limitations

- Requires Chrome browser for Speech Recognition API
- Translation accuracy depends on available free services
- Some meeting platforms may have UI conflicts
- Microphone permission required for functionality

## 🙏 Acknowledgments

- Web Speech API for speech recognition capabilities
- Free translation services (MyMemory, LibreTranslate)
- Chrome Extension APIs for platform integration
- Open source community for inspiration and feedback

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Minimum Chrome Version**: 88+