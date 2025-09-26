# Quick Start Guide

## Immediate Testing (5 minutes)

### 1. Install the Extension
1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right toggle)
3. Click "Load unpacked" and select this folder
4. Pin the extension to your toolbar

### 2. Test Basic Functionality
1. Go to [Google Meet](https://meet.google.com/new) or [Zoom](https://zoom.us/test)
2. Click the extension icon (🎙️)
3. Click "Start Assistant"
4. Allow microphone access
5. Speak and watch the overlay for transcription

### 3. Key Features to Test

**Live Transcription:**
- Speak clearly and see real-time text appear
- Try different languages if configured

**Translation:**
- Enable translation in the popup settings
- Select a target language different from your speech
- Speak and see both original and translated text

**Task Detection:**
- Say phrases like "We need to finish the report by Friday"
- Say "John should send the email to the client"
- Check the popup for detected tasks

**Meeting Summary:**
- After speaking for a few minutes
- Click "Export Summary" in the popup
- Review the generated meeting notes

## Troubleshooting First-Time Issues

### No Transcription Appearing?
1. Check microphone icon in address bar - allow access
2. Try speaking louder/clearer
3. Restart the extension

### Overlay Not Showing?
1. Make sure you're on a supported site (meet.google.com, zoom.us, etc.)
2. Refresh the page and try again
3. Check for conflicting extensions

### Performance Issues?
1. Disable features you don't need (translation, tasks)
2. Close other browser tabs
3. Use a newer version of Chrome

## Supported Test Sites

- **Google Meet**: https://meet.google.com/new (works best)
- **Zoom**: https://zoom.us/test (good for testing overlay)
- **Teams**: https://teams.microsoft.com (if you have access)

## Next Steps

Once basic functionality works:

1. **Customize Settings**: Adjust languages, features, email
2. **Test in Real Meetings**: Use during actual video calls
3. **Review Documentation**: Check README.md for advanced features
4. **Report Issues**: Note any bugs or feature requests

## Development Mode

For developers wanting to modify the extension:

1. Make changes to any file
2. Go to `chrome://extensions/`
3. Click the reload button on the extension card
4. Test changes immediately

The extension uses:
- **Web Speech API** for transcription
- **Free translation APIs** for multi-language support
- **Local storage** for settings and data
- **Content scripts** for meeting platform integration

---

**Time to get running**: ~5 minutes  
**Requirements**: Chrome browser, microphone  
**Best test platform**: Google Meet