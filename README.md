# VaniScript AI - Offline Transcription

A high-performance, private audio/video transcription web app.

## Fixed Issues
- **Language Codes**: Updated to ISO 639-1 (e.g., `ta` for Tamil) as required by Whisper AI.
- **Error Handling**: Improved UI feedback for model loading and processing errors.
- **Real-time Updates**: Fixed text appending logic in the UI.

## Supported Languages
- Tamil (`ta`)
- English (`en`)
- Hindi (`hi`)
- Kannada (`kn`)
- Malayalam (`ml`)

## Getting Started
1. `npm install`
2. `npm run dev`
3. Upload an audio/video file.
4. Select **Tamil (தமிழ்)** or any other language.
5. Click **Start Transcription**.

*Note: The model (~150MB) downloads once and is stored locally in your browser cache.*
