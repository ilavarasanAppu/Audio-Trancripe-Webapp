import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileAudio, Languages, Download, Check, AlertCircle, Loader2, Captions, Music } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ID3Writer from 'browser-id3-writer';

const LANGUAGES = [
  { label: 'English', value: 'en' },
  { label: 'Tamil (தமிழ்)', value: 'ta' },
  { label: 'Hindi (हिन्दी)', value: 'hi' },
  { label: 'Kannada (ಕನ್ನಡ)', value: 'kn' },
  { label: 'Malayalam (മലയാളം)', value: 'ml' },
];

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState('en');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [transcript, setTranscript] = useState('');
  const [chunks, setChunks] = useState<any[]>([]);
  const [isDone, setIsDone] = useState(false);
  
  const worker = useRef<Worker | null>(null);

  useEffect(() => {
    if (!worker.current) {
      worker.current = new Worker(new URL('./worker.ts', import.meta.url), {
        type: 'module',
      });
    }

    const onMessage = (e: MessageEvent) => {
      const { status, file, progress, message, output } = e.data;

      if (status === 'initiate') {
        setStatus(`Loading: ${file || 'model'}...`);
      } else if (status === 'progress') {
        if (typeof progress === 'number') {
          setProgress(progress);
          setStatus(`Downloading model... ${Math.round(progress)}%`);
        }
      } else if (status === 'done' || status === 'ready') {
        setStatus('Model loaded. Ready to transcribe.');
        setProgress(100);
      } else if (status === 'update') {
        const { text } = e.data;
        if (text) {
          setTranscript((prev) => prev + text);
        }
      } else if (status === 'complete') {
        if (output) {
          setTranscript(output.text);
          setChunks(output.chunks || []);
        }
        setIsProcessing(false);
        setIsDone(true);
        setStatus('Transcription complete!');
      } else if (status === 'error') {
        console.error('Worker Error:', message);
        setStatus(`Error: ${message || 'Unknown error occurred'}`);
        setIsProcessing(false);
      }
    };

    worker.current.addEventListener('message', onMessage);

    return () => {
      worker.current?.removeEventListener('message', onMessage);
    };
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
      setIsDone(false);
      setTranscript('');
    }
  };

  const processAudio = async (audioFile: File) => {
    setStatus('Preparing audio...');
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
    const arrayBuffer = await audioFile.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    
    // Convert to mono Float32Array at 16kHz
    const audioData = audioBuffer.getChannelData(0);
    return audioData;
  };

  const startTranscription = async () => {
    if (!file || !worker.current) return;

    try {
      setIsProcessing(true);
      setIsDone(false);
      setTranscript('');
      setStatus('Converting audio format...');
      
      const audioData = await processAudio(file);
      
      setStatus('Transcription in progress...');
      worker.current.postMessage({
        audio: audioData,
        language: language,
        task: 'transcribe',
      });
    } catch (err: any) {
      setStatus(`Error processing audio: ${err.message}`);
      setIsProcessing(false);
    }
  };

  const formatTime = (seconds: number) => {
    const pad = (n: number) => n.toString().padStart(2, '0');
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${pad(h)}:${pad(m)}:${pad(s)},${ms.toString().padStart(3, '0')}`;
  };

  const downloadSRT = () => {
    if (chunks.length === 0) return;
    
    let srtContent = '';
    chunks.forEach((chunk, i) => {
      const start = formatTime(chunk.timestamp[0] || 0);
      const end = formatTime(chunk.timestamp[1] || chunk.timestamp[0] + 2);
      srtContent += `${i + 1}\n${start} --> ${end}\n${chunk.text.trim()}\n\n`;
    });

    const blob = new Blob([srtContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file?.name.split('.')[0] || 'subtitle'}.srt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadTranscript = () => {
    const blob = new Blob([transcript], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file?.name.split('.')[0] || 'transcript'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadMP3WithLyrics = async () => {
    if (!file || !transcript) return;
    
    // Only support MP3 for tag embedding
    if (!file.name.toLowerCase().endsWith('.mp3')) {
      alert('Lyrics embedding is currently only supported for MP3 files.');
      return;
    }

    try {
      setStatus('Embedding lyrics into MP3...');
      const arrayBuffer = await file.arrayBuffer();
      const writer = new ID3Writer(arrayBuffer);
      
      // Add Unsynchronised lyrics (USLT)
      writer.setFrame('USLT', {
        description: 'Lyrics',
        lyrics: transcript,
        language: 'eng' // Use eng for broad compatibility
      });
      
      writer.addTag();
      
      const taggedBlob = writer.getBlob();
      const url = URL.createObjectURL(taggedBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${file.name.split('.')[0]}_with_lyrics.mp3`;
      a.click();
      URL.revokeObjectURL(url);
      setStatus('Lyrics embedded successfully!');
    } catch (err: any) {
      console.error('Error embedding lyrics:', err);
      setStatus(`Failed to embed lyrics: ${err.message}`);
    }
  };

  return (
    <div className="app">
      <div className="bg-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>

      <div className="app-container">
        <header>
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            VaniScript AI
          </motion.h1>
          <motion.p 
            className="subtitle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            Offline Audio & Video Transcription in Indian Languages. No cloud, no data leaks.
          </motion.p>
        </header>

        <main className="main-card">
          <div className="upload-section">
            <input 
              type="file" 
              id="file-upload" 
              accept="audio/*,video/*" 
              onChange={handleFileChange}
              hidden 
            />
            <label htmlFor="file-upload" className="upload-zone">
              {file ? (
                <>
                  <FileAudio className="upload-icon" />
                  <div>
                    <p style={{ fontWeight: 600 }}>{file.name}</p>
                    <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <Upload className="upload-icon" />
                  <div>
                    <p style={{ fontWeight: 600 }}>Click to upload or drag & drop</p>
                    <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                      Audio or Video files (MP3, WAV, MP4, etc.)
                    </p>
                  </div>
                </>
              )}
            </label>
          </div>

          <div className="controls">
            <div className="control-group">
              <label>Target Language</label>
              <select 
                value={language} 
                onChange={(e) => setLanguage(e.target.value)}
                disabled={isProcessing}
              >
                {LANGUAGES.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label>Task</label>
              <select disabled={isProcessing}>
                <option value="transcribe">Transcribe (Speech to Text)</option>
              </select>
            </div>

            <button 
              className="btn-primary" 
              onClick={startTranscription}
              disabled={!file || isProcessing}
            >
              {isProcessing ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                  <Loader2 className="animate-spin" size={20} />
                  <span>Processing...</span>
                </div>
              ) : (
                'Start Transcription'
              )}
            </button>
          </div>

          {status.includes('Error') && (
            <div className="error-box" style={{ marginTop: '1rem', color: '#ef4444', background: 'rgba(239, 68, 68, 0.1)', padding: '1rem', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlertCircle size={18} />
              <span>{status}</span>
            </div>
          )}

          {(isProcessing || status) && (
            <div className="progress-container">
              <div className="progress-bar-bg">
                <motion.div 
                  className="progress-bar-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                />
              </div>
              <div className="status-text">
                <span>{status}</span>
                <span>{Math.round(progress)}%</span>
              </div>
            </div>
          )}

          <AnimatePresence>
            {(transcript || isDone) && (
              <motion.div 
                className="transcript-card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="transcript-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Languages size={18} />
                    <span style={{ fontWeight: 600 }}>Transcript</span>
                    <span className="lang-badge">{language.toUpperCase()}</span>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    {file?.name.toLowerCase().endsWith('.mp3') && (
                      <button className="btn-icon" onClick={downloadMP3WithLyrics} title="Download MP3 with Embedded Lyrics">
                        <Music size={18} />
                      </button>
                    )}
                    <button className="btn-icon" onClick={downloadSRT} title="Download SRT Subtitles">
                      <Captions size={18} />
                    </button>
                    <button className="btn-icon" onClick={downloadTranscript} title="Download TXT">
                      <Download size={18} />
                    </button>
                  </div>
                </div>
                <div className="transcript-content">
                  {chunks.length > 0 ? (
                    <div className="chunks-list">
                      {chunks.map((chunk, i) => (
                        <div key={i} className="chunk-item">
                          <span className="chunk-time">
                            [{Math.floor(chunk.timestamp[0])}s]
                          </span>
                          <span className="chunk-text">{chunk.text}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    transcript || (isProcessing ? 'Waiting for results...' : '')
                  )}
                </div>
                {isDone && (
                  <div style={{ marginTop: '1rem', color: 'var(--success-color)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
                    <Check size={16} /> Completed Successfully
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        .btn-icon {
          background: none;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
          padding: 0.5rem;
          border-radius: 8px;
          transition: all 0.2s;
        }
        .btn-icon:hover {
          background: rgba(255, 255, 255, 0.1);
          color: white;
        }
      `}</style>
    </div>
  );
}

export default App;
