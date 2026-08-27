import React, { useState, useRef, useEffect } from 'react';
import { Upload, Languages, Loader2, Captions, Music, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';
import { ID3Writer } from 'browser-id3-writer';

const LANGUAGES = [
  { label: 'Tamil (தமிழ்)', value: 'ta' },
  { label: 'English', value: 'en' },
  { label: 'Hindi (हिन्दी)', value: 'hi' },
  { label: 'Kannada (ಕನ್ನಡ)', value: 'kn' },
  { label: 'Malayalam (മലയാളം)', value: 'ml' },
  { label: 'Telugu (తెలుగు)', value: 'te' },
];

export const BrowserTranscriber: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState('ta');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [transcript, setTranscript] = useState('');
  const [chunks, setChunks] = useState<any[]>([]);
  
  const worker = useRef<Worker | null>(null);

  useEffect(() => {
    if (!worker.current) {
      worker.current = new Worker(new URL('../worker.ts', import.meta.url), {
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
        setStatus('Transcription complete!');
      } else if (status === 'error') {
        setIsProcessing(false);
        setStatus(`Error: ${message}`);
      }
    };

    worker.current.addEventListener('message', onMessage);
    return () => {
      worker.current?.removeEventListener('message', onMessage);
    };
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setTranscript('');
      setChunks([]);
      setStatus('File loaded. Ready to transcribe.');
    }
  };

  const handleTranscribe = async () => {
    if (!file || !worker.current) return;
    setIsProcessing(true);
    setTranscript('');
    setChunks([]);
    setStatus('Decoding audio...');

    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });

      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      let audio: Float32Array;

      if (audioBuffer.numberOfChannels === 2) {
        const SCALING_FACTOR = Math.SQRT1_2;
        const left = audioBuffer.getChannelData(0);
        const right = audioBuffer.getChannelData(1);
        audio = new Float32Array(left.length);
        for (let i = 0; i < audioBuffer.length; ++i) {
          audio[i] = SCALING_FACTOR * (left[i] + right[i]);
        }
      } else {
        audio = audioBuffer.getChannelData(0);
      }

      setStatus('Running in-browser Whisper WASM model...');
      worker.current.postMessage({
        audio,
        model: 'Xenova/whisper-tiny',
        language,
        subtask: 'transcribe',
      });
    } catch (err: any) {
      setIsProcessing(false);
      setStatus(`Failed to process audio: ${err.message}`);
    }
  };

  const downloadSRT = () => {
    if (!chunks || chunks.length === 0) return;
    let srt = '';
    const pad = (n: number) => n.toString().padStart(2, '0');
    const formatTime = (sec: number) => {
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = Math.floor(sec % 60);
      const ms = Math.floor((sec % 1) * 1000);
      return `${pad(h)}:${pad(m)}:${pad(s)},${ms.toString().padStart(3, '0')}`;
    };

    chunks.forEach((chunk, i) => {
      const [start, end] = chunk.timestamp;
      srt += `${i + 1}\n${formatTime(start)} --> ${formatTime(end || start + 2)}\n${chunk.text.trim()}\n\n`;
    });

    const blob = new Blob([srt], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file?.name.replace(/\.[^/.]+$/, '') || 'transcript'}.srt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadEmbeddedMP3 = async () => {
    if (!file || !transcript) return;
    try {
      const arrayBuffer = await file.arrayBuffer();
      const writer = new ID3Writer(arrayBuffer);
      writer.setFrame('USLT', {
        description: 'Transcribed by VaniScript AI',
        lyrics: transcript,
        language: 'tam',
      });
      writer.addTag();

      const blob = writer.getBlob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `transcribed_${file.name}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      alert(`Could not embed lyrics: ${e.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="professional-card">
        <div className="card-header-traditional">
          <div>
            <h2 className="card-title">
              <Sparkles className="text-indigo-400" size={20} />
              <span>Fast In-Browser Whisper Transcriber (Client WASM)</span>
            </h2>
            <p className="card-subtitle">
              Run Whisper directly inside your web browser with zero server uploads. Transcribe audio locally on your CPU.
            </p>
          </div>
        </div>

        {/* Upload Box */}
        <div className="p-6 rounded-xl border-2 border-dashed border-gray-700 bg-gray-900/50 text-center mb-6">
          <input
            type="file"
            id="browser-file-input"
            accept="audio/*"
            className="hidden"
            onChange={handleFileChange}
            disabled={isProcessing}
          />
          <label htmlFor="browser-file-input" className="cursor-pointer block">
            <Upload size={32} className="mx-auto text-indigo-400 mb-2" />
            <p className="font-semibold text-sm text-gray-200">
              {file ? file.name : 'Click to browse or drop an audio file'}
            </p>
            <p className="text-xs text-gray-400 mt-1">MP3, WAV, M4A, OGG supported</p>
          </label>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end mb-6">
          <div className="form-group">
            <label className="form-label flex items-center gap-1.5">
              <Languages size={14} className="text-indigo-400" />
              <span>Spoken Language</span>
            </label>
            <select
              className="form-select"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={isProcessing}
            >
              {LANGUAGES.map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>

          <button
            className="btn-primary flex items-center justify-center gap-2 h-[42px]"
            onClick={handleTranscribe}
            disabled={!file || isProcessing}
          >
            {isProcessing ? <Loader2 className="animate-spin" size={16} /> : <Sparkles size={16} />}
            <span>{isProcessing ? 'Processing in Browser...' : 'Start Local Transcription'}</span>
          </button>
        </div>

        {/* Status bar */}
        {(isProcessing || status) && (
          <div className="p-3 rounded-lg bg-gray-900 border border-gray-800 mb-6">
            {isProcessing && (
              <div className="progress-bar-bg mb-2">
                <motion.div
                  className="progress-bar-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                />
              </div>
            )}
            <div className="status-text">
              <span className="text-indigo-300">{status}</span>
              {progress > 0 && <span>{Math.round(progress)}%</span>}
            </div>
          </div>
        )}

        {/* Output Section */}
        {transcript && (
          <div className="space-y-4 pt-4 border-t border-white/10">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-bold text-gray-200">Transcript Result</h3>
              <div className="flex gap-2">
                <button className="btn-secondary text-xs flex items-center gap-1.5" onClick={downloadSRT} disabled={chunks.length === 0}>
                  <Captions size={14} />
                  <span>Download SRT</span>
                </button>
                <button className="btn-secondary text-xs flex items-center gap-1.5" onClick={downloadEmbeddedMP3}>
                  <Music size={14} />
                  <span>Embed Lyrics into MP3</span>
                </button>
              </div>
            </div>

            <textarea
              className="form-input text-xs font-sans h-44 leading-relaxed"
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
            />
          </div>
        )}
      </div>
    </div>
  );
};
