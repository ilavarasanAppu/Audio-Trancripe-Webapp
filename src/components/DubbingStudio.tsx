import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic2, Play, Pause, Volume2, Download, RefreshCw, Sparkles, 
  Languages, FileAudio, Loader2, Music, 
  Sliders, Captions, Folder, Radio, Monitor,
  CheckCircle2, StopCircle, Search, List
} from 'lucide-react';
import { motion } from 'framer-motion';
import { api, type AudioFile, type DubbingProject, type DubbingSegment, type LanguageVoices, type CustomModelItem } from '../services/api';

type DubbingSourceMode = 'file' | 'folder' | 'mic_live' | 'system_sound';

export const DubbingStudio: React.FC = () => {
  // 1. Source Mode: File, Folder Batch, Voice-to-Voice (Mic), System Audio Capture
  const [sourceMode, setSourceMode] = useState<DubbingSourceMode>('file');

  // 2. Global Dubbing Config
  const [files, setFiles] = useState<AudioFile[]>([]);
  const [selectedFileId, setSelectedFileId] = useState<number | null>(null);
  const [sourceLang, setSourceLang] = useState('auto');
  const [targetLang, setTargetLang] = useState('ta');
  const [asrEngine, setAsrEngine] = useState<string>('faster_whisper');
  const [ttsEngine, setTtsEngine] = useState<string>('edge_neural');
  const [availableVoices, setAvailableVoices] = useState<LanguageVoices>({});
  const [selectedVoice, setSelectedVoice] = useState<string>('ta-IN-ValluvarNeural');
  const [customModels, setCustomModels] = useState<CustomModelItem[]>([]);
  const [selectedCustomModel, setSelectedCustomModel] = useState<string>('');
  
  // 3. Folder Batch Mode State
  const [batchFolderDir, setBatchFolderDir] = useState('');
  const [batchOutputDir, setBatchOutputDir] = useState('');
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const [batchResult, setBatchResult] = useState<any | null>(null);

  // 4. Live Streaming Mode State (Mic & Desktop Audio)
  const [isLiveActive, setIsLiveActive] = useState(false);
  const [liveLogs, setLiveLogs] = useState<Array<{ id: string; time: string; orig: string; trans: string; audioUrl?: string }>>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const liveAudioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const liveIntervalRef = useRef<any>(null);

  // 5. Active Project & Progress State
  const [currentProject, setCurrentProject] = useState<DubbingProject | null>(null);
  const [segments, setSegments] = useState<DubbingSegment[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState('');
  const [progress, setProgress] = useState(0);

  // 6. Dual-Track Player State
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [originalVolume, setOriginalVolume] = useState(0.25);
  const [dubbedVolume, setDubbedVolume] = useState(1.0);
  const [activeTrack, setActiveTrack] = useState<'dubbed' | 'original' | 'both'>('dubbed');

  // Transcript Search State
  const [transcriptSearchQuery, setTranscriptSearchQuery] = useState('');
  const [transcriptCurrentTime, setTranscriptCurrentTime] = useState(0);

  const originalAudioRef = useRef<HTMLAudioElement | null>(null);
  const dubbedAudioRef = useRef<HTMLAudioElement | null>(null);

  // Filter segments based on search query
  const filteredSegments = segments.filter(seg => {
    if (!transcriptSearchQuery.trim()) return true;
    const query = transcriptSearchQuery.toLowerCase();
    return (
      (seg.original_text || '').toLowerCase().includes(query) ||
      (seg.translated_text || '').toLowerCase().includes(query)
    );
  });

  // Highlight search text in transcript
  const highlightSearchText = (text: string, query: string) => {
    if (!query.trim()) return <span>{text}</span>;
    
    const parts = text.split(new RegExp(`(${query})`, 'gi'));
    return (
      <span>
        {parts.map((part, i) => 
          part.toLowerCase() === query.toLowerCase() 
            ? <mark key={i} className="bg-amber-500/30 text-amber-200 px-1 rounded">{part}</mark>
            : <span key={i}>{part}</span>
        )}
      </span>
    );
  };

  useEffect(() => {
    loadFiles();
    loadVoices();
    loadCustomModels();
    return () => {
      stopLiveCapture();
    };
  }, []);

  const loadCustomModels = async () => {
    try {
      const data = await api.getCustomModels();
      if (data && data.models) {
        setCustomModels(data.models);
        if (data.selected_custom_model) {
          setSelectedCustomModel(data.selected_custom_model);
        }
      }
    } catch (err) {
      console.error('Error loading custom models:', err);
    }
  };

  const loadFiles = async () => {
    try {
      const data = await api.getFiles();
      setFiles(data.files || []);
      if (data.files && data.files.length > 0 && !selectedFileId) {
        const indexed = data.files.find(f => f.status === 'indexed') || data.files[0];
        setSelectedFileId(indexed.id);
      }
    } catch (err) {
      console.error('Error loading library files:', err);
    }
  };

  const loadVoices = async () => {
    try {
      const data = await api.getVoices();
      if (data && data.languages) {
        setAvailableVoices(data.languages);
        if (data.languages[targetLang]) {
          setSelectedVoice(data.languages[targetLang].default_voice);
        }
      }
    } catch (err) {
      console.error('Error loading voice models:', err);
    }
  };

  useEffect(() => {
    if (availableVoices[targetLang]) {
      setSelectedVoice(availableVoices[targetLang].default_voice);
    }
  }, [targetLang, availableVoices]);

  // Upload handler for single file
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const file = e.target.files[0];
    setIsProcessing(true);
    setProcessStatus('Uploading media file to library...');
    try {
      const uploaded = await api.uploadFile(file);
      await loadFiles();
      setSelectedFileId(uploaded.id);
      setProcessStatus('File ready. Click "Start Dubbing Pipeline" to begin.');
    } catch (err: any) {
      setProcessStatus(`Upload error: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Pipeline execution: Transcribe -> Pivot Translation -> Same-Voice Synthesis
  const handleStartDubbing = async () => {
    if (!selectedFileId) return;
    setIsProcessing(true);
    setProgress(15);
    setProcessStatus('Step 1/3: Transcribing Audio with Timestamp Alignment...');

    try {
      const selectedFile = files.find(f => f.id === selectedFileId);
      const isCustomAsr = asrEngine.startsWith('custom:');
      const realAsrEngine = isCustomAsr ? 'faster_whisper' : asrEngine;
      const customPath = isCustomAsr ? asrEngine.replace('custom:', '') : (selectedCustomModel || undefined);

      if (!selectedFile || selectedFile.status !== 'indexed') {
        setProgress(30);
        await api.ingestFile(selectedFileId, 'base', sourceLang, realAsrEngine, customPath);
        await loadFiles();
      }

      setProgress(50);
      setProcessStatus('Step 2/3: Initializing Dubbing Project & Translating...');
      
      const created = await api.createDubbingProject({
        file_id: selectedFileId,
        source_lang: sourceLang,
        target_lang: targetLang,
        engine: ttsEngine === 'sarvam_bulbul' ? 'sarvam' : 'neural',
        tts_engine: ttsEngine,
        asr_engine: realAsrEngine,
        custom_model_path: customPath,
        voice_name: selectedVoice
      });

      setProgress(75);
      setProcessStatus(`Step 3/3: Synthesizing ${targetLang.toUpperCase()} with Same Pitch & Timing Sync...`);

      await api.processDubbing(created.project_id, undefined, ttsEngine);
      
      setProgress(100);
      setProcessStatus('Dubbing Complete! Master track ready for playback.');
      
      const projData = await api.getDubbingProject(created.project_id);
      setCurrentProject(projData.project);
      setSegments(projData.segments || []);
    } catch (err: any) {
      setProcessStatus(`Dubbing Error: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Batch folder execution
  const handleStartBatchFolder = async () => {
    if (!batchFolderDir.trim()) return;
    setIsBatchProcessing(true);
    setBatchResult(null);
    try {
      const res = await api.batchFolderDubbing({
        source_directory: batchFolderDir.trim(),
        target_lang: targetLang,
        source_lang: sourceLang,
        tts_engine: ttsEngine,
        asr_engine: asrEngine,
        output_directory: batchOutputDir.trim() || undefined
      });
      setBatchResult(res);
    } catch (err: any) {
      alert(`Batch folder dubbing failed: ${err.message}`);
    } finally {
      setIsBatchProcessing(false);
    }
  };

  // Live streaming microphone or desktop sound capture
  const startLiveCapture = async (mode: 'mic_live' | 'system_sound') => {
    try {
      let stream: MediaStream;
      if (mode === 'system_sound') {
        stream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
        });
      } else {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true }
        });
      }

      setIsLiveActive(true);
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = recorder;

      let recordedChunks: Blob[] = [];

      recorder.ondataavailable = async (e) => {
        if (e.data.size > 0) {
          recordedChunks.push(e.data);
          const chunkBlob = new Blob(recordedChunks, { type: 'audio/webm' });
          recordedChunks = [];

          try {
            const dubRes = await api.dubLiveChunk({
              audio_blob: chunkBlob,
              source_lang: sourceLang,
              target_lang: targetLang,
              tts_engine: ttsEngine,
              asr_engine: asrEngine,
              voice_name: selectedVoice
            });

            if (dubRes.status === 'success' && dubRes.original_text) {
              const newEntry = {
                id: Math.random().toString(),
                time: new Date().toLocaleTimeString(),
                orig: dubRes.original_text,
                trans: dubRes.translated_text,
                audioUrl: dubRes.audio_url
              };
              setLiveLogs(prev => [newEntry, ...prev]);

              if (dubRes.audio_url && liveAudioPlayerRef.current) {
                liveAudioPlayerRef.current.src = dubRes.audio_url;
                liveAudioPlayerRef.current.play().catch(console.error);
              }
            }
          } catch (err) {
            console.error('Live chunk dubbing error:', err);
          }
        }
      };

      recorder.start();

      liveIntervalRef.current = setInterval(() => {
        if (recorder.state === 'recording') {
          recorder.stop();
          recorder.start();
        }
      }, 4500);

    } catch (err: any) {
      alert(`Could not access audio stream: ${err.message}`);
      setIsLiveActive(false);
    }
  };

  const stopLiveCapture = () => {
    if (liveIntervalRef.current) {
      clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
    }
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
      mediaRecorderRef.current = null;
    }
    setIsLiveActive(false);
  };

  // Re-synthesize single segment
  const handleUpdateSegment = async (segmentId: number, newText: string) => {
    try {
      const res = await api.updateDubbingSegment({
        segment_id: segmentId,
        translated_text: newText,
        voice_name: selectedVoice
      });

      setSegments(prev => prev.map(s => s.id === segmentId ? { ...s, translated_text: res.translated_text, audio_path: res.audio_path } : s));
    } catch (err: any) {
      alert(`Segment update error: ${err.message}`);
    }
  };

  // Dual Player control
  const togglePlayPause = () => {
    if (isPlaying) {
      originalAudioRef.current?.pause();
      dubbedAudioRef.current?.pause();
      setIsPlaying(false);
    } else {
      if (activeTrack === 'original' || activeTrack === 'both') {
        originalAudioRef.current?.play().catch(console.error);
      }
      if (activeTrack === 'dubbed' || activeTrack === 'both') {
        dubbedAudioRef.current?.play().catch(console.error);
      }
      setIsPlaying(true);
    }
  };

  const handleTimeSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (originalAudioRef.current) originalAudioRef.current.currentTime = time;
    if (dubbedAudioRef.current) dubbedAudioRef.current.currentTime = time;
  };

  const handleTrackChange = (track: 'dubbed' | 'original' | 'both') => {
    setActiveTrack(track);
    if (track === 'dubbed') {
      if (originalAudioRef.current) originalAudioRef.current.volume = 0;
      if (dubbedAudioRef.current) dubbedAudioRef.current.volume = dubbedVolume;
    } else if (track === 'original') {
      if (originalAudioRef.current) originalAudioRef.current.volume = originalVolume;
      if (dubbedAudioRef.current) dubbedAudioRef.current.volume = 0;
    } else {
      if (originalAudioRef.current) originalAudioRef.current.volume = originalVolume;
      if (dubbedAudioRef.current) dubbedAudioRef.current.volume = dubbedVolume;
    }
  };

  const downloadDubbedSRT = () => {
    if (!segments || segments.length === 0) return;
    let srt = '';
    const pad = (n: number) => n.toString().padStart(2, '0');
    const formatTime = (sec: number) => {
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = Math.floor(sec % 60);
      const ms = Math.floor((sec % 1) * 1000);
      return `${pad(h)}:${pad(m)}:${pad(s)},${ms.toString().padStart(3, '0')}`;
    };

    segments.forEach((seg, i) => {
      srt += `${i + 1}\n${formatTime(seg.start_time)} --> ${formatTime(seg.end_time)}\n${(seg.translated_text || seg.original_text).trim()}\n\n`;
    });

    const blob = new Blob([srt], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dubbed_${targetLang}_subtitles.srt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const selectedFile = files.find(f => f.id === selectedFileId);

  return (
    <div className="space-y-6">
      {/* Hidden Live Audio Player */}
      <audio ref={liveAudioPlayerRef} hidden />

      {/* Main Studio Control Deck */}
      <div className="professional-card">
        <div className="card-header-traditional">
          <div>
            <h2 className="card-title">
              <Mic2 className="text-indigo-400" size={22} />
              <span>Realtime AI Speech & Video Dubbing Studio</span>
            </h2>
            <p className="card-subtitle">
              Professional YouTube-style dubbing for Tamil and South Indian languages with original pitch, tempo, and emotion preservation.
            </p>
          </div>
        </div>

        {/* STEP 1: SELECT SOURCE MODE (VIBRANT THEMED CARDS) */}
        <div className="step-banner">
          <span className="step-badge">Step 1</span>
          <span className="text-sm font-semibold text-slate-200">Select Input Source Mode:</span>
        </div>

        <div className="source-selector-grid">
          <button
            type="button"
            className={`source-mode-card ${sourceMode === 'file' ? 'active' : ''}`}
            onClick={() => { setSourceMode('file'); stopLiveCapture(); }}
          >
            <div className="source-icon-box">
              <FileAudio size={20} />
            </div>
            <div className="source-mode-content">
              <span className="source-mode-title">1. Single File</span>
              <span className="source-mode-desc">Upload or pick from library</span>
            </div>
          </button>

          <button
            type="button"
            className={`source-mode-card ${sourceMode === 'folder' ? 'active' : ''}`}
            onClick={() => { setSourceMode('folder'); stopLiveCapture(); }}
          >
            <div className="source-icon-box">
              <Folder size={20} />
            </div>
            <div className="source-mode-content">
              <span className="source-mode-title">2. Folder Batch</span>
              <span className="source-mode-desc">Dub all media in Windows folder</span>
            </div>
          </button>

          <button
            type="button"
            className={`source-mode-card ${sourceMode === 'mic_live' ? 'active' : ''}`}
            onClick={() => { setSourceMode('mic_live'); stopLiveCapture(); }}
          >
            <div className="source-icon-box">
              <Radio size={20} />
            </div>
            <div className="source-mode-content">
              <span className="source-mode-title">3. Voice-to-Voice (Mic)</span>
              <span className="source-mode-desc">Live speech streaming dubber</span>
            </div>
          </button>

          <button
            type="button"
            className={`source-mode-card ${sourceMode === 'system_sound' ? 'active' : ''}`}
            onClick={() => { setSourceMode('system_sound'); stopLiveCapture(); }}
          >
            <div className="source-icon-box">
              <Monitor size={20} />
            </div>
            <div className="source-mode-content">
              <span className="source-mode-title">4. System Sound Capture</span>
              <span className="source-mode-desc">Capture YouTube & desktop audio</span>
            </div>
          </button>
        </div>

        {/* STEP 2: CONFIGURE AI ENGINES & LANGUAGES */}
        <div className="step-banner">
          <span className="step-badge">Step 2</span>
          <span className="text-sm font-semibold text-slate-200">Configure Languages & AI Engines:</span>
        </div>

        <div className="engine-config-grid">
          {/* Source Language */}
          <div className="form-group">
            <label className="form-label">Source Language</label>
            <select 
              className="form-select"
              value={sourceLang}
              onChange={(e) => setSourceLang(e.target.value)}
            >
              <option value="auto">Auto Detect</option>
              <option value="en">English</option>
              <option value="ta">Tamil (தமிழ்)</option>
              <option value="hi">Hindi (हिन्दी)</option>
              <option value="te">Telugu (తెలుగు)</option>
              <option value="ml">Malayalam (മലയാളം)</option>
              <option value="kn">Kannada (ಕನ್ನಡ)</option>
            </select>
          </div>

          {/* Target Language */}
          <div className="form-group">
            <label className="form-label">Target Language</label>
            <select 
              className="form-select font-bold text-indigo-300"
              value={targetLang}
              onChange={(e) => setTargetLang(e.target.value)}
            >
              <option value="ta">Tamil (தமிழ்)</option>
              <option value="te">Telugu (తెలుగు)</option>
              <option value="ml">Malayalam (മലയാളം)</option>
              <option value="kn">Kannada (ಕನ್ನಡ)</option>
              <option value="hi">Hindi (हिन्दी)</option>
              <option value="en">English</option>
              <option value="mr">Marathi (मराठी)</option>
              <option value="bn">Bengali (বাংলা)</option>
              <option value="gu">Gujarati (ગુજરાતી)</option>
            </select>
          </div>

          {/* ASR Engine */}
          <div className="form-group">
            <label className="form-label">ASR Engine (STT)</label>
            <select
              className="form-select text-xs"
              value={asrEngine}
              onChange={(e) => setAsrEngine(e.target.value)}
            >
              <option value="faster_whisper">Faster-Whisper (Local CPU)</option>
              <option value="sarvam_saaras">Sarvam Saaras v2 (Cloud)</option>
              {customModels.map((m, idx) => (
                <option key={idx} value={`custom:${m.path}`}>
                  Custom: {m.name} ({m.extension})
                </option>
              ))}
            </select>
          </div>

          {/* TTS Engine */}
          <div className="form-group">
            <label className="form-label">TTS Voice Engine</label>
            <select
              className="form-select text-xs"
              value={ttsEngine}
              onChange={(e) => setTtsEngine(e.target.value)}
            >
              <option value="edge_neural">Edge Neural (High Fidelity)</option>
              <option value="sarvam_bulbul">Sarvam Bulbul (Indic Clone)</option>
            </select>
          </div>

          {/* Voice Model */}
          <div className="form-group">
            <label className="form-label">Voice Model</label>
            <select 
              className="form-select text-xs"
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
            >
              {availableVoices[targetLang]?.voices?.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* STEP 3: DYNAMIC CONTROLS FOR EACH MODE */}
        {sourceMode === 'file' && (
          <div className="pt-4 border-t border-white/10 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
              <div className="form-group md:col-span-2">
                <label className="form-label">Choose File from Indexed Media Library</label>
                <select 
                  className="form-select"
                  value={selectedFileId || ''}
                  onChange={(e) => setSelectedFileId(Number(e.target.value))}
                  disabled={isProcessing}
                >
                  {files.map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.filename} ({f.duration ? `${Math.round(f.duration)}s` : 'Unindexed'})
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex gap-2 items-end">
                <label className="btn-secondary cursor-pointer flex-1 flex items-center justify-center gap-2 h-[42px]">
                  <FileAudio size={16} />
                  <span>Upload File</span>
                  <input type="file" accept="audio/*,video/*" onChange={handleFileUpload} hidden />
                </label>

                <button 
                  className="btn-primary flex items-center justify-center gap-2 px-6 h-[42px]"
                  onClick={handleStartDubbing}
                  disabled={!selectedFileId || isProcessing}
                >
                  {isProcessing ? <Loader2 className="animate-spin" size={18} /> : <Sparkles size={18} />}
                  <span>{isProcessing ? 'Dubbing...' : 'Start Dubbing'}</span>
                </button>
              </div>
            </div>

            {(isProcessing || processStatus) && (
              <div className="mt-4 p-3 rounded-lg bg-gray-800/80 border border-gray-700">
                <div className="progress-bar-bg mb-2">
                  <motion.div 
                    className="progress-bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                  />
                </div>
                <div className="status-text">
                  <span className="text-indigo-300">{processStatus}</span>
                  <span className="font-bold">{progress}%</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* FOLDER BATCH CONTROLS */}
        {sourceMode === 'folder' && (
          <div className="pt-4 border-t border-white/10 space-y-4">
            <div className="p-4 rounded-xl bg-gray-800/50 border border-gray-700">
              <h3 className="text-sm font-bold text-indigo-300 flex items-center gap-2 mb-2">
                <Folder size={16} />
                <span>Batch Folder Translation & Dubbing</span>
              </h3>
              <p className="text-xs text-gray-400 mb-4">
                Enter any directory path on your Windows machine. All audio and video files inside will be transcribed and dubbed with same-voice pitch matching.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="form-group">
                  <label className="form-label">Source Folder Path (Windows)</label>
                  <input
                    type="text"
                    className="form-input text-xs font-mono"
                    placeholder="e.g. C:\Users\ela\Downloads\Videos"
                    value={batchFolderDir}
                    onChange={(e) => setBatchFolderDir(e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Target Output Folder (Optional)</label>
                  <input
                    type="text"
                    className="form-input text-xs font-mono"
                    placeholder="Leave empty for auto: SourceFolder\Dubbed_TA"
                    value={batchOutputDir}
                    onChange={(e) => setBatchOutputDir(e.target.value)}
                  />
                </div>
              </div>

              <button
                className="btn-primary flex items-center gap-2 px-6 py-2.5"
                onClick={handleStartBatchFolder}
                disabled={isBatchProcessing || !batchFolderDir.trim()}
              >
                {isBatchProcessing ? <Loader2 className="animate-spin" size={18} /> : <Sparkles size={18} />}
                <span>{isBatchProcessing ? 'Batch Dubbing...' : 'Start Batch Dubbing All Files'}</span>
              </button>
            </div>

            {batchResult && (
              <div className="p-4 rounded-xl bg-gray-900 border border-emerald-500/30">
                <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm mb-2">
                  <CheckCircle2 size={18} />
                  <span>Found {batchResult.total_files} file(s) for batch dubbing!</span>
                </div>
                <p className="text-xs text-gray-400 font-mono mb-2">
                  Output Directory: {batchResult.output_directory}
                </p>
                <div className="space-y-1.5 max-h-40 overflow-y-auto">
                  {batchResult.files.map((f: string, i: number) => (
                    <div key={i} className="text-xs text-gray-300 flex items-center gap-2 p-1.5 rounded bg-gray-800">
                      <FileAudio size={14} className="text-indigo-400" />
                      <span className="truncate">{f}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* LIVE MIC / SYSTEM AUDIO STREAMING */}
        {(sourceMode === 'mic_live' || sourceMode === 'system_sound') && (
          <div className="pt-4 border-t border-white/10 space-y-4">
            <div className="p-4 rounded-xl bg-gray-800/50 border border-gray-700 flex flex-col md:flex-row justify-between items-center gap-4">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  {sourceMode === 'mic_live' ? <Radio className="text-indigo-400" size={18} /> : <Monitor className="text-indigo-400" size={18} />}
                  <span>{sourceMode === 'mic_live' ? 'Live Microphone Speech Dubber' : 'Live YouTube & System Audio Stream Dubber'}</span>
                </h3>
                <p className="text-xs text-gray-400 mt-1">
                  {sourceMode === 'mic_live' 
                    ? 'Speak in any language. System transcribes, pivots to English, and outputs in Tamil with identical vocal tone.' 
                    : 'Captures desktop sound (YouTube video, meeting, audio). Live translates & speaks in Tamil with matching voice.'}
                </p>
              </div>

              <div>
                {isLiveActive ? (
                  <button
                    className="px-6 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-bold flex items-center gap-2 shadow-lg transition-all animate-pulse"
                    onClick={stopLiveCapture}
                  >
                    <StopCircle size={18} />
                    <span>Stop Live Capture</span>
                  </button>
                ) : (
                  <button
                    className="btn-primary flex items-center gap-2 px-6 py-3"
                    onClick={() => startLiveCapture(sourceMode)}
                  >
                    <Mic2 size={18} />
                    <span>Start {sourceMode === 'mic_live' ? 'Voice-to-Voice' : 'System Audio Capture'}</span>
                  </button>
                )}
              </div>
            </div>

            {/* Live Streaming Log */}
            <div className="p-4 rounded-xl bg-gray-900 border border-gray-800">
              <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                <Languages size={14} className="text-indigo-400" />
                <span>Live Speech Translation Stream ({liveLogs.length} chunks)</span>
              </h4>

              {liveLogs.length === 0 ? (
                <div className="text-center py-6 text-gray-500 text-xs">
                  {isLiveActive ? 'Listening to speech stream... speak or play audio now!' : 'Click "Start" to begin realtime speech capture and translation.'}
                </div>
              ) : (
                <div className="space-y-2.5 max-h-56 overflow-y-auto pr-1">
                  {liveLogs.map((log) => (
                    <div key={log.id} className="p-3 rounded-lg bg-gray-800 border border-gray-700 flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 text-xs text-gray-400 mb-1">
                          <span className="font-mono">{log.time}</span>
                          <span>•</span>
                          <span className="text-gray-300 italic">"{log.orig}"</span>
                        </div>
                        <p className="text-sm font-semibold text-indigo-300">
                          {log.trans}
                        </p>
                      </div>
                      {log.audioUrl && (
                        <audio controls src={log.audioUrl} className="h-8 max-w-[200px]" />
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* DUAL-TRACK AUDIO PLAYER & TIMELINE (Mode 1) */}
      {sourceMode === 'file' && selectedFile && (
        <div className="professional-card">
          <div className="card-header-traditional">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center">
                <Music size={20} />
              </div>
              <div>
                <h3 className="card-title">{selectedFile.filename}</h3>
                <p className="card-subtitle">
                  Interactive Dual-Track Crossfading Player • Original Speech vs {targetLang.toUpperCase()} Dubbed Voice
                </p>
              </div>
            </div>

            {/* Track Switcher */}
            <div className="flex items-center gap-1.5 bg-gray-800 p-1 rounded-lg border border-gray-700">
              <button
                className={`px-3 py-1 rounded text-xs font-semibold transition-all ${activeTrack === 'dubbed' ? 'bg-indigo-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}
                onClick={() => handleTrackChange('dubbed')}
              >
                Dubbed ({targetLang.toUpperCase()})
              </button>
              <button
                className={`px-3 py-1 rounded text-xs font-semibold transition-all ${activeTrack === 'original' ? 'bg-indigo-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}
                onClick={() => handleTrackChange('original')}
              >
                Original Audio
              </button>
              <button
                className={`px-3 py-1 rounded text-xs font-semibold transition-all ${activeTrack === 'both' ? 'bg-indigo-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}
                onClick={() => handleTrackChange('both')}
              >
                Mixed / Voiceover
              </button>
            </div>
          </div>

          <audio 
            ref={originalAudioRef} 
            src={api.getStreamUrl(selectedFile.file_path)} 
            onTimeUpdate={() => setCurrentTime(originalAudioRef.current?.currentTime || 0)}
            onLoadedMetadata={() => setDuration(originalAudioRef.current?.duration || selectedFile.duration || 0)}
            onEnded={() => setIsPlaying(false)}
          />
          {currentProject?.result_audio_path && (
            <audio 
              ref={dubbedAudioRef} 
              src={api.getStreamUrl(currentProject.result_audio_path)} 
            />
          )}

          {/* Timeline Bar */}
          <div className="mb-4">
            <input 
              type="range" 
              min={0} 
              max={duration || 100} 
              step={0.1}
              value={currentTime} 
              onChange={handleTimeSeek}
              className="w-full accent-indigo-500 cursor-pointer h-2 bg-gray-800 rounded-lg"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1 font-mono">
              <span>{Math.floor(currentTime)}s</span>
              <span>{Math.floor(duration)}s</span>
            </div>
          </div>

          {/* Playback Controls & Volume Crossfaders */}
          <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
            <div className="flex items-center gap-4">
              <button 
                className="w-11 h-11 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center shadow-lg transition-transform hover:scale-105"
                onClick={togglePlayPause}
              >
                {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
              </button>

              <div className="flex items-center gap-3 bg-gray-800 px-3.5 py-2 rounded-lg border border-gray-700">
                <Volume2 size={16} className="text-indigo-400" />
                <div className="flex flex-col">
                  <span className="text-[11px] text-gray-400 uppercase font-semibold">Dubbed Speech Volume</span>
                  <input 
                    type="range" min={0} max={1} step={0.05} 
                    value={dubbedVolume}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      setDubbedVolume(v);
                      if (dubbedAudioRef.current) dubbedAudioRef.current.volume = v;
                    }}
                    className="accent-indigo-500 w-24 h-1.5"
                  />
                </div>
              </div>

              <div className="flex items-center gap-3 bg-gray-800 px-3.5 py-2 rounded-lg border border-gray-700">
                <Volume2 size={16} className="text-gray-400" />
                <div className="flex flex-col">
                  <span className="text-[11px] text-gray-400 uppercase font-semibold">Original Background Volume</span>
                  <input 
                    type="range" min={0} max={1} step={0.05} 
                    value={originalVolume}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      setOriginalVolume(v);
                      if (originalAudioRef.current) originalAudioRef.current.volume = v;
                    }}
                    className="accent-gray-400 w-24 h-1.5"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button 
                onClick={downloadDubbedSRT} 
                className="btn-secondary text-xs flex items-center gap-2"
                disabled={segments.length === 0}
              >
                <Captions size={14} />
                <span>Export Subtitles (SRT)</span>
              </button>

              {currentProject?.result_audio_path && (
                <a 
                  href={api.getStreamUrl(currentProject.result_audio_path)}
                  download={`dubbed_${targetLang}_${selectedFile.filename}.wav`}
                  className="btn-primary text-xs flex items-center gap-2"
                >
                  <Download size={14} />
                  <span>Download Dubbed Audio Track</span>
                </a>
              )}
            </div>
          </div>
        </div>
      )}

      {/* SEGMENT-BY-SEGMENT TIMELINE EDITOR (Mode 1) */}
      {sourceMode === 'file' && segments.length > 0 && (
        <div className="professional-card">
          <div className="card-header-traditional">
            <div>
              <h3 className="card-title">
                <Sliders className="text-indigo-400" size={18} />
                <span>Segment-by-Segment Timeline Editor</span>
              </h3>
              <p className="card-subtitle">
                Review, edit, and fine-tune translated lines or re-synthesize individual audio segments.
              </p>
            </div>
            <span className="badge-pill bg-indigo-500/20 text-indigo-300">
              {segments.length} Segments
            </span>
          </div>

          <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
            {segments.map((seg) => (
              <div 
                key={seg.id} 
                className={`segment-row ${
                  currentTime >= seg.start_time && currentTime <= seg.end_time ? 'active' : ''
                }`}
              >
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-bold">
                      [{seg.start_time}s - {seg.end_time}s]
                    </span>
                    <span className="text-xs text-gray-400">
                      Duration: {(seg.end_time - seg.start_time).toFixed(1)}s
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    {seg.audio_path && (
                      <audio controls src={api.getStreamUrl(seg.audio_path)} className="h-7 max-w-[180px]" />
                    )}
                    <button 
                      className="btn-icon text-xs text-indigo-400"
                      title="Re-synthesize this line"
                      onClick={() => handleUpdateSegment(seg.id, seg.translated_text || seg.original_text)}
                    >
                      <RefreshCw size={14} />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <span className="text-[11px] text-gray-400 font-semibold uppercase mb-1 block">Original Line ({sourceLang.toUpperCase()})</span>
                    <p className="text-xs text-gray-300 p-2 rounded bg-gray-900 border border-gray-800">
                      {seg.original_text}
                    </p>
                  </div>

                  <div>
                    <span className="text-[11px] text-indigo-400 font-semibold uppercase mb-1 block">Dubbed Line ({targetLang.toUpperCase()})</span>
                    <input 
                      type="text" 
                      className="form-input text-xs"
                      value={seg.translated_text || ''}
                      onChange={(e) => {
                        const val = e.target.value;
                        setSegments(prev => prev.map(s => s.id === seg.id ? { ...s, translated_text: val } : s));
                      }}
                      onBlur={(e) => handleUpdateSegment(seg.id, e.target.value)}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TRANSCRIPT SEARCH & VIEWER */}
      {sourceMode === 'file' && selectedFile && segments.length > 0 && (
        <div className="professional-card">
          <div className="card-header-traditional">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="card-title">
                  <List className="text-indigo-400" size={18} />
                  <span>Full Transcript Search</span>
                </h3>
                <p className="card-subtitle">
                  Search within this file's transcript and jump to exact timestamps
                </p>
              </div>
            </div>
          </div>

          <div className="mb-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
              <input
                type="text"
                className="form-input pl-10"
                placeholder="Search transcript (e.g. 'machine learning', 'hello world')..."
                value={transcriptSearchQuery}
                onChange={(e) => setTranscriptSearchQuery(e.target.value)}
              />
            </div>
          </div>

          <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
            {filteredSegments.map((seg) => (
              <div key={seg.id} className={`p-3 rounded-lg border transition-all ${
                transcriptCurrentTime >= seg.start_time && transcriptCurrentTime <= seg.end_time
                  ? 'bg-indigo-950/30 border-indigo-500/50 ring-1 ring-indigo-500/20'
                  : 'bg-gray-900 border-gray-800 hover:border-gray-700'
              }`}>
                <div className="flex justify-between items-start gap-3 mb-2">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="font-mono text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-bold">
                      [{seg.start_time.toFixed(1)}s - {seg.end_time.toFixed(1)}s]
                    </span>
                    <span className="text-xs text-gray-400">
                      Duration: {(seg.end_time - seg.start_time).toFixed(1)}s
                    </span>
                    {seg.speaker_tag && (
                      <span className="badge-pill bg-indigo-500/20 text-indigo-300 text-[10px]">
                        {seg.speaker_tag}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <audio 
                      controls 
                      src={api.getClipUrl(selectedFile.file_path, seg.start_time, seg.end_time)}
                      className="h-7 max-w-[180px]"
                    />
                    <button
                      className="btn-icon text-indigo-400 hover:text-indigo-300"
                      title="Jump to this timestamp"
                      onClick={() => {
                        setTranscriptCurrentTime(seg.start_time);
                        if (originalAudioRef.current) {
                          originalAudioRef.current.currentTime = seg.start_time;
                          originalAudioRef.current.play().catch(console.error);
                        }
                      }}
                    >
                      <Play size={14} />
                    </button>
                  </div>
                </div>
                <p className="text-sm text-gray-200 leading-relaxed select-all">
                  {highlightSearchText(seg.original_text || '', transcriptSearchQuery)}
                </p>
                {seg.translated_text && (
                  <p className="text-sm text-indigo-300 leading-relaxed mt-1 select-all">
                    {highlightSearchText(seg.translated_text, transcriptSearchQuery)}
                  </p>
                )}
              </div>
            ))}
            {filteredSegments.length === 0 && transcriptSearchQuery && (
              <div className="text-center py-8 text-gray-500 text-sm">
                No matches found for "{transcriptSearchQuery}"
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
