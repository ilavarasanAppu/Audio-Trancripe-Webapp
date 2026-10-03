import React, { useState, useEffect } from 'react';
import { 
  Search, FolderSearch, FileAudio, 
  Sparkles, User, Loader2,
  Database, RefreshCw, HelpCircle
} from 'lucide-react';
import { api, type AudioFile, type RAGQueryResult } from '../services/api';

export const AudioRAG: React.FC = () => {
  const [scanDir, setScanDir] = useState('c:\\Users\\ela\\Downloads\\Github\\Audio-Trancripe---Webapp--windows\\backend\\storage');
  const [isRecursive, setIsRecursive] = useState(true);
  const [isScanning, setIsScanning] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [files, setFiles] = useState<AudioFile[]>([]);
  const [query, setQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [ragResult, setRagResult] = useState<RAGQueryResult | null>(null);

  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      const data = await api.getFiles();
      setFiles(data.files || []);
    } catch (err) {
      console.error('Error loading files:', err);
    }
  };

  const handleScan = async () => {
    if (!scanDir) return;
    setIsScanning(true);
    try {
      await api.scanDirectory(scanDir, isRecursive);
      await loadFiles();
    } catch (err: any) {
      alert(`Scan failed: ${err.message}`);
    } finally {
      setIsScanning(false);
    }
  };

  const handleIngestAll = async () => {
    setIsIngesting(true);
    try {
      await api.ingestAll();
      await loadFiles();
      const poll = setInterval(async () => {
        const data = await api.getFiles();
        setFiles(data.files || []);
        const hasPending = data.files.some(f => f.status === 'processing' || f.status === 'pending');
        if (!hasPending) {
          clearInterval(poll);
          setIsIngesting(false);
        }
      }, 3000);
    } catch (err: any) {
      alert(`Ingestion error: ${err.message}`);
      setIsIngesting(false);
    }
  };

  const handleIngestSingle = async (fileId: number) => {
    try {
      await api.ingestFile(fileId);
    } catch (err: any) {
      alert(`Transcription error: ${err.message}`);
    } finally {
      await loadFiles();
    }
  };

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;
    setIsQuerying(true);
    try {
      const res = await api.queryRAG(query.trim());
      setRagResult(res);
    } catch (err: any) {
      alert(`Search error: ${err.message}`);
    } finally {
      setIsQuerying(false);
    }
  };

  const sampleQuestions = [
    "What are the main topics discussed?",
    "Find all mentions of AI, Tamil models, or transcription",
    "Where is the speaker talking about pricing or features?",
    "Summarize what the speaker said at the beginning"
  ];

  return (
    <div className="space-y-6">
      {/* 1. Directory Scanner & Knowledge Base Indexer */}
      <div className="professional-card">
        <div className="card-header-traditional">
          <div>
            <h2 className="card-title">
              <Database className="text-indigo-400" size={20} />
              <span>Audio & Video Knowledge Base Scanner</span>
            </h2>
            <p className="card-subtitle">
              Scan folders on your computer to discover and index audio/video transcripts for instant vector search.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button className="btn-secondary text-xs" onClick={loadFiles}>
              <RefreshCw size={14} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end mb-4">
          <div className="form-group md:col-span-2">
            <label className="form-label">Windows Directory Path</label>
            <input 
              type="text" 
              className="form-input text-xs font-mono"
              placeholder="e.g. C:\Users\ela\Music or C:\Users\ela\Downloads"
              value={scanDir}
              onChange={(e) => setScanDir(e.target.value)}
            />
          </div>

          <div className="flex items-center gap-2 pb-2">
            <label className="text-xs text-gray-300 flex items-center gap-2 cursor-pointer select-none">
              <input 
                type="checkbox" 
                checked={isRecursive} 
                onChange={(e) => setIsRecursive(e.target.checked)}
                className="accent-indigo-500 rounded"
              />
              <span>Scan Subfolders</span>
            </label>
          </div>

          <div className="flex gap-2">
            <button 
              className="btn-primary text-xs flex-1"
              onClick={handleScan}
              disabled={isScanning}
            >
              {isScanning ? <Loader2 className="animate-spin" size={14} /> : <FolderSearch size={14} />}
              <span>{isScanning ? 'Scanning...' : 'Scan Folder'}</span>
            </button>

            <button 
              className="btn-secondary text-xs flex-1"
              onClick={handleIngestAll}
              disabled={isIngesting || files.length === 0}
            >
              {isIngesting ? <Loader2 className="animate-spin" size={14} /> : <Sparkles size={14} />}
              <span>{isIngesting ? 'Indexing...' : 'Index All'}</span>
            </button>
          </div>
        </div>

        {/* Files Table */}
        <div className="table-container max-h-56 overflow-y-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>File Name</th>
                <th>Format</th>
                <th>Duration</th>
                <th>Status</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {files.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-6 text-gray-500 text-xs">
                    No files scanned yet. Enter a directory path above and click "Scan Folder".
                  </td>
                </tr>
              ) : (
                files.map((file) => (
                  <tr key={file.id}>
                    <td className="font-medium flex items-center gap-2">
                      <FileAudio size={14} className="text-indigo-400 shrink-0" />
                      <span className="truncate max-w-[280px]">{file.filename}</span>
                    </td>
                    <td><span className="uppercase text-xs font-mono">{file.file_format}</span></td>
                    <td className="font-mono text-xs">{file.duration ? `${Math.round(file.duration)}s` : '-'}</td>
                    <td>
                      <span className={`badge-pill ${
                        file.status === 'indexed' ? 'bg-emerald-500/20 text-emerald-300' :
                        file.status === 'processing' ? 'bg-amber-500/20 text-amber-300' :
                        'bg-gray-700 text-gray-300'
                      }`}>
                        {file.status}
                      </span>
                    </td>
                    <td>
                      {file.status === 'pending' || file.status === 'error' || (file.status === 'indexed' && (file.chunk_count ?? 0) === 0) ? (
                        <button 
                          className="btn-secondary text-xs py-1 px-2.5"
                          onClick={() => handleIngestSingle(file.id)}
                        >
                          {file.status === 'pending' ? 'Index Now' : 'Retry Transcription'}
                        </button>
                      ) : (
                        <span className="text-xs text-gray-500">Ready</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. Natural Language Semantic Search & Audio Q&A */}
      <div className="professional-card">
        <div className="card-header-traditional">
          <div>
            <h2 className="card-title">
              <Search className="text-indigo-400" size={20} />
              <span>Natural Language Audio Q&A and Clip Search</span>
            </h2>
            <p className="card-subtitle">
              Ask questions to your media library or search for specific phrases to jump directly to exact audio timestamps.
            </p>
          </div>
        </div>

        <form onSubmit={handleSearch} className="flex gap-2 mb-4">
          <input 
            type="text" 
            className="form-input text-sm"
            placeholder="Ask a question or search words (e.g. 'What did the speaker say about machine learning?')..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button 
            type="submit" 
            className="btn-primary flex items-center gap-2 px-6 shrink-0"
            disabled={isQuerying || !query.trim()}
          >
            {isQuerying ? <Loader2 className="animate-spin" size={16} /> : <Search size={16} />}
            <span>{isQuerying ? 'Searching...' : 'Search Audio'}</span>
          </button>
        </form>

        {/* Quick Questions */}
        <div className="flex flex-wrap items-center gap-2 mb-6">
          <span className="text-xs text-gray-400 flex items-center gap-1">
            <HelpCircle size={12} />
            <span>Suggested Questions:</span>
          </span>
          {sampleQuestions.map((q, idx) => (
            <button 
              key={idx}
              type="button"
              className="text-xs px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 transition-all"
              onClick={() => { setQuery(q); }}
            >
              {q}
            </button>
          ))}
        </div>

        {/* Search Results & Audio Clips */}
        {ragResult && (
          <div className="space-y-4 pt-4 border-t border-white/10">
            {/* AI Synthesized Answer */}
            {ragResult.answer && (
              <div className="p-4 rounded-xl bg-indigo-950/20 border border-indigo-500/30">
                <h4 className="text-xs font-bold uppercase text-indigo-300 tracking-wider mb-1 flex items-center gap-1.5">
                  <Sparkles size={14} />
                  <span>AI Synthesized Answer</span>
                </h4>
                <p className="text-sm text-gray-200 leading-relaxed">{ragResult.answer}</p>
              </div>
            )}

            {/* Matched Audio Segments */}
            <h4 className="text-xs font-bold uppercase text-gray-400 tracking-wider">
              Relevant Audio Clips & Exact Timestamp Mentions ({ragResult.citations?.length || 0})
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {ragResult.citations?.map((m) => (
                <div key={m.chunk_id} className="p-3.5 rounded-xl bg-gray-900 border border-gray-800 hover:border-gray-700 space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-semibold text-indigo-300 truncate max-w-[200px]">{m.filename}</span>
                    <span className="font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-bold">
                      {Math.floor(m.start_time)}s - {Math.floor(m.end_time)}s
                    </span>
                  </div>

                  <p className="text-xs text-gray-300 italic p-2 rounded bg-gray-800">
                    "{m.text}"
                  </p>

                  <div className="flex justify-between items-center pt-1">
                    <span className="text-[11px] text-gray-400 flex items-center gap-1">
                      <User size={12} />
                      <span>{m.speaker_name}</span>
                    </span>

                    <audio 
                      controls 
                      src={api.getClipUrl(m.file_path, m.start_time, m.end_time)}
                      className="h-7 max-w-[180px]"
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
