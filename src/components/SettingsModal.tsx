import React, { useState, useEffect } from 'react';
import { 
  X, Key, Cpu, Globe, Check, AlertCircle, Save, 
  Activity, Wrench, RefreshCw, CheckCircle2, Mic2, Volume2, Languages,
  Folder, FileCode, CheckCircle, Plus, Trash2, Layers, Search
} from 'lucide-react';
import { api, type DoctorReport, type CustomModelItem } from '../services/api';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const [sarvamApiKey, setSarvamApiKey] = useState('');
  const [whisperModelSize, setWhisperModelSize] = useState('base');
  const [defaultTargetLang, setDefaultTargetLang] = useState('ta');
  const [defaultAsrEngine, setDefaultAsrEngine] = useState('faster_whisper');
  const [defaultTtsEngine, setDefaultTtsEngine] = useState('edge_neural');
  const [defaultTranslationEngine, setDefaultTranslationEngine] = useState('google_deep');
  
  // Multiple Custom Local Model Folders & Selection
  const [customModelsDirs, setCustomModelsDirs] = useState<string[]>([]);
  const [newFolderPath, setNewFolderPath] = useState('');
  const [selectedCustomModel, setSelectedCustomModel] = useState('');
  const [detectedModels, setDetectedModels] = useState<CustomModelItem[]>([]);
  const [isScanningModels, setIsScanningModels] = useState(false);
  const [modelScanStatus, setModelScanStatus] = useState('');
  const [modelSearchQuery, setModelSearchQuery] = useState('');

  const [statusMessage, setStatusMessage] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // Doctor diagnostics state
  const [doctorReport, setDoctorReport] = useState<DoctorReport | null>(null);
  const [isCheckingDoctor, setIsCheckingDoctor] = useState(false);
  const [isFixingDoctor, setIsFixingDoctor] = useState(false);
  const [fixMessage, setFixMessage] = useState('');

  useEffect(() => {
    loadSettings();
    loadCustomModels();
    runDoctorCheck();
  }, []);

  const loadSettings = async () => {
    try {
      const data = await api.getSettings();
      if (data) {
        setSarvamApiKey(data.sarvam_api_key || '');
        setWhisperModelSize(data.whisper_model_size || 'base');
        setDefaultTargetLang(data.default_target_lang || 'ta');
        setDefaultAsrEngine(data.default_asr_engine || 'faster_whisper');
        setDefaultTtsEngine(data.default_tts_engine || 'edge_neural');
        setDefaultTranslationEngine(data.default_translation_engine || 'google_deep');
        
        if (data.custom_models_dirs && Array.isArray(data.custom_models_dirs)) {
          setCustomModelsDirs(data.custom_models_dirs);
        } else if (data.custom_models_dir) {
          setCustomModelsDirs([data.custom_models_dir]);
        }
        
        setSelectedCustomModel(data.selected_custom_model || '');
      }
    } catch (err) {
      console.error('Error loading settings:', err);
    }
  };

  const loadCustomModels = async () => {
    try {
      const data = await api.getCustomModels();
      if (data) {
        setDetectedModels(data.models || []);
        if (data.custom_models_dirs && data.custom_models_dirs.length > 0) {
          setCustomModelsDirs(data.custom_models_dirs);
        }
        if (data.selected_custom_model) {
          setSelectedCustomModel(data.selected_custom_model);
        }
      }
    } catch (err) {
      console.error('Error loading custom models:', err);
    }
  };

  const handleAddFolder = () => {
    const trimmed = newFolderPath.trim();
    if (!trimmed) return;
    if (!customModelsDirs.includes(trimmed)) {
      const updated = [...customModelsDirs, trimmed];
      setCustomModelsDirs(updated);
      setNewFolderPath('');
      handleScanMultipleFolders(updated);
    } else {
      setNewFolderPath('');
    }
  };

  const handleRemoveFolder = (folderToRemove: string) => {
    const updated = customModelsDirs.filter(f => f !== folderToRemove);
    setCustomModelsDirs(updated);
    handleScanMultipleFolders(updated);
  };

  const handleScanMultipleFolders = async (foldersToScan?: string[]) => {
    const dirs = foldersToScan || customModelsDirs;
    if (dirs.length === 0) {
      setDetectedModels([]);
      setModelScanStatus('No folders specified. Add a directory path above.');
      return;
    }

    setIsScanningModels(true);
    setModelScanStatus(`Gathering models across ${dirs.length} folder(s)...`);
    try {
      const res = await api.scanCustomModels(dirs);
      setDetectedModels(res.models || []);
      setModelScanStatus(`Gathered ${res.total_models} supported model(s) across ${res.directories.length} folder(s)!`);
      window.dispatchEvent(new Event('custom-models-updated'));
    } catch (err: any) {
      setModelScanStatus(`Scan error: ${err.message}`);
    } finally {
      setIsScanningModels(false);
    }
  };

  const runDoctorCheck = async () => {
    setIsCheckingDoctor(true);
    try {
      const report = await api.getDoctor();
      setDoctorReport(report);
    } catch (err) {
      console.error('Doctor check failed:', err);
    } finally {
      setIsCheckingDoctor(false);
    }
  };

  const handleFixDoctor = async () => {
    setIsFixingDoctor(true);
    setFixMessage('Running online auto-installer & dependency repair...');
    try {
      const res = await api.fixDoctor();
      setFixMessage(res.message || 'Auto-repair completed.');
      await runDoctorCheck();
    } catch (err: any) {
      setFixMessage(`Repair failed: ${err.message}`);
    } finally {
      setIsFixingDoctor(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setStatusMessage('');
    try {
      await api.updateSettings({
        sarvam_api_key: sarvamApiKey,
        whisper_model_size: whisperModelSize,
        default_target_lang: defaultTargetLang,
        default_asr_engine: defaultAsrEngine,
        default_tts_engine: defaultTtsEngine,
        default_translation_engine: defaultTranslationEngine,
        custom_models_dirs: customModelsDirs,
        custom_models_dir: customModelsDirs.length > 0 ? customModelsDirs[0] : '',
        selected_custom_model: selectedCustomModel,
      });
      window.dispatchEvent(new Event('custom-models-updated'));
      setStatusMessage('Settings saved successfully!');
      setTimeout(() => {
        setStatusMessage('');
      }, 1500);
    } catch (err: any) {
      setStatusMessage(`Error saving settings: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const filteredModels = detectedModels.filter(m => 
    m.name.toLowerCase().includes(modelSearchQuery.toLowerCase()) ||
    m.format.toLowerCase().includes(modelSearchQuery.toLowerCase()) ||
    m.extension.toLowerCase().includes(modelSearchQuery.toLowerCase()) ||
    (m.folder_origin && m.folder_origin.toLowerCase().includes(modelSearchQuery.toLowerCase()))
  );

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="flex items-center gap-2">
            <Cpu className="text-indigo-400" size={22} />
            <h2>System Doctor & AI Engine Configuration</h2>
          </div>
          <button className="btn-icon" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {/* SECTION 1: MULTIPLE CUSTOM MODEL FOLDERS & GATHERING */}
          <div className="model-library">
            <div className="model-library-header">
              <h3 className="model-library-title">
                <Layers size={14} />
                <span>Local ASR Model Library</span>
              </h3>
              <span className="badge-pill bg-indigo-500/20 text-indigo-300 text-[10px]">
                {customModelsDirs.length} Folder(s) Configured
              </span>
            </div>

            <p className="model-library-intro">
              Add a local folder to scan for models. Faster-Whisper can run CTranslate2 model folders containing both model.bin and config.json; other detected files remain visible in the library but cannot be selected for transcription.
            </p>

            {/* Folder Input + Add Button */}
            <div className="model-folder-toolbar">
              <div className="model-folder-input-wrap">
                <Folder size={16} className="model-folder-icon" />
                <input
                  type="text"
                  className="form-input model-folder-input"
                  placeholder="Paste a local folder path, e.g. D:\AI_Models"
                  value={newFolderPath}
                  onChange={(e) => setNewFolderPath(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleAddFolder(); }}
                />
              </div>

              <button
                type="button"
                className="btn-primary model-folder-action"
                onClick={handleAddFolder}
                disabled={!newFolderPath.trim()}
              >
                <Plus size={14} />
                <span>Add Folder</span>
              </button>

              <button
                type="button"
                className="btn-secondary model-folder-action"
                onClick={() => handleScanMultipleFolders()}
                disabled={isScanningModels || customModelsDirs.length === 0}
              >
                <RefreshCw size={12} className={isScanningModels ? 'animate-spin' : ''} />
                <span>{isScanningModels ? 'Scanning...' : 'Scan All Folders'}</span>
              </button>
            </div>

            {/* Configured Folders List */}
            {customModelsDirs.length > 0 && (
              <div className="model-folder-list">
                <span className="model-library-label">
                  Configured Model Directories:
                </span>
                <div className="model-folder-items">
                  {customModelsDirs.map((dir, idx) => (
                    <div key={idx} className="model-folder-row">
                      <div className="flex items-center gap-2 truncate">
                        <Folder size={12} className="text-indigo-400 shrink-0" />
                        <span className="font-mono text-gray-200 truncate">{dir}</span>
                      </div>
                      <button
                        type="button"
                        className="text-gray-400 hover:text-red-400 p-1 transition-colors"
                        onClick={() => handleRemoveFolder(dir)}
                        title="Remove Folder"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {modelScanStatus && (
              <p className="model-scan-status">{modelScanStatus}</p>
            )}

            {/* Gathered Models List & Selector */}
            <div className="model-results">
              <div className="model-results-header">
                <div className="flex justify-between items-center">
                  <span className="text-[11px] text-gray-400 uppercase font-semibold">
                    All Gathered Models ({detectedModels.length}):
                  </span>
                  {detectedModels.length > 4 && (
                    <div className="relative w-44">
                      <Search size={10} className="absolute left-2 top-2 text-gray-400" />
                      <input
                        type="text"
                        className="form-input pl-6 py-0.5 text-[10px]"
                        placeholder="Filter models..."
                        value={modelSearchQuery}
                        onChange={(e) => setModelSearchQuery(e.target.value)}
                      />
                    </div>
                  )}
                </div>
              </div>

              <div className="model-options-list">
                  {/* Default Built-in Option */}
                  <div
                    role="button"
                    tabIndex={0}
                    className={`model-choice ${selectedCustomModel === '' ? 'selected' : ''}`}
                    onClick={() => setSelectedCustomModel('')}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setSelectedCustomModel(''); }}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-gray-200">Use Standard Faster-Whisper Built-in</span>
                    </div>
                    {selectedCustomModel === '' && <CheckCircle size={14} className="text-indigo-400" />}
                  </div>

                  {filteredModels.map((m, idx) => {
                    const isSelected = selectedCustomModel === m.path;
                    const isSupported = Boolean(m.asr_compatible);
                    return (
                      <div
                        role={isSupported ? 'button' : undefined}
                        tabIndex={isSupported ? 0 : -1}
                        aria-disabled={!isSupported}
                        key={idx}
                        className={`model-choice ${isSelected ? 'selected' : ''} ${isSupported ? '' : 'unsupported'}`}
                        onClick={() => { if (isSupported) setSelectedCustomModel(m.path); }}
                        onKeyDown={(e) => { if (isSupported && (e.key === 'Enter' || e.key === ' ')) setSelectedCustomModel(m.path); }}
                      >
                        <div className="flex items-center gap-2 overflow-hidden">
                          <FileCode size={14} className="text-indigo-400 shrink-0" />
                          <div className="model-choice-copy">
                            <span className="model-choice-name">{m.name}</span>
                            <span className="model-choice-meta">
                              {m.format} • {m.size_mb} MB {m.folder_origin ? `• from: ${m.folder_origin}` : ''}
                            </span>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 shrink-0 ml-2">
                          <span className={`model-support-badge ${isSupported ? 'supported' : ''}`}>
                            {isSupported ? 'ASR ready' : m.extension}
                          </span>
                          {isSelected && <CheckCircle size={14} className="text-indigo-400" />}
                        </div>
                      </div>
                    );
                  })}
                  {detectedModels.length === 0 && (
                    <p className="model-empty-state">No model files found yet. Add a folder above and scan it to populate the ASR selector.</p>
                  )}
              </div>
            </div>
          </div>

          {/* SECTION 2: AI ENGINE PRESETS */}
          <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-4">
            <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider flex items-center gap-2">
              <Cpu size={14} />
              <span>Default AI Engine Presets</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {/* ASR Engine */}
              <div className="form-group">
                <label className="form-label flex items-center gap-1">
                  <Mic2 size={12} className="text-indigo-400" />
                  <span>ASR Engine</span>
                </label>
                <select
                  className="form-select text-xs"
                  value={defaultAsrEngine}
                  onChange={(e) => setDefaultAsrEngine(e.target.value)}
                >
                  <option value="faster_whisper">Faster-Whisper (Local)</option>
                  <option value="sarvam_saaras">Sarvam Saaras (Cloud)</option>
                </select>
              </div>

              {/* TTS Engine */}
              <div className="form-group">
                <label className="form-label flex items-center gap-1">
                  <Volume2 size={12} className="text-indigo-400" />
                  <span>TTS Voice Engine</span>
                </label>
                <select
                  className="form-select text-xs"
                  value={defaultTtsEngine}
                  onChange={(e) => setDefaultTtsEngine(e.target.value)}
                >
                  <option value="edge_neural">Edge Neural (High-Fi)</option>
                  <option value="sarvam_bulbul">Sarvam Bulbul (Indic)</option>
                </select>
              </div>

              {/* Translation Engine */}
              <div className="form-group">
                <label className="form-label flex items-center gap-1">
                  <Languages size={12} className="text-indigo-400" />
                  <span>Translator</span>
                </label>
                <select
                  className="form-select text-xs"
                  value={defaultTranslationEngine}
                  onChange={(e) => setDefaultTranslationEngine(e.target.value)}
                >
                  <option value="google_deep">Google / DeepTranslate</option>
                  <option value="sarvam_mayura">Sarvam Mayura v1</option>
                </select>
              </div>
            </div>
          </div>

          {/* SECTION 3: SARVAM AI API KEY */}
          <div className="form-group">
            <label className="form-label flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <Key size={14} className="text-indigo-400" />
                <span>Sarvam AI Subscription Key</span>
              </span>
              <span className="badge-pill bg-indigo-500/20 text-indigo-300">Optional</span>
            </label>
            <input
              type="password"
              className="form-input"
              placeholder="Enter Sarvam AI API Key from sarvam.ai"
              value={sarvamApiKey}
              onChange={(e) => setSarvamApiKey(e.target.value)}
            />
            <p className="form-hint">
              Required for Sarvam Saaras ASR, Mayura translation, and Bulbul voice cloning models.
            </p>
          </div>

          {/* SECTION 4: WHISPER SIZE & TARGET LANG */}
          <div className="grid grid-cols-2 gap-4">
            <div className="form-group">
              <label className="form-label flex items-center gap-1.5">
                <Cpu size={14} className="text-indigo-400" />
                <span>Whisper Model Size</span>
              </label>
              <select
                className="form-select"
                value={whisperModelSize}
                onChange={(e) => setWhisperModelSize(e.target.value)}
              >
                <option value="tiny">Tiny (Fastest, ~39M)</option>
                <option value="base">Base (Recommended, ~74M)</option>
                <option value="small">Small (High Accuracy, ~244M)</option>
                <option value="medium">Medium (Ultra Precision, ~769M)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label flex items-center gap-1.5">
                <Globe size={14} className="text-indigo-400" />
                <span>Default Dubbing Language</span>
              </label>
              <select
                className="form-select"
                value={defaultTargetLang}
                onChange={(e) => setDefaultTargetLang(e.target.value)}
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
          </div>

          {/* SECTION 5: SYSTEM DOCTOR DIAGNOSTICS */}
          <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 space-y-4">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Activity size={16} className="text-emerald-400" />
                <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider">System Doctor Diagnostics</h3>
              </div>
              <button 
                className="btn-secondary text-xs flex items-center gap-1.5 py-1 px-3"
                onClick={runDoctorCheck}
                disabled={isCheckingDoctor}
              >
                <RefreshCw size={12} className={isCheckingDoctor ? 'animate-spin' : ''} />
                <span>{isCheckingDoctor ? 'Checking...' : 'Re-Check'}</span>
              </button>
            </div>

            {doctorReport && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                  <div className="p-2.5 rounded-lg bg-gray-800 border border-gray-700 flex flex-col">
                    <span className="text-[11px] text-gray-400">Local Whisper ASR</span>
                    <span className="text-xs font-bold text-emerald-400 flex items-center gap-1 mt-0.5">
                      <CheckCircle2 size={12} /> Ready
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-gray-800 border border-gray-700 flex flex-col">
                    <span className="text-[11px] text-gray-400">Edge Neural TTS</span>
                    <span className="text-xs font-bold text-emerald-400 flex items-center gap-1 mt-0.5">
                      <CheckCircle2 size={12} /> Ready
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-gray-800 border border-gray-700 flex flex-col">
                    <span className="text-[11px] text-gray-400">Translators</span>
                    <span className="text-xs font-bold text-emerald-400 flex items-center gap-1 mt-0.5">
                      <CheckCircle2 size={12} /> Ready
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-gray-800 border border-gray-700 flex flex-col">
                    <span className="text-[11px] text-gray-400">Database & RAG</span>
                    <span className="text-xs font-bold text-emerald-400 flex items-center gap-1 mt-0.5">
                      <CheckCircle2 size={12} /> Ready
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-gray-800 border border-gray-700 flex flex-col">
                    <span className="text-[11px] text-gray-400">Whisper compute</span>
                    <span className={`text-xs font-bold flex items-center gap-1 mt-0.5 ${doctorReport.asr_runtime?.gpu_ready ? 'text-emerald-400' : 'text-amber-300'}`}>
                      {doctorReport.asr_runtime?.gpu_ready ? 'NVIDIA GPU ready' : `CPU fallback${doctorReport.asr_runtime?.gpu_available ? ' · GPU libraries missing' : ''}`}
                    </span>
                    {doctorReport.asr_runtime?.gpu_error && <span className="text-[10px] text-gray-400 mt-1 break-words">{doctorReport.asr_runtime.gpu_error}</span>}
                  </div>
                </div>

                {doctorReport.missing_count > 0 ? (
                  <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-between">
                    <div className="text-xs text-amber-300">
                      <span>Found {doctorReport.missing_count} missing package(s): {doctorReport.missing_packages.join(', ')}</span>
                    </div>
                    <button 
                      className="btn-primary text-xs flex items-center gap-1.5 py-1 px-3"
                      onClick={handleFixDoctor}
                      disabled={isFixingDoctor}
                    >
                      <Wrench size={12} />
                      <span>{isFixingDoctor ? 'Installing...' : 'Auto-Install & Fix'}</span>
                    </button>
                  </div>
                ) : (
                  <div className="text-xs text-emerald-400 flex items-center gap-1.5 p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
                    <CheckCircle2 size={14} />
                    <span>All Python & Audio libraries are installed, healthy, and up-to-date!</span>
                  </div>
                )}

                {fixMessage && (
                  <p className="text-xs text-indigo-300 font-mono">{fixMessage}</p>
                )}
              </div>
            )}
          </div>

          {statusMessage && (
            <div className={`p-3 rounded-lg flex items-center gap-2 text-xs font-medium ${statusMessage.includes('Error') ? 'bg-red-500/15 text-red-300 border border-red-500/30' : 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'}`}>
              {statusMessage.includes('Error') ? <AlertCircle size={16} /> : <Check size={16} />}
              <span>{statusMessage}</span>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>
            Close
          </button>
          <button className="btn-primary flex items-center gap-2" onClick={handleSave} disabled={isSaving}>
            <Save size={16} />
            <span>{isSaving ? 'Saving...' : 'Save Settings'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
