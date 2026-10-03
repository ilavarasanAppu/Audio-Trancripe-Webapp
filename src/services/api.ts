export interface AudioFile {
  id: number;
  file_path: string;
  filename: string;
  file_size: number;
  duration: number;
  file_format: string;
  status: 'pending' | 'processing' | 'indexed' | 'error';
  language?: string;
  created_at?: string;
  chunk_count?: number;
}

export interface TranscriptChunk {
  id: number;
  file_id: number;
  chunk_index: number;
  start_time: number;
  end_time: number;
  text: string;
  language: string;
  speaker_id?: number;
  speaker_name?: string;
}

export interface Speaker {
  id: number;
  name: string;
  display_label: string;
  created_at: string;
  segment_count: number;
  sample_file?: string;
  sample_file_path?: string;
  sample_start?: number;
  sample_end?: number;
  sample_text?: string;
}

export interface RAGCitation {
  chunk_id: number;
  file_id: number;
  filename: string;
  file_path: string;
  start_time: number;
  end_time: number;
  text: string;
  language: string;
  speaker_name: string;
  similarity_score: number;
}

export interface RAGQueryResult {
  question: string;
  answer: string;
  citations: RAGCitation[];
}

export interface DubbingSegment {
  id: number;
  project_id: number;
  segment_index: number;
  start_time: number;
  end_time: number;
  original_text: string;
  translated_text?: string;
  voice_name?: string;
  audio_path?: string;
  speaker_tag?: string;
  pitch_shift: number;
  speed_rate: number;
}

export interface DubbingProject {
  id: number;
  title: string;
  source_file_path: string;
  source_lang: string;
  target_lang: string;
  engine: 'sarvam' | 'neural' | string;
  status: 'created' | 'processing' | 'completed' | 'error';
  created_at: string;
  result_audio_path?: string;
  result_video_path?: string;
  segment_count?: number;
}

export interface VoiceOption {
  id: string;
  name: string;
  gender: 'male' | 'female';
}

export interface LanguageVoices {
  [code: string]: {
    name: string;
    code: string;
    sarvam_code: string;
    voices: VoiceOption[];
    default_voice: string;
  };
}

export interface CustomModelItem {
  name: string;
  path: string;
  folder_origin?: string;
  is_directory: boolean;
  format: string;
  extension: string;
  size_mb: number;
  model_type: string;
  asr_compatible?: boolean;
}

export interface DoctorReport {
  healthy: boolean;
  missing_count: number;
  missing_packages: string[];
  packages: Array<{ package: string; status: string; version?: string; installed: boolean; error?: string }>;
  ffmpeg: { installed: boolean; path: string };
  asr_engines: { faster_whisper: boolean; sarvam_saaras: boolean; browser_wasm: boolean };
  asr_runtime?: { device: string; compute_type: string; gpu_available: boolean; gpu_ready?: boolean; gpu_count: number; gpu_error?: string };
  tts_engines: { edge_neural: boolean; sarvam_bulbul: boolean; deep_translator: boolean };
  python: { python_version: string; executable: string; platform: string };
}

export const API_BASE = '/api';

export const api = {
  // Health & Settings
  checkHealth: async () => {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  },
  getDoctor: async (): Promise<DoctorReport> => {
    const res = await fetch(`${API_BASE}/doctor`);
    return res.json();
  },
  fixDoctor: async () => {
    const res = await fetch(`${API_BASE}/doctor/fix`, { method: 'POST' });
    return res.json();
  },
  getSettings: async () => {
    const res = await fetch(`${API_BASE}/settings`);
    return res.json();
  },
  updateSettings: async (settings: { 
    sarvam_api_key?: string; 
    whisper_model_size?: string; 
    default_target_lang?: string;
    default_asr_engine?: string;
    default_tts_engine?: string;
    default_translation_engine?: string;
    custom_models_dir?: string;
    custom_models_dirs?: string[];
    selected_custom_model?: string;
  }) => {
    const res = await fetch(`${API_BASE}/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    });
    return res.json();
  },
  scanCustomModels: async (directories: string[] | string): Promise<{ directories: string[]; total_models: number; models: CustomModelItem[] }> => {
    const dirs = Array.isArray(directories) ? directories : [directories];
    const res = await fetch(`${API_BASE}/models/scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ directories: dirs })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Model scan failed');
    }
    return res.json();
  },
  getCustomModels: async (): Promise<{ custom_models_dirs: string[]; custom_models_dir?: string; selected_custom_model: string; total_models: number; models: CustomModelItem[] }> => {
    const res = await fetch(`${API_BASE}/models`);
    return res.json();
  },
  getVoices: async () => {
    const res = await fetch(`${API_BASE}/dubbing/voices`);
    return res.json();
  },

  // Directory Scanner & Files
  scanDirectory: async (directory: string, recursive: boolean = true) => {
    const res = await fetch(`${API_BASE}/scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ directory, recursive })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Scan failed');
    }
    return res.json();
  },
  getFiles: async (): Promise<{ files: AudioFile[] }> => {
    const res = await fetch(`${API_BASE}/files`);
    return res.json();
  },
  deleteFile: async (fileId: number) => {
    const res = await fetch(`${API_BASE}/files/${fileId}`, { method: 'DELETE' });
    return res.json();
  },
  uploadFile: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData
    });
    return res.json();
  },
  transcribeAudio: async (file: File, language: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', language);
    const res = await fetch(`${API_BASE}/transcribe`, { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Transcription failed');
    return data;
  },
  ingestFile: async (fileId: number, modelSize: string = 'base', language: string = 'auto', asrEngine: string = 'faster_whisper', customModelPath?: string) => {
    const res = await fetch(`${API_BASE}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: fileId, model_size: modelSize, language, asr_engine: asrEngine, custom_model_path: customModelPath })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Ingestion failed');
    }
    return res.json();
  },
  ingestAll: async () => {
    const res = await fetch(`${API_BASE}/ingest_all`, { method: 'POST' });
    return res.json();
  },

  // RAG Query
  queryRAG: async (question: string, topK: number = 6): Promise<RAGQueryResult> => {
    const res = await fetch(`${API_BASE}/rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: topK })
    });
    return res.json();
  },

  // Audio Stream & Clip URLs
  getStreamUrl: (filePath: string) => {
    return `${API_BASE}/media/stream?file_path=${encodeURIComponent(filePath)}`;
  },
  getClipUrl: (filePath: string, start: number, end: number) => {
    return `${API_BASE}/media/clip?file_path=${encodeURIComponent(filePath)}&start=${start}&end=${end}`;
  },

  // Work Capture & File Copy
  copyFiles: async (payload: { file_ids: number[]; target_directory: string; export_as_clips?: boolean; chunk_ids?: number[] }) => {
    const res = await fetch(`${API_BASE}/work_capture/copy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Copy failed');
    }
    return res.json();
  },

  // Speakers
  getSpeakers: async (): Promise<{ speakers: Speaker[] }> => {
    const res = await fetch(`${API_BASE}/speakers`);
    return res.json();
  },
  assignSpeakerName: async (speakerId: number, name: string) => {
    const res = await fetch(`${API_BASE}/speakers/assign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speaker_id: speakerId, name })
    });
    return res.json();
  },
  getSpeakerFiles: async (speakerId: number) => {
    const res = await fetch(`${API_BASE}/speakers/${speakerId}/files`);
    return res.json();
  },

  // Dubbing
  createDubbingProject: async (payload: { 
    file_id: number; 
    title?: string; 
    source_lang?: string; 
    target_lang?: string; 
    engine?: string; 
    tts_engine?: string;
    asr_engine?: string;
    custom_model_path?: string;
    translation_engine?: string;
    voice_name?: string 
  }) => {
    const res = await fetch(`${API_BASE}/dubbing/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Creation failed');
    }
    return res.json();
  },
  getDubbingProjects: async (): Promise<{ projects: DubbingProject[] }> => {
    const res = await fetch(`${API_BASE}/dubbing/projects`);
    return res.json();
  },
  getDubbingProject: async (projectId: number): Promise<{ project: DubbingProject; segments: DubbingSegment[] }> => {
    const res = await fetch(`${API_BASE}/dubbing/projects/${projectId}`);
    return res.json();
  },
  processDubbing: async (projectId: number, sarvamApiKey?: string, ttsEngine?: string) => {
    const res = await fetch(`${API_BASE}/dubbing/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: projectId, sarvam_api_key: sarvamApiKey, tts_engine: ttsEngine })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Dubbing process failed');
    }
    return res.json();
  },
  updateDubbingSegment: async (payload: { segment_id: number; translated_text: string; voice_name?: string; pitch_shift?: number; speed_rate?: number }) => {
    const res = await fetch(`${API_BASE}/dubbing/segment/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    return res.json();
  },
  dubLiveChunk: async (payload: { audio_blob: Blob; source_lang?: string; target_lang?: string; tts_engine?: string; voice_name?: string; asr_engine?: string }) => {
    const formData = new FormData();
    formData.append('file', payload.audio_blob, 'chunk.wav');
    formData.append('source_lang', payload.source_lang || 'auto');
    formData.append('target_lang', payload.target_lang || 'ta');
    formData.append('tts_engine', payload.tts_engine || 'edge_neural');
    formData.append('asr_engine', payload.asr_engine || 'faster_whisper');
    if (payload.voice_name) {
      formData.append('voice_name', payload.voice_name);
    }

    const res = await fetch(`${API_BASE}/dubbing/live_chunk`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Live chunk dubbing failed');
    }
    return res.json();
  },
  batchFolderDubbing: async (payload: { source_directory: string; target_lang: string; source_lang?: string; tts_engine?: string; asr_engine?: string; output_directory?: string }) => {
    const res = await fetch(`${API_BASE}/dubbing/batch_folder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Batch folder dubbing failed');
    }
    return res.json();
  }
};
