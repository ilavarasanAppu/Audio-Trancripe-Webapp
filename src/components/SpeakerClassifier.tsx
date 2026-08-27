import React, { useState, useEffect } from 'react';
import { 
  Users, UserCheck, Edit3, Check, Search, FileAudio, 
  RefreshCw 
} from 'lucide-react';
import { api, type Speaker } from '../services/api';

export const SpeakerClassifier: React.FC = () => {
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [editingSpeakerId, setEditingSpeakerId] = useState<number | null>(null);
  const [editName, setEditName] = useState('');
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedSpeakerId, setSelectedSpeakerId] = useState<number | null>(null);
  const [speakerFilesData, setSpeakerFilesData] = useState<{ files: any[]; segments: any[] } | null>(null);

  useEffect(() => {
    loadSpeakers();
  }, []);

  const loadSpeakers = async () => {
    try {
      const res = await api.getSpeakers();
      setSpeakers(res.speakers || []);
    } catch (err) {
      console.error('Error loading speakers:', err);
    }
  };

  const handleStartRename = (sp: Speaker) => {
    setEditingSpeakerId(sp.id);
    setEditName(sp.name || sp.display_label);
  };

  const handleSaveRename = async (speakerId: number) => {
    if (!editName.trim()) return;
    try {
      await api.assignSpeakerName(speakerId, editName.trim());
      setEditingSpeakerId(null);
      await loadSpeakers();
    } catch (err: any) {
      alert(`Error assigning name: ${err.message}`);
    }
  };

  const handleSelectSpeaker = async (speakerId: number) => {
    setSelectedSpeakerId(speakerId);
    try {
      const data = await api.getSpeakerFiles(speakerId);
      setSpeakerFilesData(data);
    } catch (err) {
      console.error(err);
    }
  };

  const filteredSpeakers = speakers.filter(s => 
    s.name.toLowerCase().includes(searchFilter.toLowerCase()) || 
    s.display_label.toLowerCase().includes(searchFilter.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Top Header Card */}
      <div className="professional-card">
        <div className="card-header-traditional">
          <div>
            <h2 className="card-title">
              <Users className="text-indigo-400" size={20} />
              <span>Voice Diarization & Speaker Classifier</span>
            </h2>
            <p className="card-subtitle">
              Acoustic voice clustering groups speech into individual speakers so you can assign human names and filter files by person.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button className="btn-secondary text-xs" onClick={loadSpeakers}>
              <RefreshCw size={14} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Search Bar */}
        <div className="relative mb-6">
          <Search className="absolute left-3 top-3 text-gray-400" size={16} />
          <input 
            type="text" 
            className="form-input pl-9 text-xs"
            placeholder="Search by assigned speaker name or voice label..."
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
          />
        </div>

        {/* Speaker Profiles Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {filteredSpeakers.length === 0 ? (
            <div className="col-span-3 text-center py-8 text-gray-500 text-xs">
              No speakers detected yet. Index audio or video files in Tab 2 (Audio RAG) first.
            </div>
          ) : (
            filteredSpeakers.map((sp) => (
              <div 
                key={sp.id}
                className={`p-4 rounded-xl border transition-all cursor-pointer ${
                  selectedSpeakerId === sp.id 
                    ? 'bg-indigo-950/30 border-indigo-500/60 shadow-md' 
                    : 'bg-gray-900 border-gray-800 hover:border-gray-700'
                }`}
                onClick={() => handleSelectSpeaker(sp.id)}
              >
                <div className="flex justify-between items-start mb-3">
                  <div className="w-10 h-10 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold">
                    <UserCheck size={20} />
                  </div>
                  <span className="badge-pill bg-gray-800 text-gray-300 font-mono text-xs">
                    {sp.segment_count} segments
                  </span>
                </div>

                {editingSpeakerId === sp.id ? (
                  <div className="flex gap-1.5 mb-2" onClick={(e) => e.stopPropagation()}>
                    <input 
                      type="text" 
                      className="form-input text-xs py-1"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      autoFocus
                    />
                    <button 
                      className="btn-primary text-xs py-1 px-2.5"
                      onClick={() => handleSaveRename(sp.id)}
                    >
                      <Check size={14} />
                    </button>
                  </div>
                ) : (
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-bold text-sm text-white truncate max-w-[170px]">
                      {sp.name}
                    </h4>
                    <button 
                      className="btn-icon w-6 h-6 text-gray-400 hover:text-indigo-400"
                      onClick={(e) => { e.stopPropagation(); handleStartRename(sp); }}
                      title="Rename Speaker"
                    >
                      <Edit3 size={12} />
                    </button>
                  </div>
                )}

                <p className="text-xs text-gray-400 italic line-clamp-2 mb-3">
                  "{sp.sample_text || 'Sample speech segment'}"
                </p>

                {sp.sample_file_path && (
                  <div onClick={(e) => e.stopPropagation()}>
                    <audio 
                      controls 
                      src={api.getClipUrl(sp.sample_file_path, sp.sample_start || 0, sp.sample_end || 5)}
                      className="h-6 w-full"
                    />
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Selected Speaker Files & Segments */}
      {speakerFilesData && (
        <div className="professional-card">
          <div className="card-header-traditional">
            <h3 className="card-title">
              <span>All Files & Spoken Segments by {speakers.find(s => s.id === selectedSpeakerId)?.name}</span>
            </h3>
            <span className="badge-pill bg-indigo-500/20 text-indigo-300">
              {speakerFilesData.files.length} Files • {speakerFilesData.segments.length} Segments
            </span>
          </div>

          <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
            {speakerFilesData.segments.map((seg) => (
              <div key={seg.id} className="p-3 rounded-lg bg-gray-900 border border-gray-800 flex justify-between items-center gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 text-xs text-indigo-400 font-semibold mb-1">
                    <FileAudio size={14} />
                    <span>{seg.filename}</span>
                    <span className="font-mono text-gray-500">[{seg.start_time}s - {seg.end_time}s]</span>
                  </div>
                  <p className="text-xs text-gray-300 italic">"{seg.text}"</p>
                </div>

                <audio 
                  controls 
                  src={api.getClipUrl(seg.file_path, seg.start_time, seg.end_time)}
                  className="h-7 max-w-[180px]"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
