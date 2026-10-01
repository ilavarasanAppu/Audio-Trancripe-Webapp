import React, { useState, useEffect, useRef } from 'react';
import { 
  Users, UserCheck, Edit3, Check, Search, FileAudio, 
  RefreshCw, Mic, User, Users as UsersIcon, Play, Pause
} from 'lucide-react';
import { api, type Speaker } from '../services/api';

interface SpeakerWithGender extends Speaker {
  gender: 'male' | 'female' | 'unknown' | 'group';
}

export const VoiceProfiles: React.FC = () => {
  const [speakers, setSpeakers] = useState<SpeakerWithGender[]>([]);
  const [editingSpeakerId, setEditingSpeakerId] = useState<number | null>(null);
  const [editName, setEditName] = useState('');
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedSpeakerId, setSelectedSpeakerId] = useState<number | null>(null);
  const [speakerFilesData, setSpeakerFilesData] = useState<{ files: any[]; segments: any[] } | null>(null);
  const [genderFilter, setGenderFilter] = useState<'all' | 'male' | 'female' | 'unknown' | 'group'>('all');
  const [isPlaying, setIsPlaying] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    loadSpeakers();
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  const loadSpeakers = async () => {
    try {
      const res = await api.getSpeakers();
      const speakersWithGender: SpeakerWithGender[] = (res.speakers || []).map(s => ({
        ...s,
        gender: s.gender || 'unknown'
      }));
      setSpeakers(speakersWithGender);
    } catch (err) {
      console.error('Error loading speakers:', err);
    }
  };

  const handleStartRename = (sp: SpeakerWithGender) => {
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
    stopAudio();
    try {
      const data = await api.getSpeakerFiles(speakerId);
      setSpeakerFilesData(data);
    } catch (err) {
      console.error(err);
    }
  };

  const playReferenceAudio = (speakerId: number) => {
    if (isPlaying === speakerId) {
      stopAudio();
      return;
    }
    
    stopAudio();
    setIsPlaying(speakerId);
    
    audioRef.current = new Audio(api.getSpeakerReferenceAudio(speakerId));
    audioRef.current.onended = () => setIsPlaying(null);
    audioRef.current.onerror = () => setIsPlaying(null);
    audioRef.current.play().catch(console.error);
  };

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setIsPlaying(null);
  };

  const getGenderIcon = (gender: string) => {
    switch (gender) {
      case 'male': return <Mic className="text-blue-400" size={14} />;
      case 'female': return <User className="text-pink-400" size={14} />;
      case 'group': return <UsersIcon className="text-amber-400" size={14} />;
      default: return <Mic className="text-gray-400" size={14} />;
    }
  };

  const getGenderLabel = (gender: string) => {
    switch (gender) {
      case 'male': return 'Male';
      case 'female': return 'Female';
      case 'group': return 'Group';
      default: return 'Unknown';
    }
  };

  const getGenderColor = (gender: string) => {
    switch (gender) {
      case 'male': return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      case 'female': return 'bg-pink-500/20 text-pink-300 border-pink-500/30';
      case 'group': return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
    }
  };

  const filteredSpeakers = speakers.filter(s => {
    const matchesSearch = s.name.toLowerCase().includes(searchFilter.toLowerCase()) || 
      s.display_label.toLowerCase().includes(searchFilter.toLowerCase());
    const matchesGender = genderFilter === 'all' || s.gender === genderFilter;
    return matchesSearch && matchesGender;
  });

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <div className="professional-card">
        <div className="card-header-traditional">
          <div>
            <h2 className="card-title">
              <Users className="text-indigo-400" size={20} />
              <span>Voice Profiles & Speaker Identity</span>
            </h2>
            <p className="card-subtitle">
              Manage speaker identities across all indexed media. Voices are automatically classified by gender and matched across files.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button className="btn-secondary text-xs" onClick={loadSpeakers}>
              <RefreshCw size={14} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Stats Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
            <p className="text-xs text-gray-400">Total Speakers</p>
            <p className="text-2xl font-bold text-white">{speakers.length}</p>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
            <p className="text-xs text-gray-400">Male Voices</p>
            <p className="text-2xl font-bold text-blue-400">{speakers.filter(s => s.gender === 'male').length}</p>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
            <p className="text-xs text-gray-400">Female Voices</p>
            <p className="text-2xl font-bold text-pink-400">{speakers.filter(s => s.gender === 'female').length}</p>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
            <p className="text-xs text-gray-400">Total Segments</p>
            <p className="text-2xl font-bold text-indigo-400">{speakers.reduce((acc, s) => acc + s.segment_count, 0)}</p>
          </div>
        </div>

        {/* Search & Filter Bar */}
        <div className="flex flex-wrap items-center gap-3 mb-6">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-3 text-gray-400" size={16} />
            <input 
              type="text" 
              className="form-input pl-9 text-xs"
              placeholder="Search by speaker name or voice label..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400">Gender:</label>
            <select 
              className="form-select text-xs w-auto"
              value={genderFilter}
              onChange={(e) => setGenderFilter(e.target.value as any)}
            >
              <option value="all">All</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="unknown">Unknown</option>
              <option value="group">Group</option>
            </select>
          </div>
        </div>

        {/* Speaker Profiles Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredSpeakers.length === 0 ? (
            <div className="col-span-3 text-center py-12 text-gray-500 text-xs">
              <Users className="mx-auto text-gray-600 mb-2" size={48} />
              <p>No speakers detected yet.</p>
              <p className="text-xs mt-1">Index audio or video files in the Audio RAG tab first.</p>
            </div>
          ) : (
            filteredSpeakers.map((sp) => (
              <div 
                key={sp.id}
                className={`p-4 rounded-xl border transition-all cursor-pointer relative overflow-hidden ${
                  selectedSpeakerId === sp.id 
                    ? 'bg-indigo-950/30 border-indigo-500/60 shadow-lg ring-1 ring-indigo-500/20' 
                    : 'bg-gray-900 border-gray-800 hover:border-gray-700 hover:bg-gray-800/50'
                }`}
                onClick={() => handleSelectSpeaker(sp.id)}
              >
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 to-purple-500 opacity-0 transition-opacity" 
                     style={{ opacity: selectedSpeakerId === sp.id ? 1 : 0 }} />
                
                {/* Header */}
                <div className="flex justify-between items-start mb-3">
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-indigo-600/20 to-purple-600/20 text-indigo-400 flex items-center justify-center font-bold border border-indigo-500/30">
                    <UserCheck size={24} />
                  </div>
                  <span className="badge-pill bg-gray-800 text-gray-300 font-mono text-xs">
                    {sp.segment_count} segments
                  </span>
                </div>

                {/* Name & Gender */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    {editingSpeakerId === sp.id ? (
                      <input 
                        type="text" 
                        className="form-input text-sm font-bold bg-transparent border-none text-white focus:outline-none"
                        value={editName}
                        onChange={(e) => { e.stopPropagation(); setEditName(e.target.value); }}
                        autoFocus
                        onBlur={(e) => { e.stopPropagation(); handleSaveRename(sp.id); }}
                        onKeyDown={(e) => { if (e.key === 'Enter') { e.stopPropagation(); handleSaveRename(sp.id); }}}
                      />
                    ) : (
                      <h4 className="font-bold text-sm text-white truncate">
                        {sp.name}
                      </h4>
                    )}
                    <span className={`badge-pill text-[10px] font-medium ${getGenderColor(sp.gender)}`}>
                      {getGenderIcon(sp.gender)}
                      {getGenderLabel(sp.gender)}
                    </span>
                  </div>
                  
                  {editingSpeakerId !== sp.id && (
                    <div className="flex items-center gap-1">
                      <button 
                        className="btn-icon w-7 h-7 text-gray-400 hover:text-indigo-400"
                        onClick={(e) => { e.stopPropagation(); handleStartRename(sp); }}
                        title="Rename Speaker"
                      >
                        <Edit3 size={14} />
                      </button>
                    </div>
                  )}
                </div>

                {editingSpeakerId === sp.id ? (
                  <button 
                    className="btn-primary text-xs w-full"
                    onClick={(e) => { e.stopPropagation(); handleSaveRename(sp.id); }}
                  >
                    <Check size={14} className="mr-1" />
                    Save Name
                  </button>
                ) : (
                  <>
                    <p className="text-xs text-gray-400 italic line-clamp-2 mb-3">
                      "{sp.sample_text || 'Sample speech segment'}"
                    </p>

                    {/* Reference Audio Player */}
                    <div className="flex items-center gap-2 p-2 rounded-lg bg-gray-800/50 border border-gray-700">
                      <button
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                          isPlaying === sp.id 
                            ? 'bg-indigo-600 text-white' 
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                        onClick={(e) => { e.stopPropagation(); playReferenceAudio(sp.id); }}
                      >
                        {isPlaying === sp.id ? (
                          <Pause size={14} />
                        ) : (
                          <Play size={14} />
                        )}
                        <span>{isPlaying === sp.id ? 'Playing...' : 'Play Voice Sample'}</span>
                      </button>
                    </div>
                  </>
                )}

                {/* Files count badge */}
                <div className="absolute bottom-3 right-3 text-xs text-gray-500 bg-gray-900/80 px-2 py-1 rounded backdrop-blur-sm">
                  {sp.segment_count} segments
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Selected Speaker Details */}
      {speakerFilesData && (
        <div className="professional-card">
          <div className="card-header-traditional">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center">
                <UserCheck size={20} />
              </div>
              <div>
                <h3 className="card-title">
                  {speakers.find(s => s.id === selectedSpeakerId)?.name}
                </h3>
                <p className="card-subtitle">
                  Voice profile details and all occurrences across media library
                </p>
              </div>
            </div>
            <span className="badge-pill bg-indigo-500/20 text-indigo-300">
              {speakerFilesData.files.length} Files • {speakerFilesData.segments.length} Segments
            </span>
          </div>

          <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
            {speakerFilesData.segments.map((seg) => (
              <div key={seg.id} className="p-3 rounded-lg bg-gray-900 border border-gray-800 flex justify-between items-center gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 text-xs text-indigo-400 font-semibold mb-1">
                    <FileAudio size={14} />
                    <span className="truncate max-w-[300px]">{seg.filename}</span>
                    <span className="font-mono text-gray-500">[{seg.start_time.toFixed(1)}s - {seg.end_time.toFixed(1)}s]</span>
                  </div>
                  <p className="text-xs text-gray-300 italic">"{seg.text}"</p>
                </div>

                <audio 
                  controls 
                  src={api.getClipUrl(seg.file_path, seg.start_time, seg.end_time)}
                  className="h-7 max-w-[200px]"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};