import React, { useEffect, useState } from 'react';
import { 
  Mic2, Search, Users, FolderOutput, Sparkles, 
  Settings as SettingsIcon, Cpu 
} from 'lucide-react';
import { api } from '../services/api';

export type TabType = 'dubbing' | 'rag' | 'speakers' | 'work_capture' | 'browser_transcribe';

interface NavbarProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  onOpenSettings: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, onOpenSettings }) => {
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(false);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await api.checkHealth();
        setIsBackendOnline(res?.status === 'online');
      } catch (err) {
        setIsBackendOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 8000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'dubbing', label: '1. AI Dubbing Studio', icon: Mic2, desc: 'Realtime & YouTube Style' },
    { id: 'rag', label: '2. Audio Q&A & Search', icon: Search, desc: 'Semantic Vector Retrieval' },
    { id: 'speakers', label: '3. Voice Classifier', icon: Users, desc: 'Group By Person' },
    { id: 'work_capture', label: '4. Work Capture', icon: FolderOutput, desc: 'Batch Copy & Export' },
    { id: 'browser_transcribe', label: '5. Fast Transcriber', icon: Sparkles, desc: 'Local Whisper · GPU when available' },
  ];

  return (
    <header className="navbar-container">
      {/* Brand Logo & Info */}
      <div className="navbar-brand">
        <div className="brand-icon-wrapper">
          <Cpu size={22} />
        </div>
        <div className="brand-info">
          <h1>VaniScript AI</h1>
          <p>Indic Speech Dubbing & Audio File Intelligence</p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="nav-tabs-wrapper">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id as TabType)}
              className={`nav-tab-item ${isActive ? 'active' : ''}`}
              title={item.desc}
            >
              <Icon size={16} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Right Actions: System Doctor & Settings */}
      <div className="nav-actions">
        <div 
          className={`status-pill ${isBackendOnline ? 'online' : 'offline'}`}
          title={isBackendOnline ? "AI Server Connected on Port 8000" : "Server Disconnected"}
        >
          <span className="pulse-dot"></span>
          <span>{isBackendOnline ? 'AI Core Ready' : 'Connecting...'}</span>
        </div>

        <button 
          onClick={onOpenSettings} 
          className="btn-icon" 
          title="System Doctor, ASR/TTS Engine Selectors & Settings"
        >
          <SettingsIcon size={18} />
        </button>
      </div>
    </header>
  );
};
