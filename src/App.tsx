import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Navbar, type TabType } from './components/Navbar';
import { DubbingStudio } from './components/DubbingStudio';
import { AudioRAG } from './components/AudioRAG';
import { SpeakerClassifier } from './components/SpeakerClassifier';
import { VoiceProfiles } from './components/VoiceProfiles';
import { WorkCapture } from './components/WorkCapture';
import { BrowserTranscriber } from './components/BrowserTranscriber';
import { SettingsModal } from './components/SettingsModal';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dubbing');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  return (
    <div className="app-layout min-h-screen">
      {/* Background Animated Blobs */}
      <div className="bg-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>

      <div className="app-wrapper">
        {/* Navigation Bar */}
        <Navbar 
          activeTab={activeTab} 
          setActiveTab={setActiveTab} 
          onOpenSettings={() => setIsSettingsOpen(true)} 
        />

        {/* Tab Views */}
        <main className="content-container">
          <AnimatePresence mode="wait">
            {activeTab === 'dubbing' && (
              <motion.div
                key="dubbing"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.25 }}
              >
                <DubbingStudio />
              </motion.div>
            )}

            {activeTab === 'rag' && (
              <motion.div
                key="rag"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.25 }}
              >
                <AudioRAG />
              </motion.div>
            )}

            {activeTab === 'speakers' && (
              <motion.div
                key="speakers"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.25 }}
              >
                <SpeakerClassifier />
              </motion.div>
            )}

            {activeTab === 'voice_profiles' && (
              <motion.div
                key="voice_profiles"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.25 }}
              >
                <VoiceProfiles />
              </motion.div>
            )}

            {activeTab === 'work_capture' && (
              <motion.div
                key="work_capture"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.25 }}
              >
                <WorkCapture />
              </motion.div>
            )}

            {activeTab === 'browser_transcribe' && (
              <motion.div
                key="browser_transcribe"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.25 }}
              >
                <BrowserTranscriber />
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* Settings Modal */}
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
      />
    </div>
  );
}

export default App;
