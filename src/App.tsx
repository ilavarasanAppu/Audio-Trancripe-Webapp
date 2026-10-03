import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Navbar, type TabType } from './components/Navbar';
import { DubbingStudio } from './components/DubbingStudio';
import { AudioRAG } from './components/AudioRAG';
import { SpeakerClassifier } from './components/SpeakerClassifier';
import { WorkCapture } from './components/WorkCapture';
import { BrowserTranscriber } from './components/BrowserTranscriber';
import { SettingsModal } from './components/SettingsModal';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dubbing');
  const [visitedTabs, setVisitedTabs] = useState<Set<TabType>>(() => new Set(['dubbing']));
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const handleTabChange = (tab: TabType) => {
    setVisitedTabs((visited) => visited.has(tab) ? visited : new Set(visited).add(tab));
    setActiveTab(tab);
  };

  useEffect(() => {
    const reportClientError = (error: Record<string, unknown>) => {
      void fetch('/api/errors/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...error,
          page_url: window.location.href,
          user_agent: navigator.userAgent,
          occurred_at: new Date().toISOString(),
        }),
      }).catch(() => {
        // Error reporting must never create another unhandled browser error.
      });
    };

    const onError = (event: ErrorEvent) => {
      if (event.filename?.startsWith('chrome-extension://') || event.filename?.startsWith('moz-extension://')) return;
      reportClientError({
      error_code: 'BROWSER_UNCAUGHT_ERROR',
      error_details: event.message || 'Uncaught browser error',
      source_file: event.filename,
      error_line: event.lineno,
      error_column: event.colno,
      stack: event.error instanceof Error ? event.error.stack : undefined,
      });
    };
    const onRejection = (event: PromiseRejectionEvent) => {
      const stack = event.reason instanceof Error ? event.reason.stack || '' : '';
      if (stack.includes('chrome-extension://') || stack.includes('moz-extension://')) return;
      reportClientError({
      error_code: 'BROWSER_UNHANDLED_REJECTION',
      error_details: event.reason instanceof Error ? event.reason.message : String(event.reason),
      stack: event.reason instanceof Error ? event.reason.stack : undefined,
      });
    };

    window.addEventListener('error', onError);
    window.addEventListener('unhandledrejection', onRejection);
    return () => {
      window.removeEventListener('error', onError);
      window.removeEventListener('unhandledrejection', onRejection);
    };
  }, []);

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
          setActiveTab={handleTabChange}
          onOpenSettings={() => setIsSettingsOpen(true)} 
        />

        {/* Tab Views */}
        <main className="content-container">
          {/* Keep each workspace mounted so tab changes preserve its selections and results. */}
          <motion.div style={{ display: activeTab === 'dubbing' ? undefined : 'none' }}>
            {visitedTabs.has('dubbing') && <DubbingStudio />}
          </motion.div>
          <motion.div style={{ display: activeTab === 'rag' ? undefined : 'none' }}>
            {visitedTabs.has('rag') && <AudioRAG />}
          </motion.div>
          <motion.div style={{ display: activeTab === 'speakers' ? undefined : 'none' }}>
            {visitedTabs.has('speakers') && <SpeakerClassifier active={activeTab === 'speakers'} />}
          </motion.div>
          <motion.div style={{ display: activeTab === 'work_capture' ? undefined : 'none' }}>
            {visitedTabs.has('work_capture') && <WorkCapture />}
          </motion.div>
          <motion.div style={{ display: activeTab === 'browser_transcribe' ? undefined : 'none' }}>
            {visitedTabs.has('browser_transcribe') && <BrowserTranscriber />}
          </motion.div>
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
