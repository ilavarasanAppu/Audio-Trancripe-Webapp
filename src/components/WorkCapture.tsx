import React, { useState, useEffect } from 'react';
import { 
  FolderOutput, CheckSquare, Square, Copy, CheckCircle2, 
  Loader2, FileAudio, Folder, RefreshCw 
} from 'lucide-react';
import { api, type AudioFile } from '../services/api';

export const WorkCapture: React.FC = () => {
  const [files, setFiles] = useState<AudioFile[]>([]);
  const [selectedFileIds, setSelectedFileIds] = useState<number[]>([]);
  const [targetDir, setTargetDir] = useState('c:\\Users\\ela\\Downloads\\CapturedAudio');
  const [isExporting, setIsExporting] = useState(false);
  const [exportResult, setExportResult] = useState<any | null>(null);

  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      const data = await api.getFiles();
      setFiles(data.files || []);
      if (data.files) {
        setSelectedFileIds(data.files.map(f => f.id));
      }
    } catch (err) {
      console.error(err);
    }
  };

  const toggleSelectAll = () => {
    if (selectedFileIds.length === files.length) {
      setSelectedFileIds([]);
    } else {
      setSelectedFileIds(files.map(f => f.id));
    }
  };

  const toggleSelectFile = (id: number) => {
    if (selectedFileIds.includes(id)) {
      setSelectedFileIds(selectedFileIds.filter(i => i !== id));
    } else {
      setSelectedFileIds([...selectedFileIds, id]);
    }
  };

  const handleCopyFiles = async () => {
    if (!targetDir.trim() || selectedFileIds.length === 0) return;
    setIsExporting(true);
    setExportResult(null);

    try {
      const result = await api.copyFiles({
        file_ids: selectedFileIds,
        target_directory: targetDir.trim(),
        export_as_clips: false
      });
      setExportResult(result);
    } catch (err: any) {
      alert(`Export failed: ${err.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="professional-card">
        <div className="card-header-traditional">
          <div>
            <h2 className="card-title">
              <FolderOutput className="text-indigo-400" size={20} />
              <span>Batch Work Capture & File Organizer</span>
            </h2>
            <p className="card-subtitle">
              Batch select indexed audio/video files and copy them directly to any target folder on your Windows computer.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button className="btn-secondary text-xs" onClick={loadFiles}>
              <RefreshCw size={14} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Destination Path Input */}
        <div className="p-4 rounded-xl bg-gray-900 border border-gray-800 mb-6 space-y-4">
          <div className="form-group">
            <label className="form-label flex items-center gap-1.5">
              <Folder size={14} className="text-indigo-400" />
              <span>Target Destination Folder (Windows)</span>
            </label>
            <input 
              type="text" 
              className="form-input text-xs font-mono"
              placeholder="e.g. C:\Users\ela\Desktop\MyWorkAudio"
              value={targetDir}
              onChange={(e) => setTargetDir(e.target.value)}
            />
            <p className="form-hint">The system will create the destination folder automatically if it doesn't exist.</p>
          </div>

          <div className="flex flex-col md:flex-row justify-between items-center gap-3 pt-2">
            <div className="flex items-center gap-2">
              <button 
                type="button" 
                className="btn-secondary text-xs flex items-center gap-1.5 py-1 px-3"
                onClick={toggleSelectAll}
              >
                {selectedFileIds.length === files.length ? <CheckSquare size={14} className="text-indigo-400" /> : <Square size={14} />}
                <span>{selectedFileIds.length === files.length ? 'Deselect All' : 'Select All'}</span>
              </button>
              <span className="text-xs text-gray-400 font-mono">
                {selectedFileIds.length} of {files.length} selected
              </span>
            </div>

            <button 
              className="btn-primary text-xs flex items-center gap-2 px-6 py-2"
              onClick={handleCopyFiles}
              disabled={isExporting || selectedFileIds.length === 0}
            >
              {isExporting ? <Loader2 className="animate-spin" size={14} /> : <Copy size={14} />}
              <span>{isExporting ? 'Copying Files...' : 'Batch Copy Selected Files'}</span>
            </button>
          </div>
        </div>

        {/* Selection Table */}
        <div className="table-container max-h-72 overflow-y-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '40px' }}>Select</th>
                <th>File Name</th>
                <th>File Path</th>
                <th>Duration</th>
                <th>Format</th>
              </tr>
            </thead>
            <tbody>
              {files.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-6 text-gray-500 text-xs">
                    No files available in library.
                  </td>
                </tr>
              ) : (
                files.map((file) => {
                  const isSelected = selectedFileIds.includes(file.id);
                  return (
                    <tr 
                      key={file.id} 
                      className={`cursor-pointer ${isSelected ? 'bg-indigo-950/20' : ''}`}
                      onClick={() => toggleSelectFile(file.id)}
                    >
                      <td>
                        {isSelected ? <CheckSquare size={16} className="text-indigo-400" /> : <Square size={16} className="text-gray-500" />}
                      </td>
                      <td className="font-semibold text-xs flex items-center gap-2">
                        <FileAudio size={14} className="text-indigo-400 shrink-0" />
                        <span className="truncate max-w-[220px]">{file.filename}</span>
                      </td>
                      <td className="text-xs font-mono text-gray-400 truncate max-w-[320px]">
                        {file.file_path}
                      </td>
                      <td className="font-mono text-xs text-gray-400">
                        {file.duration ? `${Math.round(file.duration)}s` : '-'}
                      </td>
                      <td>
                        <span className="badge-pill bg-gray-800 text-gray-300 uppercase text-[10px] font-mono">
                          {file.file_format}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Export Results */}
        {exportResult && (
          <div className="mt-6 p-4 rounded-xl bg-gray-900 border border-emerald-500/30">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm mb-2">
              <CheckCircle2 size={18} />
              <span>Successfully copied {exportResult.total_copied} item(s) to destination!</span>
            </div>
            <p className="text-xs text-gray-400 font-mono mb-2">
              Destination: {exportResult.target_directory}
            </p>
            <div className="space-y-1.5 max-h-36 overflow-y-auto">
              {exportResult.copied_items?.map((item: any, idx: number) => (
                <div key={idx} className="text-xs text-gray-300 p-1.5 rounded bg-gray-800 flex items-center gap-2">
                  <FileAudio size={12} className="text-indigo-400" />
                  <span className="truncate">{item.destination}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
