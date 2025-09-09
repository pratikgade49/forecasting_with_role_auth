import React, { useState } from 'react';
import { X, Download, Terminal, Clock, CheckCircle, AlertCircle, Info } from 'lucide-react';

interface ProcessLogViewerProps {
  isOpen: boolean;
  onClose: () => void;
  processLog: string[];
  title?: string;
}

export const ProcessLogViewer: React.FC<ProcessLogViewerProps> = ({
  isOpen,
  onClose,
  processLog,
  title = "Process Log"
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  const downloadLog = () => {
    const logContent = processLog.join('\n');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `forecast_process_log_${timestamp}.txt`;
    
    const blob = new Blob([logContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const getLogEntryIcon = (entry: string) => {
    const lowerEntry = entry.toLowerCase();
    if (lowerEntry.includes('error') || lowerEntry.includes('failed')) {
      return <AlertCircle className="w-4 h-4 text-red-500" />;
    } else if (lowerEntry.includes('success') || lowerEntry.includes('completed')) {
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    } else if (lowerEntry.includes('warning')) {
      return <AlertCircle className="w-4 h-4 text-yellow-500" />;
    } else {
      return <Info className="w-4 h-4 text-blue-500" />;
    }
  };

  const getLogEntryStyle = (entry: string) => {
    const lowerEntry = entry.toLowerCase();
    if (lowerEntry.includes('error') || lowerEntry.includes('failed')) {
      return 'bg-red-50 border-red-200 text-red-800';
    } else if (lowerEntry.includes('success') || lowerEntry.includes('completed')) {
      return 'bg-green-50 border-green-200 text-green-800';
    } else if (lowerEntry.includes('warning')) {
      return 'bg-yellow-50 border-yellow-200 text-yellow-800';
    } else {
      return 'bg-gray-50 border-gray-200 text-gray-700';
    }
  };

  const filteredLog = processLog.filter(entry =>
    entry.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-4xl mx-4 h-[80vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <Terminal className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
            <span className="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full">
              {processLog.length} entries
            </span>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={downloadLog}
              disabled={processLog.length === 0}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Download className="w-4 h-4" />
              <span>Download Log</span>
            </button>
            
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="p-4 border-b border-gray-200">
          <div className="relative">
            <input
              type="text"
              placeholder="Search log entries..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-4 py-2 pl-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
              <Terminal className="w-4 h-4 text-gray-400" />
            </div>
          </div>
          {searchTerm && (
            <p className="text-sm text-gray-600 mt-2">
              Showing {filteredLog.length} of {processLog.length} entries
            </p>
          )}
        </div>

        {/* Log Content */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {processLog.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <Terminal className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No process log available</p>
                <p className="text-sm text-gray-500 mt-2">
                  Process logs will appear here during forecast generation
                </p>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {filteredLog.map((entry, index) => (
                <div
                  key={index}
                  className={`flex items-start space-x-3 p-3 border rounded-lg ${getLogEntryStyle(entry)}`}
                >
                  <div className="flex-shrink-0 mt-0.5">
                    {getLogEntryIcon(entry)}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <Clock className="w-3 h-3 text-gray-500" />
                      <span className="text-xs text-gray-500">
                        Entry {processLog.indexOf(entry) + 1}
                      </span>
                    </div>
                    <p className="text-sm font-mono leading-relaxed">
                      {entry}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <CheckCircle className="w-3 h-3 text-green-500" />
                <span>Success</span>
              </div>
              <div className="flex items-center space-x-1">
                <AlertCircle className="w-3 h-3 text-red-500" />
                <span>Error</span>
              </div>
              <div className="flex items-center space-x-1">
                <AlertCircle className="w-3 h-3 text-yellow-500" />
                <span>Warning</span>
              </div>
              <div className="flex items-center space-x-1">
                <Info className="w-3 h-3 text-blue-500" />
                <span>Info</span>
              </div>
            </div>
            <div>
              Total entries: {processLog.length}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};