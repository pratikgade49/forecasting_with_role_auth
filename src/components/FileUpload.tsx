import React, { useCallback } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  loading: boolean;
  error: string | null;
  disabled?: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, loading, error, disabled = false }) => {
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
          disabled
            ? 'border-gray-200 bg-gray-50 cursor-not-allowed'
            : loading
            ? 'border-blue-300 bg-blue-50'
            : error
            ? 'border-red-300 bg-red-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          type="file"
          accept=".xlsx,.xls,.csv"
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={loading || disabled}
        />
        
        <div className="space-y-4">
          {loading ? (
            <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
          ) : error ? (
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto" />
          ) : (
            <Upload className="w-12 h-12 text-gray-400 mx-auto" />
          )}
          
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              {disabled 
                ? 'Backend server required' 
                : loading 
                ? 'Processing file...' 
                : 'Upload your data file'
              }
            </h3>
            <p className="text-gray-600 mb-4">
              {disabled
                ? 'Please start the Python backend server to upload files'
                : error 
                ? error 
                : 'Drag and drop your Excel (.xlsx, .xls) or CSV file here, or click to browse'
              }
            </p>
          </div>
          
          <div className="flex items-center justify-center space-x-4 text-sm text-gray-500">
            <div className="flex items-center">
              <FileText className="w-4 h-4 mr-1" />
              Excel
            </div>
            <div className="flex items-center">
              <FileText className="w-4 h-4 mr-1" />
              CSV
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};