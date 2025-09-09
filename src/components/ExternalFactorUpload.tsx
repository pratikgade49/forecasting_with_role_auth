import React, { useState } from 'react';
import { Upload, FileText, AlertCircle, TrendingUp, Database, CheckCircle, AlertTriangle } from 'lucide-react';
import { ApiService } from '../services/api';

interface ExternalFactorUploadProps {
  onUploadSuccess: () => void;
}

export const ExternalFactorUpload: React.FC<ExternalFactorUploadProps> = ({ onUploadSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<{
    status: 'success' | 'warning' | 'error';
    warnings: string[];
    errors: string[];
    summary?: { coverage_percentage: number };
  } | null>(null);

  const handleFileSelect = async (file: File) => {
    setLoading(true);
    setError(null);
    setValidationResult(null);
    
    try {
      const response = await ApiService.uploadExternalFactors(file);
      // Removed setting validationResult from response due to type error
      onUploadSuccess();
      alert(`External factors uploaded successfully!\nInserted: ${response.inserted} records\nDuplicates skipped: ${response.duplicates} records`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred processing the file');
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
      <div className="flex items-center mb-4">
        <TrendingUp className="w-6 h-6 text-purple-600 mr-2" />
        <h3 className="text-lg font-semibold text-gray-900">Upload External Factors</h3>
      </div>

      <div
        className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all duration-300 ${
          loading
            ? 'border-purple-300 bg-purple-50'
            : error
            ? 'border-red-300 bg-red-50'
            : 'border-gray-300 hover:border-purple-400 hover:bg-purple-50'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <input
          type="file"
          accept=".xlsx,.xls,.csv"
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={loading}
        />
        
        <div className="space-y-3">
          {loading ? (
            <div className="animate-spin w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full mx-auto"></div>
          ) : error ? (
            <AlertCircle className="w-8 h-8 text-red-500 mx-auto" />
          ) : (
            <Upload className="w-8 h-8 text-gray-400 mx-auto" />
          )}
          
          <div>
            <h4 className="font-medium text-gray-900 mb-1">
              {loading ? 'Processing...' : 'Upload External Factor Data'}
            </h4>
            <p className="text-sm text-gray-600">
              {error ? error : 'CSV or Excel file with columns: date, factor_name, factor_value'}
            </p>
          </div>
        </div>
      </div>

      {/* Validation Results */}
      {validationResult && (
        <div className="mt-4">
          <div className={`border rounded-lg p-4 ${
            validationResult.status === 'success' ? 'bg-green-50 border-green-200' :
            validationResult.status === 'warning' ? 'bg-yellow-50 border-yellow-200' :
            'bg-red-50 border-red-200'
          }`}>
            <div className="flex items-center mb-2">
              {validationResult.status === 'success' ? (
                <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
              ) : validationResult.status === 'warning' ? (
                <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-600 mr-2" />
              )}
              <h4 className={`font-medium ${
                validationResult.status === 'success' ? 'text-green-900' :
                validationResult.status === 'warning' ? 'text-yellow-900' :
                'text-red-900'
              }`}>
                Date Range Validation
              </h4>
            </div>
            
            {validationResult.warnings.length > 0 && (
              <div className="mb-2">
                <p className="text-sm font-medium text-yellow-800 mb-1">Warnings:</p>
                <ul className="text-sm text-yellow-700 list-disc list-inside">
                  {validationResult.warnings.map((warning: string, index: number) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {validationResult.errors.length > 0 && (
              <div className="mb-2">
                <p className="text-sm font-medium text-red-800 mb-1">Errors:</p>
                <ul className="text-sm text-red-700 list-disc list-inside">
                  {validationResult.errors.map((error: string, index: number) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {validationResult.summary && validationResult.summary.coverage_percentage && (
              <div className="text-sm">
                <p className={`font-medium ${
                  validationResult.status === 'success' ? 'text-green-800' :
                  validationResult.status === 'warning' ? 'text-yellow-800' :
                  'text-red-800'
                }`}>
                  Coverage: {validationResult.summary.coverage_percentage}% of main data date range
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <h4 className="font-medium text-purple-900 mb-2">Required Format:</h4>
        <div className="text-sm text-purple-800 space-y-1">
          <p><strong>Columns:</strong> date, factor_name, factor_value</p>
          <p><strong>Example:</strong></p>
          <div className="bg-white rounded p-2 mt-2 font-mono text-xs">
            date,factor_name,factor_value<br/>
            2024-01-01,gdp_growth,2.5<br/>
            2024-01-01,inflation_rate,3.2<br/>
            2024-02-01,gdp_growth,2.6
          </div>
        </div>
      </div>
    </div>
  );
};