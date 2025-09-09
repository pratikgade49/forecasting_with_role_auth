import React, { useState } from 'react';
import { X, Save, AlertCircle } from 'lucide-react';
import { ApiService, SaveForecastRequest, ForecastResult, MultiForecastResult, ForecastConfig } from '../services/api';

interface SaveForecastModalProps {
  isOpen: boolean;
  onClose: () => void;
  forecastResult: ForecastResult | MultiForecastResult;
  forecastConfig: ForecastConfig;
  onSaveSuccess: () => void;
}

export const SaveForecastModal: React.FC<SaveForecastModalProps> = ({
  isOpen,
  onClose,
  forecastResult,
  forecastConfig,
  onSaveSuccess
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    if (!name.trim()) {
      setError('Forecast name is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request: SaveForecastRequest = {
        name: name.trim(),
        description: description.trim() || undefined,
        forecast_config: forecastConfig,
        forecast_data: forecastResult
      };

      await ApiService.saveForecast(request);
      onSaveSuccess();
      onClose();
      
      // Reset form
      setName('');
      setDescription('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save forecast');
    } finally {
      setLoading(false);
    }
  };

  const generateSuggestedName = () => {
    const timestamp = new Date().toLocaleDateString();
    const algorithm = forecastConfig.algorithm.replace('_', ' ');
    
    if (forecastConfig.multiSelect) {
      return `Multi-Forecast ${algorithm} - ${timestamp}`;
    } else if (forecastConfig.selectedItems && forecastConfig.selectedItems.length > 1) {
      return `${forecastConfig.selectedItems.length} ${forecastConfig.forecastBy}s ${algorithm} - ${timestamp}`;
    } else if (forecastConfig.selectedItem) {
      return `${forecastConfig.selectedItem} ${algorithm} - ${timestamp}`;
    } else {
      return `Forecast ${algorithm} - ${timestamp}`;
    }
  };

  const handleUseSuggested = () => {
    setName(generateSuggestedName());
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-md mx-4 shadow-2xl">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Save className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-semibold text-gray-900">Save Forecast</h2>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center">
                <AlertCircle className="w-4 h-4 text-red-500 mr-2" />
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Forecast Name *
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter a name for this forecast"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                maxLength={255}
              />
              <button
                type="button"
                onClick={handleUseSuggested}
                className="mt-1 text-xs text-blue-600 hover:text-blue-700 font-medium"
              >
                Use suggested name
              </button>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Add a description for this forecast..."
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                maxLength={1000}
              />
            </div>

            {/* Forecast Summary */}
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <h4 className="font-medium text-blue-900 mb-2">Forecast Summary</h4>
              <div className="text-sm text-blue-800 space-y-1">
                <p><strong>Algorithm:</strong> {forecastConfig.algorithm.replace('_', ' ')}</p>
                <p><strong>Interval:</strong> {forecastConfig.interval}</p>
                <p><strong>Periods:</strong> {forecastConfig.historicPeriod}H / {forecastConfig.forecastPeriod}F</p>
                {'results' in forecastResult ? (
                  <p><strong>Type:</strong> Multi-Forecast ({(forecastResult as MultiForecastResult).totalCombinations} combinations)</p>
                ) : (
                  <p><strong>Accuracy:</strong> {(forecastResult as ForecastResult).accuracy}%</p>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center justify-end space-x-3 mt-6">
            <button
              onClick={onClose}
              className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={loading || !name.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                  <span>Saving...</span>
                </div>
              ) : (
                'Save Forecast'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};