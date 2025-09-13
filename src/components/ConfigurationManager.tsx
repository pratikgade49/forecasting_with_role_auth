import React, { useState, useEffect } from 'react';
import { Save, Settings, Trash2, Clock, Eye, Plus, X, Edit } from 'lucide-react';
import { ForecastConfig, SavedConfiguration, ApiService, SaveConfigRequest } from '../services/api';

interface ConfigurationManagerProps {
  config: ForecastConfig;
  onLoadConfiguration: (config: ForecastConfig) => void;
  onApply?: (config: ForecastConfig) => void;
  onUpdate?: (updatedConfig: any) => void;
  onDelete?: (configId: number) => void;
  onClose?: () => void;
  onSaveSuccess?: () => void;
}

export const ConfigurationManager: React.FC<ConfigurationManagerProps> = ({
  config,
  onLoadConfiguration,
  onApply,
  onUpdate,
  onDelete,
  onClose,
  onSaveSuccess
}) => {
  const [savedConfigurations, setSavedConfigurations] = useState<SavedConfiguration[]>([]);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showLoadDialog, setShowLoadDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [editingConfig, setEditingConfig] = useState<SavedConfiguration | null>(null);
  const [saveName, setSaveName] = useState('');
  const [saveDescription, setSaveDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadConfigurations();
  }, []);

  const loadConfigurations = async () => {
    try {
      const response = await ApiService.getConfigurations();
      setSavedConfigurations(response.configurations);
    } catch (error) {
      console.error('Failed to load configurations:', error);
    }
  };

  const handleSaveConfiguration = async () => {
    if (!saveName.trim()) {
      setError('Configuration name is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await ApiService.saveConfiguration({
        name: saveName.trim(),
        description: saveDescription.trim() || undefined,
        config
      });

      setShowSaveDialog(false);
      setSaveName('');
      setSaveDescription('');
      await loadConfigurations();
      
      if (onSaveSuccess) {
        onSaveSuccess();
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to save configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadConfiguration = async (configId: number) => {
    try {
      const savedConfig = await ApiService.getConfiguration(configId);
      onLoadConfiguration(savedConfig.config);
      setShowLoadDialog(false);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to load configuration');
    }
  };

  const handleDeleteConfiguration = async (configId: number, configName: string) => {
    if (!confirm(`Are you sure you want to delete the configuration "${configName}"?`)) {
      return;
    }

    try {
      await ApiService.deleteConfiguration(configId);
      await loadConfigurations();
      if (onDelete) {
        onDelete(configId);
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to delete configuration');
    }
  };

  const handleEditConfiguration = (config: SavedConfiguration) => {
    setEditingConfig(config);
    setSaveName(config.name);
    setSaveDescription(config.description || '');
    setShowEditDialog(true);
  };

  const handleUpdateConfiguration = async () => {
    if (!editingConfig || !saveName.trim()) {
      setError('Configuration name is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await ApiService.updateConfiguration(editingConfig.id, {
        name: saveName.trim(),
        description: saveDescription.trim() || undefined,
        config: editingConfig.config
      });

      setShowEditDialog(false);
      setEditingConfig(null);
      setSaveName('');
      setSaveDescription('');
      await loadConfigurations();
      
      if (onUpdate) {
        onUpdate(editingConfig);
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to update configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyConfiguration = async (config: ForecastConfig) => {
    if (onApply) {
      onApply(config);
    } else {
      onLoadConfiguration(config);
    }
    setShowLoadDialog(false);
  };
  const getConfigurationSummary = (config: ForecastConfig) => {
    if (config.selectedProducts && config.selectedProducts.length === 1 &&
        config.selectedCustomers && config.selectedCustomers.length === 1 &&
        config.selectedLocations && config.selectedLocations.length === 1) {
      return `${config.selectedProducts[0]} → ${config.selectedCustomers[0]} → ${config.selectedLocations[0]}`;
    } else if (config.selectedItems && config.selectedItems.length === 1) {
      return `${config.forecastBy}: ${config.selectedItems[0]}`;
    } else if (config.selectedItems && config.selectedItems.length > 1) {
      return `${config.selectedItems.length} ${config.forecastBy}s selected`;
    } else {
      return `${config.forecastBy}: Not selected`;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Settings className="w-6 h-6 text-purple-600" />
          <h2 className="text-xl font-semibold text-gray-900">Configuration Manager</h2>
        </div>
        
        <div className="flex items-center space-x-3">
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          )}
          
          <button
            onClick={() => setShowLoadDialog(true)}
            disabled={savedConfigurations.length === 0}
            className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Eye className="w-4 h-4 mr-2" />
            Load ({savedConfigurations.length})
          </button>
          
          <button
            onClick={() => setShowSaveDialog(true)}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 transition-colors"
          >
            <Save className="w-4 h-4 mr-2" />
            Save Current
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      {/* Current Configuration Summary */}
      <div className="p-4 bg-gray-50 rounded-lg">
        <h3 className="font-medium text-gray-900 mb-2">Current Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Selection:</span>
            <p className="font-medium text-gray-900">{getConfigurationSummary(config)}</p>
          </div>
          <div>
            <span className="text-gray-600">Algorithm:</span>
            <p className="font-medium text-gray-900">{config.algorithm.replace('_', ' ')}</p>
          </div>
          <div>
            <span className="text-gray-600">Interval:</span>
            <p className="font-medium text-gray-900">{config.interval}</p>
          </div>
          <div>
            <span className="text-gray-600">Periods:</span>
            <p className="font-medium text-gray-900">{config.historicPeriod}H / {config.forecastPeriod}F</p>
          </div>
        </div>
      </div>

      {/* Save Configuration Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Save Configuration</h3>
              <button
                onClick={() => setShowSaveDialog(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Configuration Name *
                </label>
                <input
                  type="text"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder="e.g., Product A Monthly Forecast"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description (Optional)
                </label>
                <textarea
                  value={saveDescription}
                  onChange={(e) => setSaveDescription(e.target.value)}
                  placeholder="Brief description of this configuration..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
            </div>
            
            <div className="flex items-center justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveConfiguration}
                disabled={loading || !saveName.trim()}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Saving...' : 'Save Configuration'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Load Configuration Dialog */}
      {showLoadDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-4xl mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Load Configuration</h3>
              <button
                onClick={() => setShowLoadDialog(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {savedConfigurations.length === 0 ? (
              <div className="text-center py-8">
                <Settings className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No saved configurations found.</p>
                <p className="text-sm text-gray-500 mt-1">Save your current configuration to get started.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {savedConfigurations.map((savedConfig) => (
                  <div
                    key={savedConfig.id}
                    className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <h4 className="font-medium text-gray-900">{savedConfig.name}</h4>
                          <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                            {savedConfig.config.algorithm.replace('_', ' ')}
                          </span>
                        </div>
                        
                        {savedConfig.description && (
                          <p className="text-sm text-gray-600 mb-2">{savedConfig.description}</p>
                        )}
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs text-gray-500">
                          <div>
                            <span className="font-medium">Selection:</span>
                            <p>{getConfigurationSummary(savedConfig.config)}</p>
                          </div>
                          <div>
                            <span className="font-medium">Interval:</span>
                            <p>{savedConfig.config.interval}</p>
                          </div>
                          <div>
                            <span className="font-medium">Periods:</span>
                            <p>{savedConfig.config.historicPeriod}H / {savedConfig.config.forecastPeriod}F</p>
                          </div>
                          <div>
                            <span className="font-medium">Updated:</span>
                            <p>{new Date(savedConfig.updatedAt).toLocaleDateString()}</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2 ml-4">
                        <button
                          onClick={() => handleApplyConfiguration(savedConfig.config)}
                          className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 transition-colors"
                        >
                          {onApply ? 'Apply' : 'Load'}
                        </button>
                        <button
                          onClick={() => handleEditConfiguration(savedConfig)}
                          className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700 transition-colors"
                        >
                          <Edit className="w-3 h-3" />
                        </button>
                        <button
                          onClick={() => handleDeleteConfiguration(savedConfig.id, savedConfig.name)}
                          className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Edit Configuration Dialog */}
      {showEditDialog && editingConfig && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Edit Configuration</h3>
              <button
                onClick={() => setShowEditDialog(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Configuration Name *
                </label>
                <input
                  type="text"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder="e.g., Product A Monthly Forecast"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description (Optional)
                </label>
                <textarea
                  value={saveDescription}
                  onChange={(e) => setSaveDescription(e.target.value)}
                  placeholder="Brief description of this configuration..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
            </div>
            
            <div className="flex items-center justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowEditDialog(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleUpdateConfiguration}
                disabled={loading || !saveName.trim()}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Updating...' : 'Update Configuration'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};