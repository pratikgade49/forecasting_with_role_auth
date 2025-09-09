import React, { useState, useEffect } from 'react';
import { Database, Trash2, RefreshCw, Clock, Target, TrendingUp } from 'lucide-react';
import { ApiService, ModelCacheInfo } from '../services/api';

export const ModelCacheManager: React.FC = () => {
  const [cacheInfo, setCacheInfo] = useState<ModelCacheInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    loadCacheInfo();
  }, []);

  const loadCacheInfo = async () => {
    setLoading(true);
    try {
      const info = await ApiService.getModelCacheInfo();
      setCacheInfo(info);
    } catch (error) {
      console.error('Failed to load cache info:', error);
    } finally {
      setLoading(false);
    }
  };

  const clearCache = async () => {
    if (!confirm('Are you sure you want to clear old cached models? This will remove models older than 7 days and keep only the 50 most recent models.')) {
      return;
    }

    setClearing(true);
    try {
      const result = await ApiService.clearModelCache();
      alert(`Cache cleared successfully! Removed ${result.cleared_count} models.`);
      await loadCacheInfo();
    } catch (error) {
      console.error('Failed to clear cache:', error);
      alert('Failed to clear cache');
    } finally {
      setClearing(false);
    }
  };

  const clearAllCache = async () => {
    if (!confirm('Are you sure you want to clear ALL cached models? This action cannot be undone and will remove all cached models immediately.')) {
      return;
    }

    setClearing(true);
    try {
      const result = await ApiService.clearAllModelCache();
      alert(`All cache cleared successfully! Removed ${result.cleared_count} models.`);
      await loadCacheInfo();
    } catch (error) {
      console.error('Failed to clear all cache:', error);
      alert('Failed to clear all cache');
    } finally {
      setClearing(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString() + ' ' + new Date(dateString).toLocaleTimeString();
  };

  const getAlgorithmDisplayName = (algorithmKey: string) => {
    const algorithmMap: Record<string, string> = {
      'linear_regression': 'Linear Regression',
      'polynomial_regression': 'Polynomial Regression',
      'exponential_smoothing': 'Exponential Smoothing',
      'holt_winters': 'Holt-Winters',
      'arima': 'ARIMA',
      'random_forest': 'Random Forest',
      'seasonal_decomposition': 'Seasonal Decomposition',
      'moving_average': 'Moving Average',
      'sarima': 'SARIMA',
      'prophet_like': 'Prophet-like Forecasting',
      'lstm_like': 'Simple LSTM-like',
      'xgboost': 'XGBoost',
      'svr': 'Support Vector Regression',
      'knn': 'K-Nearest Neighbors',
      'gaussian_process': 'Gaussian Process',
      'neural_network': 'Neural Network',
      'theta_method': 'Theta Method',
      'croston': 'Croston\'s Method',
      'ses': 'Simple Exponential Smoothing',
      'damped_trend': 'Damped Trend',
      'naive_seasonal': 'Naive Seasonal',
      'drift_method': 'Drift Method',
      'best_fit': 'Best Fit'
    };
    return algorithmMap[algorithmKey] || algorithmKey;
  };

  const getAlgorithmColor = (algorithmKey: string) => {
    const displayName = getAlgorithmDisplayName(algorithmKey);
    const colors = {
      'Linear Regression': 'bg-blue-100 text-blue-800',
      'Random Forest': 'bg-green-100 text-green-800',
      'XGBoost': 'bg-purple-100 text-purple-800',
      'Neural Network': 'bg-red-100 text-red-800',
      'ARIMA': 'bg-yellow-100 text-yellow-800',
      'Holt-Winters': 'bg-orange-100 text-orange-800',
      'Seasonal Decomposition': 'bg-pink-100 text-pink-800',
      'Simple Exponential Smoothing': 'bg-indigo-100 text-indigo-800',
      'K-Nearest Neighbors': 'bg-teal-100 text-teal-800',
      'Gaussian Process': 'bg-cyan-100 text-cyan-800',
      'Support Vector Regression': 'bg-lime-100 text-lime-800',
      'Best Fit': 'bg-emerald-100 text-emerald-800',
    };
    return colors[displayName as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Database className="w-6 h-6 text-blue-600" />
          <h3 className="text-xl font-semibold text-gray-900">Model Cache Manager</h3>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={loadCacheInfo}
            disabled={loading}
            className="flex items-center space-x-2 px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span className="text-sm">Refresh</span>
          </button>
          
          <button
            onClick={clearCache}
            disabled={clearing || cacheInfo.length === 0}
            className="flex items-center space-x-2 px-3 py-1 bg-yellow-100 hover:bg-yellow-200 text-yellow-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <Trash2 className="w-4 h-4" />
            <span className="text-sm">{clearing ? 'Clearing...' : 'Clear Old'}</span>
          </button>
          
          <button
            onClick={clearAllCache}
            disabled={clearing || cacheInfo.length === 0}
            className="flex items-center space-x-2 px-3 py-1 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <Trash2 className="w-4 h-4" />
            <span className="text-sm">{clearing ? 'Clearing...' : 'Clear All'}</span>
          </button>
        </div>
      </div>

      {/* Cache Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-600">Total Models</p>
              <p className="text-2xl font-bold text-blue-900">{cacheInfo.length}</p>
            </div>
            <Database className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-600">Avg Accuracy</p>
              <p className="text-2xl font-bold text-green-900">
                {cacheInfo.length > 0 
                  ? (cacheInfo.reduce((sum, model) => sum + (model.accuracy || 0), 0) / cacheInfo.length).toFixed(1)
                  : '0'
                }%
              </p>
            </div>
            <Target className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-purple-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-600">Total Uses</p>
              <p className="text-2xl font-bold text-purple-900">
                {cacheInfo.reduce((sum, model) => sum + model.use_count, 0)}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Cache Table */}
      <div className="overflow-x-auto">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
          </div>
        ) : cacheInfo.length > 0 ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-medium text-gray-700">Algorithm</th>
                <th className="text-center py-3 px-4 font-medium text-gray-700">Accuracy</th>
                <th className="text-center py-3 px-4 font-medium text-gray-700">Use Count</th>
                <th className="text-center py-3 px-4 font-medium text-gray-700">Created</th>
                <th className="text-center py-3 px-4 font-medium text-gray-700">Last Used</th>
                <th className="text-center py-3 px-4 font-medium text-gray-700">Model Hash</th>
              </tr>
            </thead>
            <tbody>
              {cacheInfo.map((model, index) => (
                <tr key={model.model_hash} className={`border-b border-gray-100 ${
                  index % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                }`}>
                  <td className="py-3 px-4">
                    <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                      getAlgorithmColor(model.algorithm)
                    }`}>
                      {getAlgorithmDisplayName(model.algorithm)}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className="font-medium text-gray-900">
                      {model.accuracy ? `${model.accuracy.toFixed(1)}%` : 'N/A'}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className="font-medium text-gray-900">{model.use_count}</span>
                  </td>
                  <td className="py-3 px-4 text-center text-gray-600">
                    {formatDate(model.created_at)}
                  </td>
                  <td className="py-3 px-4 text-center text-gray-600">
                    {formatDate(model.last_used)}
                  </td>
                  <td className="py-3 px-4 text-center">
                    <code className="text-xs bg-gray-100 px-2 py-1 rounded">
                      {model.model_hash.substring(0, 8)}...
                    </code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="text-center py-8">
            <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No cached models found</p>
            <p className="text-sm text-gray-500 mt-2">
              Generate some forecasts to start building the model cache
            </p>
          </div>
        )}
      </div>

      {/* Cache Information */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="font-medium text-blue-900 mb-2">About Model Caching</h4>
        <div className="text-sm text-blue-700 space-y-1">
          <p>• Models are automatically cached when forecasts are generated</p>
          <p>• Cached models are reused for identical configurations and data</p>
          <p>• This significantly speeds up re-forecasting with the same parameters</p>
          <p>• Old models are automatically cleaned up to save storage space</p>
        </div>
      </div>
    </div>
  );
};