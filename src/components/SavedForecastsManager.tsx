import React, { useState, useEffect } from 'react';
import { X, Save, Trash2, Eye, Calendar, Target, Brain, AlertCircle, Search, Filter } from 'lucide-react';
import { ApiService, SavedForecast, ForecastResult, MultiForecastResult } from '../services/api';

interface SavedForecastsManagerProps {
  isOpen: boolean;
  onClose: () => void;
  onViewForecast: (forecast: ForecastResult | MultiForecastResult, config: any) => void;
}

export const SavedForecastsManager: React.FC<SavedForecastsManagerProps> = ({
  isOpen,
  onClose,
  onViewForecast
}) => {
  const [savedForecasts, setSavedForecasts] = useState<SavedForecast[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'name' | 'created_at' | 'algorithm'>('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    if (isOpen) {
      loadSavedForecasts();
    }
  }, [isOpen]);

  const loadSavedForecasts = async () => {
    setLoading(true);
    setError(null);
    try {
      const forecasts = await ApiService.getSavedForecasts();
      setSavedForecasts(forecasts);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load saved forecasts');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteForecast = async (id: number, name: string) => {
    if (!confirm(`Are you sure you want to delete the forecast "${name}"?`)) {
      return;
    }

    try {
      await ApiService.deleteSavedForecast(id);
      await loadSavedForecasts(); // Refresh the list
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete forecast');
    }
  };

  const handleViewForecast = (savedForecast: SavedForecast) => {
    onViewForecast(savedForecast.forecast_data, savedForecast.forecast_config);
    onClose();
  };

  const getAlgorithmDisplayName = (algorithm: string) => {
    const algorithmMap: Record<string, string> = {
      'linear_regression': 'Linear Regression',
      'polynomial_regression': 'Polynomial Regression',
      'exponential_smoothing': 'Exponential Smoothing',
      'holt_winters': 'Holt-Winters',
      'arima': 'ARIMA',
      'random_forest': 'Random Forest',
      'seasonal_decomposition': 'Seasonal Decomposition',
      'moving_average': 'Moving Average',
      'best_fit': 'Best Fit'
    };
    return algorithmMap[algorithm] || algorithm;
  };

  const getAlgorithmColor = (algorithm: string) => {
    const colors: Record<string, string> = {
      'linear_regression': 'bg-blue-100 text-blue-800',
      'random_forest': 'bg-green-100 text-green-800',
      'best_fit': 'bg-purple-100 text-purple-800',
      'holt_winters': 'bg-orange-100 text-orange-800',
      'arima': 'bg-yellow-100 text-yellow-800'
    };
    return colors[algorithm] || 'bg-gray-100 text-gray-800';
  };

  const getForecastTypeDisplay = (config: any) => {
    if (config.multiSelect) {
      const dimensions = [];
      if (config.selectedProducts && config.selectedProducts.length > 0) {
        dimensions.push(`${config.selectedProducts.length} Products`);
      }
      if (config.selectedCustomers && config.selectedCustomers.length > 0) {
        dimensions.push(`${config.selectedCustomers.length} Customers`);
      }
      if (config.selectedLocations && config.selectedLocations.length > 0) {
        dimensions.push(`${config.selectedLocations.length} Locations`);
      }
      return dimensions.join(' × ') || 'Multi-select';
    } else if (config.selectedItems && config.selectedItems.length > 1) {
      return `${config.selectedItems.length} ${config.forecastBy}s`;
    } else if (config.selectedProduct && config.selectedCustomer && config.selectedLocation) {
      return `${config.selectedProduct} → ${config.selectedCustomer} → ${config.selectedLocation}`;
    } else if (config.selectedItem) {
      return `${config.forecastBy}: ${config.selectedItem}`;
    }
    return 'Unknown configuration';
  };

  const isMultiForecast = (forecastData: any) => {
    return 'results' in forecastData && Array.isArray(forecastData.results);
  };

  const getAccuracyDisplay = (forecastData: any) => {
    if (isMultiForecast(forecastData)) {
      return `${forecastData.summary?.averageAccuracy || 0}% avg`;
    } else {
      return `${forecastData.accuracy || 0}%`;
    }
  };

  // Filter and sort forecasts
  const filteredAndSortedForecasts = savedForecasts
    .filter(forecast => 
      forecast.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (forecast.description && forecast.description.toLowerCase().includes(searchTerm.toLowerCase()))
    )
    .sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'name':
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case 'created_at':
          aValue = new Date(a.created_at).getTime();
          bValue = new Date(b.created_at).getTime();
          break;
        case 'algorithm':
          aValue = a.forecast_config.algorithm;
          bValue = b.forecast_config.algorithm;
          break;
        default:
          aValue = new Date(a.created_at).getTime();
          bValue = new Date(b.created_at).getTime();
      }
      
      if (sortOrder === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-6xl mx-4 h-[85vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <Save className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">My Saved Forecasts</h2>
            <span className="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full">
              {savedForecasts.length} saved
            </span>
          </div>
          
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Controls */}
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between space-x-4">
            {/* Search */}
            <div className="flex-1 max-w-md">
              <div className="relative">
                <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search forecasts..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
            
            {/* Sort Controls */}
            <div className="flex items-center space-x-2">
              <Filter className="w-4 h-4 text-gray-500" />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="created_at">Date Created</option>
                <option value="name">Name</option>
                <option value="algorithm">Algorithm</option>
              </select>
              
              <button
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 transition-colors"
              >
                {sortOrder === 'asc' ? '↑' : '↓'}
              </button>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                <p className="text-red-600 mb-4">{error}</p>
                <button
                  onClick={loadSavedForecasts}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : filteredAndSortedForecasts.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Save className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {searchTerm ? 'No forecasts found' : 'No saved forecasts'}
                </h3>
                <p className="text-gray-600 mb-4">
                  {searchTerm 
                    ? `No forecasts match "${searchTerm}"`
                    : 'Generate and save forecasts to see them here'
                  }
                </p>
                {searchTerm && (
                  <button
                    onClick={() => setSearchTerm('')}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Clear Search
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="overflow-y-auto h-full">
              <div className="p-4 space-y-4">
                {filteredAndSortedForecasts.map((savedForecast) => (
                  <div
                    key={savedForecast.id}
                    className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <h3 className="font-semibold text-gray-900">{savedForecast.name}</h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            getAlgorithmColor(savedForecast.forecast_config.algorithm)
                          }`}>
                            {getAlgorithmDisplayName(savedForecast.forecast_config.algorithm)}
                          </span>
                          {isMultiForecast(savedForecast.forecast_data) && (
                            <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded-full text-xs font-medium">
                              Multi-Forecast
                            </span>
                          )}
                        </div>
                        
                        {savedForecast.description && (
                          <p className="text-sm text-gray-600 mb-2">{savedForecast.description}</p>
                        )}
                        
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-500">
                          <div>
                            <span className="font-medium">Configuration:</span>
                            <p className="text-gray-700">{getForecastTypeDisplay(savedForecast.forecast_config)}</p>
                          </div>
                          <div>
                            <span className="font-medium">Accuracy:</span>
                            <p className="text-gray-700 font-medium">{getAccuracyDisplay(savedForecast.forecast_data)}</p>
                          </div>
                          <div>
                            <span className="font-medium">Created:</span>
                            <p className="text-gray-700">{new Date(savedForecast.created_at).toLocaleDateString()}</p>
                          </div>
                        </div>
                        
                        <div className="mt-2 text-xs text-gray-500">
                          <span className="font-medium">Periods:</span> {savedForecast.forecast_config.historicPeriod}H / {savedForecast.forecast_config.forecastPeriod}F
                          <span className="ml-4 font-medium">Interval:</span> {savedForecast.forecast_config.interval}
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2 ml-4">
                        <button
                          onClick={() => handleViewForecast(savedForecast)}
                          className="flex items-center space-x-1 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                          <span>View</span>
                        </button>
                        
                        <button
                          onClick={() => handleDeleteForecast(savedForecast.id, savedForecast.name)}
                          className="flex items-center space-x-1 px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                          <span>Delete</span>
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div>
              Showing {filteredAndSortedForecasts.length} of {savedForecasts.length} forecasts
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <Target className="w-3 h-3 text-blue-500" />
                <span>Single Forecast</span>
              </div>
              <div className="flex items-center space-x-1">
                <Brain className="w-3 h-3 text-indigo-500" />
                <span>Multi-Forecast</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};