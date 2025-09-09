import React, { useState, useEffect } from 'react';
import { 
  Clock, Plus, Play, Pause, Trash2, Eye, Calendar, Settings, 
  CheckCircle, XCircle, AlertCircle, RefreshCw, X, Edit
} from 'lucide-react';
import { ApiService, ScheduledForecast, ForecastExecution, ForecastConfig } from '../services/api';

interface ScheduledForecastManagerProps {
  isOpen: boolean;
  onClose: () => void;
  currentConfig?: ForecastConfig;
}

export const ScheduledForecastManager: React.FC<ScheduledForecastManagerProps> = ({
  isOpen,
  onClose,
  currentConfig
}) => {
  const [scheduledForecasts, setScheduledForecasts] = useState<ScheduledForecast[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showExecutionHistory, setShowExecutionHistory] = useState<number | null>(null);
  const [executions, setExecutions] = useState<ForecastExecution[]>([]);
  const [schedulerStatus, setSchedulerStatus] = useState<{ running: boolean; check_interval: number } | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadScheduledForecasts();
      loadSchedulerStatus();
    }
  }, [isOpen]);

  const loadScheduledForecasts = async () => {
    setLoading(true);
    setError(null);
    try {
      const forecasts = await ApiService.getScheduledForecasts();
      setScheduledForecasts(forecasts);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load scheduled forecasts');
    } finally {
      setLoading(false);
    }
  };

  const loadSchedulerStatus = async () => {
    try {
      const status = await ApiService.getSchedulerStatus();
      setSchedulerStatus(status);
    } catch (err) {
      console.error('Failed to load scheduler status:', err);
    }
  };

  const handleDeleteForecast = async (id: number, name: string) => {
    if (!confirm(`Are you sure you want to delete the scheduled forecast "${name}"?`)) {
      return;
    }

    try {
      await ApiService.deleteScheduledForecast(id);
      await loadScheduledForecasts();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete scheduled forecast');
    }
  };

  const handleToggleStatus = async (forecast: ScheduledForecast) => {
    const newStatus = forecast.status === 'active' ? 'paused' : 'active';
    
    try {
      await ApiService.updateScheduledForecast(forecast.id, { status: newStatus });
      await loadScheduledForecasts();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update forecast status');
    }
  };

  const loadExecutionHistory = async (forecastId: number) => {
    try {
      const executions = await ApiService.getForecastExecutions(forecastId);
      setExecutions(executions);
      setShowExecutionHistory(forecastId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load execution history');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'paused':
        return <Pause className="w-4 h-4 text-yellow-500" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-blue-500" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800';
      case 'completed':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getFrequencyDisplay = (frequency: string) => {
    return frequency.charAt(0).toUpperCase() + frequency.slice(1);
  };

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getSuccessRate = (forecast: ScheduledForecast) => {
    if (forecast.run_count === 0) return 0;
    return Math.round((forecast.success_count / forecast.run_count) * 100);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-7xl mx-4 h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <Clock className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">Scheduled Forecasts</h2>
            <span className="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full">
              {scheduledForecasts.length} schedules
            </span>
            {schedulerStatus && (
              <span className={`text-sm px-3 py-1 rounded-full ${
                schedulerStatus.running 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                Scheduler: {schedulerStatus.running ? 'Running' : 'Stopped'}
              </span>
            )}
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Schedule Forecast</span>
            </button>
            
            <button
              onClick={loadScheduledForecasts}
              disabled={loading}
              className="flex items-center space-x-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
            
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <AlertCircle className="w-4 h-4 text-red-500 mr-2" />
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
            </div>
          ) : scheduledForecasts.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Clock className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Scheduled Forecasts</h3>
                <p className="text-gray-600 mb-4">
                  Create your first scheduled forecast to automate your forecasting process
                </p>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Schedule Your First Forecast
                </button>
              </div>
            </div>
          ) : (
            <div className="overflow-y-auto h-full">
              <div className="p-6 space-y-4">
                {scheduledForecasts.map((forecast) => (
                  <div
                    key={forecast.id}
                    className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <h3 className="font-semibold text-gray-900">{forecast.name}</h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(forecast.status)}`}>
                            <div className="flex items-center space-x-1">
                              {getStatusIcon(forecast.status)}
                              <span>{forecast.status.charAt(0).toUpperCase() + forecast.status.slice(1)}</span>
                            </div>
                          </span>
                          <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                            {getFrequencyDisplay(forecast.frequency)}
                          </span>
                        </div>
                        
                        {forecast.description && (
                          <p className="text-sm text-gray-600 mb-3">{forecast.description}</p>
                        )}
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500">Next Run:</span>
                            <p className="font-medium text-gray-900">
                              {formatDateTime(forecast.next_run)}
                            </p>
                          </div>
                          <div>
                            <span className="text-gray-500">Last Run:</span>
                            <p className="font-medium text-gray-900">
                              {forecast.last_run ? formatDateTime(forecast.last_run) : 'Never'}
                            </p>
                          </div>
                          <div>
                            <span className="text-gray-500">Success Rate:</span>
                            <p className="font-medium text-gray-900">
                              {getSuccessRate(forecast)}% ({forecast.success_count}/{forecast.run_count})
                            </p>
                          </div>
                          <div>
                            <span className="text-gray-500">Algorithm:</span>
                            <p className="font-medium text-gray-900">
                              {forecast.forecast_config.algorithm.replace('_', ' ')}
                            </p>
                          </div>
                        </div>
                        
                        {forecast.last_error && (
                          <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm">
                            <span className="font-medium text-red-800">Last Error:</span>
                            <p className="text-red-700 mt-1">{forecast.last_error}</p>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex items-center space-x-2 ml-4">
                        <button
                          onClick={() => loadExecutionHistory(forecast.id)}
                          className="flex items-center space-x-1 px-3 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                          <span>History</span>
                        </button>
                        
                        <button
                          onClick={() => handleToggleStatus(forecast)}
                          className={`flex items-center space-x-1 px-3 py-2 rounded-lg transition-colors ${
                            forecast.status === 'active'
                              ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                              : 'bg-green-600 hover:bg-green-700 text-white'
                          }`}
                        >
                          {forecast.status === 'active' ? (
                            <>
                              <Pause className="w-4 h-4" />
                              <span>Pause</span>
                            </>
                          ) : (
                            <>
                              <Play className="w-4 h-4" />
                              <span>Resume</span>
                            </>
                          )}
                        </button>
                        
                        <button
                          onClick={() => handleDeleteForecast(forecast.id, forecast.name)}
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

        {/* Create Modal */}
        {showCreateModal && (
          <CreateScheduledForecastModal
            isOpen={showCreateModal}
            onClose={() => setShowCreateModal(false)}
            onSuccess={() => {
              setShowCreateModal(false);
              loadScheduledForecasts();
            }}
            currentConfig={currentConfig}
          />
        )}

        {/* Execution History Modal */}
        {showExecutionHistory && (
          <ExecutionHistoryModal
            isOpen={!!showExecutionHistory}
            onClose={() => setShowExecutionHistory(null)}
            executions={executions}
            forecastName={scheduledForecasts.find(f => f.id === showExecutionHistory)?.name || ''}
          />
        )}
      </div>
    </div>
  );
};

// Create Scheduled Forecast Modal Component
interface CreateScheduledForecastModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  currentConfig?: ForecastConfig;
}

const CreateScheduledForecastModal: React.FC<CreateScheduledForecastModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
  currentConfig
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [frequency, setFrequency] = useState<'daily' | 'weekly' | 'monthly'>('daily');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      // Set default start date to tomorrow
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      setStartDate(tomorrow.toISOString().slice(0, 16));
    }
  }, [isOpen]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentConfig) {
      setError('No forecast configuration available. Please configure a forecast first.');
      return;
    }

    if (!name.trim()) {
      setError('Schedule name is required');
      return;
    }

    if (!startDate) {
      setError('Start date is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await ApiService.createScheduledForecast({
        name: name.trim(),
        description: description.trim() || undefined,
        forecast_config: currentConfig,
        frequency,
        start_date: startDate,
        end_date: endDate || undefined
      });

      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create scheduled forecast');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-md mx-4 shadow-2xl">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">Schedule Forecast</h3>
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

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Schedule Name *
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., Daily Product A Forecast"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of this scheduled forecast..."
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Frequency *
              </label>
              <select
                value={frequency}
                onChange={(e) => setFrequency(e.target.value as any)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Start Date & Time *
              </label>
              <input
                type="datetime-local"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                End Date & Time (Optional)
              </label>
              <input
                type="datetime-local"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">
                Leave empty to run indefinitely
              </p>
            </div>

            {currentConfig && (
              <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <h4 className="font-medium text-blue-900 mb-2">Forecast Configuration</h4>
                <div className="text-sm text-blue-800 space-y-1">
                  <p><strong>Algorithm:</strong> {currentConfig.algorithm.replace('_', ' ')}</p>
                  <p><strong>Interval:</strong> {currentConfig.interval}</p>
                  <p><strong>Periods:</strong> {currentConfig.historicPeriod}H / {currentConfig.forecastPeriod}F</p>
                </div>
              </div>
            )}

            <div className="flex items-center justify-end space-x-3 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                    <span>Creating...</span>
                  </div>
                ) : (
                  'Create Schedule'
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

// Execution History Modal Component
interface ExecutionHistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  executions: ForecastExecution[];
  forecastName: string;
}

const ExecutionHistoryModal: React.FC<ExecutionHistoryModalProps> = ({
  isOpen,
  onClose,
  executions,
  forecastName
}) => {
  const getExecutionStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'running':
        return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-4xl mx-4 h-[80vh] flex flex-col shadow-2xl">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">
            Execution History: {forecastName}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {executions.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Calendar className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No execution history available</p>
              </div>
            </div>
          ) : (
            <div className="p-6 space-y-4">
              {executions.map((execution) => (
                <div
                  key={execution.id}
                  className="border border-gray-200 rounded-lg p-4"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        {getExecutionStatusIcon(execution.status)}
                        <span className="font-medium text-gray-900">
                          {formatDateTime(execution.execution_time)}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          execution.status === 'success' ? 'bg-green-100 text-green-800' :
                          execution.status === 'failed' ? 'bg-red-100 text-red-800' :
                          'bg-blue-100 text-blue-800'
                        }`}>
                          {execution.status.charAt(0).toUpperCase() + execution.status.slice(1)}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Duration:</span>
                          <p className="font-medium text-gray-900">
                            {formatDuration(execution.duration_seconds)}
                          </p>
                        </div>
                        {execution.result_summary && (
                          <div>
                            <span className="text-gray-500">Result:</span>
                            <p className="font-medium text-gray-900">
                              {execution.result_summary.type === 'multi_forecast' 
                                ? `${execution.result_summary.successful}/${execution.result_summary.total_combinations} successful`
                                : `${execution.result_summary.accuracy}% accuracy`
                              }
                            </p>
                          </div>
                        )}
                      </div>
                      
                      {execution.error_message && (
                        <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm">
                          <span className="font-medium text-red-800">Error:</span>
                          <p className="text-red-700 mt-1">{execution.error_message}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};