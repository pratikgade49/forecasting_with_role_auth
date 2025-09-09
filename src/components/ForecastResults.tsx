import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { BarChart3, TrendingUp, TrendingDown, Minus, Target, Award, Brain, Eye, EyeOff, Plus, GitCompare } from 'lucide-react';
import { Terminal } from 'lucide-react';
import { Save } from 'lucide-react';
import { ForecastResult, ForecastConfig, API_BASE_URL } from '../services/api';
import { saveAs } from 'file-saver';
import { AdvancedCharts } from './AdvancedCharts';
import { AccuracyTracker } from './AccuracyTracker';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ForecastResultsProps {
  result: ForecastResult;
  forecastBy: string;
  selectedItem: string;
  multiResult?: any; // For multi-forecast mode
  onAddToComparison?: (forecast: ForecastResult) => void;
  comparisonList?: ForecastResult[];
  onShowProcessLog?: (processLog: string[]) => void;
  onSaveForecast?: () => void;
}

export const ForecastResults: React.FC<ForecastResultsProps> = ({ 
  result, 
  forecastBy, 
  selectedItem, 
  onAddToComparison,
  comparisonList = [],
  onShowProcessLog,
  onSaveForecast
}) => {
  const [showAllAlgorithms, setShowAllAlgorithms] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [activeTab, setActiveTab] = useState<'basic' | 'advanced' | 'tracking'>('basic');

  // Download Excel handler
  const handleDownloadExcel = async () => {
    setDownloading(true);
    try {
      // Create the payload with forecastResult only
      const payload = {
        forecastResult: result,
        forecastBy: forecastBy,
        selectedItem: selectedItem
      };
      
      const token = localStorage.getItem('access_token');
      const response = await fetch(`${API_BASE_URL}/download_forecast_excel`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to download Excel');
      }
      
      const blob = await response.blob();
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = 'forecast.xlsx';
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?([^";]+)"?/);
        if (match) filename = match[1];
      }
      saveAs(blob, filename);
    } catch (err) {
      console.error('Download error:', err);
      if (err instanceof Error && err.message.includes('No data found for selected criteria')) {
        alert('No data found for selected criteria.');
      } else {
        alert('Failed to download Excel file: ' + (err instanceof Error ? err.message : 'Unknown error'));
      }
    } finally {
      setDownloading(false);
    }
  };

  const isInComparison = comparisonList.some(f => f.configHash === result.configHash);

  const handleAddToComparison = () => {
    if (onAddToComparison && !isInComparison) {
      onAddToComparison(result);
    }
  };

  const chartData = {
    labels: [...result.historicData.map(d => d.period), ...result.forecastData.map(d => d.period)],
    datasets: [
      {
        label: 'Historical Data',
        data: [...result.historicData.map(d => d.quantity), ...Array(result.forecastData.length).fill(null)],
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        pointBackgroundColor: '#3B82F6',
        pointBorderColor: '#3B82F6',
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4,
      },
      {
        label: 'Forecast',
        data: [...Array(result.historicData.length).fill(null), ...result.forecastData.map(d => d.quantity)],
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        pointBackgroundColor: '#10B981',
        pointBorderColor: '#10B981',
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4,
        borderDash: [5, 5],
      }
    ]
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          usePointStyle: true,
          padding: 20
        }
      },
      title: {
        display: true,
        text: `Forecast for ${selectedItem} using ${result.selectedAlgorithm}`,
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.2)',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: (context) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            return `${label}: ${value.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        },
        ticks: {
          maxRotation: 45,
          minRotation: 0
        }
      },
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          callback: (value) => Number(value).toFixed(0)
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  const getTrendIcon = () => {
    switch (result.trend) {
      case 'increasing':
        return <TrendingUp className="w-5 h-5 text-green-500" />;
      case 'decreasing':
        return <TrendingDown className="w-5 h-5 text-red-500" />;
      default:
        return <Minus className="w-5 h-5 text-gray-500" />;
    }
  };

  const getTrendColor = () => {
    switch (result.trend) {
      case 'increasing':
        return 'text-green-600 bg-green-50';
      case 'decreasing':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 90) return 'text-green-600 bg-green-50';
    if (accuracy >= 80) return 'text-blue-600 bg-blue-50';
    if (accuracy >= 70) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getAccuracyIcon = (accuracy: number) => {
    if (accuracy >= 90) return '🏆';
    if (accuracy >= 80) return '🥇';
    if (accuracy >= 70) return '🥈';
    return '🥉';
  };

  return (
    <div className="space-y-6">
      {/* Download Button */}
      <div className="flex justify-end space-x-3">
        {/* Add to Comparison Button */}
        {onAddToComparison && (
          <button
            onClick={handleAddToComparison}
            disabled={isInComparison}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg shadow transition-colors ${
              isInComparison
                ? 'bg-gray-100 text-gray-500 cursor-not-allowed'
                : 'bg-orange-600 hover:bg-orange-700 text-white'
            }`}
          >
            {isInComparison ? (
              <>
                <GitCompare className="w-5 h-5" />
                <span>Added to Comparison</span>
              </>
            ) : (
              <>
                <Plus className="w-5 h-5" />
                <span>Add to Comparison</span>
              </>
            )}
          </button>
        )}
        
        {/* Save Forecast Button */}
        {onSaveForecast && (
          <button
            onClick={onSaveForecast}
            className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg shadow transition-colors"
          >
            <Save className="w-5 h-5" />
            <span>Save Forecast</span>
          </button>
        )}
        
        {/* Process Log Button */}
        {onShowProcessLog && result.processLog && result.processLog.length > 0 && (
          <button
            onClick={() => onShowProcessLog(result.processLog || [])}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow transition-colors"
          >
            <Terminal className="w-5 h-5" />
            <span>View Process Log ({result.processLog.length})</span>
          </button>
        )}
        
        <button
          onClick={handleDownloadExcel}
          disabled={downloading}
          className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg shadow transition-colors disabled:opacity-50"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V4" /></svg>
          <span>{downloading ? 'Downloading...' : 'Download Excel'}</span>
        </button>
      </div>

      {/* Selected Algorithm Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl shadow-lg p-6 border border-blue-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Brain className="w-8 h-8 text-blue-600" />
            <div>
              <h3 className="text-xl font-bold text-gray-900">Selected Algorithm</h3>
              <p className="text-blue-700 font-medium">{result.selectedAlgorithm}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl">{getAccuracyIcon(result.accuracy)}</div>
            <p className="text-sm text-gray-600 mt-1">Best Performance</p>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Forecast Accuracy</p>
              <p className={`text-2xl font-bold ${getAccuracyColor(result.accuracy)}`}>
                {result.accuracy.toFixed(1)}%
              </p>
            </div>
            <Target className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Mean Absolute Error</p>
              <p className="text-2xl font-bold text-gray-900">{result.mae.toFixed(2)}</p>
            </div>
            <BarChart3 className="w-8 h-8 text-orange-500" />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Root Mean Square Error</p>
              <p className="text-2xl font-bold text-gray-900">{result.rmse.toFixed(2)}</p>
            </div>
            <Award className="w-8 h-8 text-purple-500" />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Trend Direction</p>
              <p className={`text-xl font-bold capitalize ${getTrendColor()}`}>
                {result.trend}
              </p>
            </div>
            {getTrendIcon()}
          </div>
        </div>
      </div>

      {/* Chart */}
      {/* Chart Tabs */}
      <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
        <button
          onClick={() => setActiveTab('basic')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'basic'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Basic Chart
        </button>
        <button
          onClick={() => setActiveTab('advanced')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'advanced'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Advanced Visualizations
        </button>
        <button
          onClick={() => setActiveTab('tracking')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'tracking'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Accuracy Tracking
        </button>
      </div>

      {/* Chart Display */}
      {activeTab === 'basic' && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="h-96">
            <Line data={chartData} options={chartOptions} />
          </div>
        </div>
      )}

      {activeTab === 'advanced' && (
        <AdvancedCharts result={result} selectedItem={selectedItem} />
      )}

      {activeTab === 'tracking' && (
        <AccuracyTracker configHash={result.configHash} />
      )}

      {/* All Algorithms Comparison (if Best Fit was used) */}
      {result.allAlgorithms && result.allAlgorithms.length > 0 && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Brain className="w-6 h-6 text-purple-600" />
              <h3 className="text-xl font-semibold text-gray-900">Algorithm Comparison</h3>
              <span className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full font-medium">
                Best Fit Analysis
              </span>
            </div>
            <button
              onClick={() => setShowAllAlgorithms(!showAllAlgorithms)}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              {showAllAlgorithms ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              <span className="text-sm font-medium">
                {showAllAlgorithms ? 'Hide Details' : 'Show Details'}
              </span>
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Algorithm</th>
                  <th className="text-center py-3 px-4 font-medium text-gray-700">Accuracy</th>
                  <th className="text-center py-3 px-4 font-medium text-gray-700">MAE</th>
                  <th className="text-center py-3 px-4 font-medium text-gray-700">RMSE</th>
                  <th className="text-center py-3 px-4 font-medium text-gray-700">Trend</th>
                  <th className="text-center py-3 px-4 font-medium text-gray-700">Rank</th>
                </tr>
              </thead>
              <tbody>
                {result.allAlgorithms
                  .sort((a, b) => b.accuracy - a.accuracy)
                  .map((algorithm, index) => (
                    <tr key={algorithm.algorithm} className={`border-b border-gray-100 ${
                      index === 0 ? 'bg-green-50' : ''
                    }`}>
                      <td className="py-3 px-4">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium text-gray-900">{algorithm.algorithm}</span>
                          {index === 0 && <span className="text-lg">👑</span>}
                        </div>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`font-bold ${getAccuracyColor(algorithm.accuracy)}`}>
                          {algorithm.accuracy.toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center font-medium text-gray-900">
                        {algorithm.mae.toFixed(2)}
                      </td>
                      <td className="py-3 px-4 text-center font-medium text-gray-900">
                        {algorithm.rmse.toFixed(2)}
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`capitalize px-2 py-1 rounded-full text-xs font-medium ${getTrendColor()}`}>
                          {algorithm.trend}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className="font-bold text-gray-700">#{index + 1}</span>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {showAllAlgorithms && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">Algorithm Performance Insights</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
                <div>
                  <p><strong>Best Accuracy:</strong> {result.allAlgorithms[0]?.algorithm} ({result.allAlgorithms[0]?.accuracy.toFixed(1)}%)</p>
                  <p><strong>Lowest MAE:</strong> {result.allAlgorithms.reduce((min, alg) => alg.mae < min.mae ? alg : min).algorithm}</p>
                </div>
                <div>
                  <p><strong>Lowest RMSE:</strong> {result.allAlgorithms.reduce((min, alg) => alg.rmse < min.rmse ? alg : min).algorithm}</p>
                  <p><strong>Total Algorithms Tested:</strong> {result.allAlgorithms.length}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Data Tables */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Historical Data Table */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Historical Data</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 font-medium text-gray-700">Period</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-700">Quantity</th>
                </tr>
              </thead>
              <tbody>
                {result.historicData.map((item, index) => (
                  <tr key={index} className="border-b border-gray-100">
                    <td className="py-2 px-3 text-gray-600">{item.period}</td>
                    <td className="py-2 px-3 text-right font-medium text-gray-900">
                      {item.quantity.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Forecast Data Table */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Forecast Data</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 font-medium text-gray-700">Period</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-700">Quantity</th>
                </tr>
              </thead>
              <tbody>
                {result.forecastData.map((item, index) => (
                  <tr key={index} className="border-b border-gray-100">
                    <td className="py-2 px-3 text-gray-600">{item.period}</td>
                    <td className="py-2 px-3 text-right font-medium text-green-600">
                      {item.quantity.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};