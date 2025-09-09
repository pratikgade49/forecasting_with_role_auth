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
import { 
  BarChart3, TrendingUp, TrendingDown, Minus, Target, Award, Brain, 
  Eye, EyeOff, Grid, ChevronDown, ChevronUp, Package, Users, MapPin,
  Download, AlertCircle, CheckCircle, Terminal,
  Save
} from 'lucide-react';
import { MultiForecastResult } from '../services/api';
import { ForecastConfig } from '../services/api';
import { saveAs } from 'file-saver';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface MultiForecastResultsProps {
  result: MultiForecastResult;
  onShowProcessLog?: (processLog: string[]) => void;
  onSaveForecast?: () => void;
}

export const MultiForecastResults: React.FC<MultiForecastResultsProps> = ({ 
  result, 
  onShowProcessLog,
  onSaveForecast
}) => {
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set([0])); // Expand first result by default
  const [showSummaryDetails, setShowSummaryDetails] = useState(false);
  const [sortBy, setSortBy] = useState<'accuracy' | 'combination'>('accuracy');
  const [downloading, setDownloading] = useState(false);

  const toggleResultExpansion = (index: number) => {
    const newExpanded = new Set(expandedResults);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedResults(newExpanded);
  };

  const expandAll = () => {
    setExpandedResults(new Set(result.results.map((_, index) => index)));
  };

  const collapseAll = () => {
    setExpandedResults(new Set());
  };

  // Download Excel handler for multi-forecast results
  const handleDownloadExcel = async () => {
    setDownloading(true);
    try {
      const payload = {
        multiForecastResult: result
      };
      
      const token = localStorage.getItem('access_token');
      const response = await fetch('http://localhost:8000/download_multi_forecast_excel', {
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
      let filename = 'multi_forecast.xlsx';
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?([^";]+)"?/);
        if (match) filename = match[1];
      }
      saveAs(blob, filename);
    } catch (err) {
      console.error('Download error:', err);
      alert('Failed to download Excel file: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setDownloading(false);
    }
  };

  const sortedResults = [...result.results].sort((a, b) => {
    if (sortBy === 'accuracy') {
      return b.accuracy - a.accuracy;
    } else {
      const aCombination = `${a.combination?.product} - ${a.combination?.customer} - ${a.combination?.location}`;
      const bCombination = `${b.combination?.product} - ${b.combination?.customer} - ${b.combination?.location}`;
      return aCombination.localeCompare(bCombination);
    }
  });

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing':
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'decreasing':
        return <TrendingDown className="w-4 h-4 text-red-500" />;
      default:
        return <Minus className="w-4 h-4 text-gray-500" />;
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

  const createChartData = (forecastResult: any) => {
    return {
      labels: [...forecastResult.historicData.map((d: any) => d.period), ...forecastResult.forecastData.map((d: any) => d.period)],
      datasets: [
        {
          label: 'Historical Data',
          data: [...forecastResult.historicData.map((d: any) => d.quantity), ...Array(forecastResult.forecastData.length).fill(null)],
          borderColor: '#3B82F6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          pointBackgroundColor: '#3B82F6',
          pointBorderColor: '#3B82F6',
          pointRadius: 3,
          pointHoverRadius: 5,
          tension: 0.4,
        },
        {
          label: 'Forecast',
          data: [...Array(forecastResult.historicData.length).fill(null), ...forecastResult.forecastData.map((d: any) => d.quantity)],
          borderColor: '#10B981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          pointBackgroundColor: '#10B981',
          pointBorderColor: '#10B981',
          pointRadius: 3,
          pointHoverRadius: 5,
          tension: 0.4,
          borderDash: [5, 5],
        }
      ]
    };
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          usePointStyle: true,
          padding: 15,
          font: { size: 12 }
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
        grid: { display: false },
        ticks: { maxRotation: 45, minRotation: 0, font: { size: 10 } }
      },
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(0, 0, 0, 0.1)' },
        ticks: { 
          callback: (value) => Number(value).toFixed(0),
          font: { size: 10 }
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  return (
    <div className="space-y-6">
      {/* Summary Header */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl shadow-lg p-6 border border-purple-200">
        {/* Download Button */}
        <div className="flex justify-end mb-4">
          <div className="flex items-center space-x-3">
            {/* Save Forecast Button */}
            {onSaveForecast && (
              <button
                onClick={onSaveForecast}
                className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg shadow transition-colors"
              >
                <Save className="w-5 h-5" />
                <span>Save Multi-Forecast</span>
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
              <Download className="w-5 h-5" />
              <span>{downloading ? 'Downloading...' : 'Download All Results'}</span>
            </button>
          </div>
        </div>
        
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Grid className="w-8 h-8 text-purple-600" />
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Multi-Selection Forecast Results</h2>
              <p className="text-purple-700">Generated {result.totalCombinations} forecasts across all combinations</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl">🎯</div>
            <p className="text-sm text-gray-600 mt-1">Multi-Forecast</p>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Combinations</p>
                <p className="text-2xl font-bold text-gray-900">{result.totalCombinations}</p>
              </div>
              <Grid className="w-6 h-6 text-purple-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Successful</p>
                <p className="text-2xl font-bold text-green-600">{result.summary.successfulCombinations}</p>
              </div>
              <CheckCircle className="w-6 h-6 text-green-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Failed</p>
                <p className="text-2xl font-bold text-red-600">{result.summary.failedCombinations}</p>
              </div>
              <AlertCircle className="w-6 h-6 text-red-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Accuracy</p>
                <p className={`text-2xl font-bold ${getAccuracyColor(result.summary.averageAccuracy)}`}>
                  {result.summary.averageAccuracy}%
                </p>
              </div>
              <Target className="w-6 h-6 text-blue-500" />
            </div>
          </div>
        </div>

        {/* Best/Worst Performance */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h4 className="font-medium text-green-900 mb-2 flex items-center">
              <Award className="w-4 h-4 mr-2" />
              Best Performance
            </h4>
            <p className="text-sm text-green-800">
              <strong>{result.summary.bestCombination.combination.product}</strong> → 
              <strong>{result.summary.bestCombination.combination.customer}</strong> → 
              <strong>{result.summary.bestCombination.combination.location}</strong>
            </p>
            <p className="text-lg font-bold text-green-600 mt-1">
              {result.summary.bestCombination.accuracy}% accuracy
            </p>
          </div>

          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <h4 className="font-medium text-red-900 mb-2 flex items-center">
              <AlertCircle className="w-4 h-4 mr-2" />
              Needs Improvement
            </h4>
            <p className="text-sm text-red-800">
              <strong>{result.summary.worstCombination.combination.product}</strong> → 
              <strong>{result.summary.worstCombination.combination.customer}</strong> → 
              <strong>{result.summary.worstCombination.combination.location}</strong>
            </p>
            <p className="text-lg font-bold text-red-600 mt-1">
              {result.summary.worstCombination.accuracy}% accuracy
            </p>
          </div>
        </div>

        {/* Failed Combinations Details */}
        {result.summary.failedCombinations > 0 && (
          <div className="mt-4">
            <button
              onClick={() => setShowSummaryDetails(!showSummaryDetails)}
              className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-800"
            >
              {showSummaryDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              <span>View failed combinations ({result.summary.failedCombinations})</span>
            </button>
            
            {showSummaryDetails && (
              <div className="mt-2 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                <div className="space-y-2">
                  {result.summary.failedDetails.map((failed, index) => (
                    <div key={index} className="text-sm">
                      <span className="font-medium text-yellow-800">{failed.combination}:</span>
                      <span className="text-yellow-700 ml-2">{failed.error}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between bg-white rounded-lg shadow-sm p-4">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'accuracy' | 'combination')}
              className="px-3 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            >
              <option value="accuracy">Accuracy (High to Low)</option>
              <option value="combination">Combination (A-Z)</option>
            </select>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={expandAll}
            className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded transition-colors"
          >
            Expand All
          </button>
          <button
            onClick={collapseAll}
            className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded transition-colors"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Individual Results */}
      <div className="space-y-4">
        {sortedResults.map((forecastResult, index) => {
          const originalIndex = result.results.indexOf(forecastResult);
          const isExpanded = expandedResults.has(originalIndex);
          
          return (
            <div key={originalIndex} className="bg-white rounded-xl shadow-lg border border-gray-200">
              {/* Result Header */}
              <div 
                className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => toggleResultExpansion(originalIndex)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      {isExpanded ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
                      <div className="text-2xl">{getAccuracyIcon(forecastResult.accuracy)}</div>
                    </div>
                    
                    <div>
                      <div className="flex items-center space-x-2 text-sm text-gray-600 mb-1">
                        <Package className="w-4 h-4" />
                        <span>{forecastResult.combination?.product}</span>
                        <span>→</span>
                        <Users className="w-4 h-4" />
                        <span>{forecastResult.combination?.customer}</span>
                        <span>→</span>
                        <MapPin className="w-4 h-4" />
                        <span>{forecastResult.combination?.location}</span>
                      </div>
                      <p className="font-medium text-gray-900">{forecastResult.selectedAlgorithm}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-6">
                    <div className="text-center">
                      <p className="text-sm text-gray-600">Accuracy</p>
                      <p className={`text-lg font-bold ${getAccuracyColor(forecastResult.accuracy)}`}>
                        {forecastResult.accuracy}%
                      </p>
                    </div>
                    
                    <div className="text-center">
                      <p className="text-sm text-gray-600">Trend</p>
                      <div className="flex items-center justify-center space-x-1">
                        {getTrendIcon(forecastResult.trend)}
                        <span className="text-sm font-medium capitalize">{forecastResult.trend}</span>
                      </div>
                    </div>
                    
                    <div className="text-center">
                      <p className="text-sm text-gray-600">MAE</p>
                      <p className="text-lg font-medium text-gray-900">{forecastResult.mae.toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Expanded Content */}
              {isExpanded && (
                <div className="border-t border-gray-200 p-6">
                  {/* Metrics */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="bg-blue-50 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-blue-600">Accuracy</p>
                          <p className="text-xl font-bold text-blue-900">{forecastResult.accuracy}%</p>
                        </div>
                        <Target className="w-6 h-6 text-blue-500" />
                      </div>
                    </div>

                    <div className="bg-orange-50 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-orange-600">MAE</p>
                          <p className="text-xl font-bold text-orange-900">{forecastResult.mae.toFixed(2)}</p>
                        </div>
                        <BarChart3 className="w-6 h-6 text-orange-500" />
                      </div>
                    </div>

                    <div className="bg-purple-50 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-purple-600">RMSE</p>
                          <p className="text-xl font-bold text-purple-900">{forecastResult.rmse.toFixed(2)}</p>
                        </div>
                        <Award className="w-6 h-6 text-purple-500" />
                      </div>
                    </div>
                  </div>

                  {/* Chart */}
                  <div className="bg-gray-50 rounded-lg p-4 mb-6">
                    <h4 className="font-medium text-gray-900 mb-4">
                      Forecast Chart: {forecastResult.combination?.product} → {forecastResult.combination?.customer} → {forecastResult.combination?.location}
                    </h4>
                    <div className="h-64">
                      <Line data={createChartData(forecastResult)} options={chartOptions} />
                    </div>
                  </div>

                  {/* Data Tables */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Historical Data</h4>
                      <div className="bg-gray-50 rounded-lg p-3 max-h-48 overflow-y-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-gray-200">
                              <th className="text-left py-1 font-medium text-gray-700">Period</th>
                              <th className="text-right py-1 font-medium text-gray-700">Quantity</th>
                            </tr>
                          </thead>
                          <tbody>
                            {forecastResult.historicData.map((item: any, idx: number) => (
                              <tr key={idx} className="border-b border-gray-100">
                                <td className="py-1 text-gray-600">{item.period}</td>
                                <td className="py-1 text-right font-medium text-gray-900">{item.quantity.toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Forecast Data</h4>
                      <div className="bg-gray-50 rounded-lg p-3 max-h-48 overflow-y-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-gray-200">
                              <th className="text-left py-1 font-medium text-gray-700">Period</th>
                              <th className="text-right py-1 font-medium text-gray-700">Quantity</th>
                            </tr>
                          </thead>
                          <tbody>
                            {forecastResult.forecastData.map((item: any, idx: number) => (
                              <tr key={idx} className="border-b border-gray-100">
                                <td className="py-1 text-gray-600">{item.period}</td>
                                <td className="py-1 text-right font-medium text-green-600">{item.quantity.toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};