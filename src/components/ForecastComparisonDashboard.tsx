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
  X, GitCompare, Target, Award, BarChart3, TrendingUp, TrendingDown, 
  Minus, Trash2, Download, Eye, EyeOff, Filter 
} from 'lucide-react';
import { ForecastResult } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ForecastComparisonDashboardProps {
  isOpen: boolean;
  onClose: () => void;
  forecasts: ForecastResult[];
  onRemoveForecast: (configHash: string) => void;
}

export const ForecastComparisonDashboard: React.FC<ForecastComparisonDashboardProps> = ({
  isOpen,
  onClose,
  forecasts,
  onRemoveForecast
}) => {
  const [showHistorical, setShowHistorical] = useState(true);
  const [showForecast, setShowForecast] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<'accuracy' | 'mae' | 'rmse'>('accuracy');

  if (!isOpen) return null;

  // Generate distinct colors for each forecast
  const colors = [
    '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
    '#06B6D4', '#84CC16', '#F97316', '#EC4899', '#6B7280',
    '#14B8A6', '#F472B6', '#A78BFA', '#FB7185', '#34D399'
  ];

  // Create chart data with all forecasts
  const createComparisonChartData = () => {
    if (forecasts.length === 0) return { labels: [], datasets: [] };

    // Get all unique periods from all forecasts
    const allPeriods = new Set<string>();
    forecasts.forEach(forecast => {
      forecast.historicData.forEach(d => allPeriods.add(d.period));
      forecast.forecastData.forEach(d => allPeriods.add(d.period));
    });
    
    const sortedPeriods = Array.from(allPeriods).sort();

    const datasets: any[] = [];

    forecasts.forEach((forecast, index) => {
      const color = colors[index % colors.length];
      const forecastLabel = `${forecast.selectedAlgorithm} (${forecast.accuracy.toFixed(1)}%)`;

      // Historical data dataset
      if (showHistorical) {
        const historicalData = sortedPeriods.map(period => {
          const dataPoint = forecast.historicData.find(d => d.period === period);
          return dataPoint ? dataPoint.quantity : null;
        });

        datasets.push({
          label: `${forecastLabel} - Historical`,
          data: historicalData,
          borderColor: color,
          backgroundColor: `${color}20`,
          pointBackgroundColor: color,
          pointBorderColor: color,
          pointRadius: 3,
          pointHoverRadius: 5,
          tension: 0.4,
          borderWidth: 2,
        });
      }

      // Forecast data dataset
      if (showForecast) {
        const forecastData = sortedPeriods.map(period => {
          const dataPoint = forecast.forecastData.find(d => d.period === period);
          return dataPoint ? dataPoint.quantity : null;
        });

        datasets.push({
          label: `${forecastLabel} - Forecast`,
          data: forecastData,
          borderColor: color,
          backgroundColor: `${color}20`,
          pointBackgroundColor: color,
          pointBorderColor: color,
          pointRadius: 3,
          pointHoverRadius: 5,
          tension: 0.4,
          borderWidth: 2,
          borderDash: [5, 5],
        });
      }
    });

    return {
      labels: sortedPeriods,
      datasets
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
          font: { size: 11 },
          filter: (legendItem) => {
            if (!showHistorical && legendItem.text?.includes('Historical')) {
              return false;
            }
            if (!showForecast && legendItem.text?.includes('Forecast')) {
              return false;
            }
            return true;
          }
        }
      },
      title: {
        display: true,
        text: `Forecast Comparison (${forecasts.length} forecasts)`,
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
            return `${label}: ${value?.toFixed(2) || 'N/A'}`;
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
          minRotation: 0,
          font: { size: 10 }
        }
      },
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
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
    if (accuracy >= 90) return 'text-green-600';
    if (accuracy >= 80) return 'text-blue-600';
    if (accuracy >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getBestForecast = () => {
    if (forecasts.length === 0) return null;
    return forecasts.reduce((best, current) => 
      current.accuracy > best.accuracy ? current : best
    );
  };

  const getWorstForecast = () => {
    if (forecasts.length === 0) return null;
    return forecasts.reduce((worst, current) => 
      current.accuracy < worst.accuracy ? current : worst
    );
  };

  const getAverageMetrics = () => {
    if (forecasts.length === 0) return { accuracy: 0, mae: 0, rmse: 0 };
    
    const totals = forecasts.reduce((acc, forecast) => ({
      accuracy: acc.accuracy + forecast.accuracy,
      mae: acc.mae + forecast.mae,
      rmse: acc.rmse + forecast.rmse
    }), { accuracy: 0, mae: 0, rmse: 0 });

    return {
      accuracy: totals.accuracy / forecasts.length,
      mae: totals.mae / forecasts.length,
      rmse: totals.rmse / forecasts.length
    };
  };

  const chartData = createComparisonChartData();
  const bestForecast = getBestForecast();
  const worstForecast = getWorstForecast();
  const averageMetrics = getAverageMetrics();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-7xl mx-4 h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <GitCompare className="w-6 h-6 text-orange-600" />
            <h2 className="text-xl font-semibold text-gray-900">Forecast Comparison Dashboard</h2>
            <span className="bg-orange-100 text-orange-800 text-sm px-3 py-1 rounded-full">
              {forecasts.length} forecasts
            </span>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Chart Controls */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowHistorical(!showHistorical)}
                className={`flex items-center space-x-1 px-3 py-1 rounded text-sm transition-colors ${
                  showHistorical ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
                }`}
              >
                {showHistorical ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                <span>Historical</span>
              </button>
              
              <button
                onClick={() => setShowForecast(!showForecast)}
                className={`flex items-center space-x-1 px-3 py-1 rounded text-sm transition-colors ${
                  showForecast ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
                }`}
              >
                {showForecast ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                <span>Forecast</span>
              </button>
            </div>
            
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex flex-col p-6 space-y-6">
          {forecasts.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <GitCompare className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Forecasts to Compare</h3>
                <p className="text-gray-600">
                  Generate some forecasts and add them to comparison to see them here.
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Summary Statistics */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-blue-600">Average Accuracy</p>
                      <p className="text-2xl font-bold text-blue-900">
                        {averageMetrics.accuracy.toFixed(1)}%
                      </p>
                    </div>
                    <Target className="w-8 h-8 text-blue-500" />
                  </div>
                </div>

                <div className="bg-green-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-green-600">Best Forecast</p>
                      <p className="text-lg font-bold text-green-900">
                        {bestForecast?.accuracy.toFixed(1)}%
                      </p>
                      <p className="text-xs text-green-700 truncate">
                        {bestForecast?.selectedAlgorithm}
                      </p>
                    </div>
                    <Award className="w-8 h-8 text-green-500" />
                  </div>
                </div>

                <div className="bg-red-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-red-600">Lowest Accuracy</p>
                      <p className="text-lg font-bold text-red-900">
                        {worstForecast?.accuracy.toFixed(1)}%
                      </p>
                      <p className="text-xs text-red-700 truncate">
                        {worstForecast?.selectedAlgorithm}
                      </p>
                    </div>
                    <BarChart3 className="w-8 h-8 text-red-500" />
                  </div>
                </div>

                <div className="bg-purple-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-purple-600">Forecasts</p>
                      <p className="text-2xl font-bold text-purple-900">{forecasts.length}</p>
                      <p className="text-xs text-purple-700">In comparison</p>
                    </div>
                    <GitCompare className="w-8 h-8 text-purple-500" />
                  </div>
                </div>
              </div>

              {/* Chart */}
              <div className="bg-gray-50 rounded-lg p-4 flex-1">
                <div className="h-96">
                  <Line data={chartData} options={chartOptions} />
                </div>
              </div>

              {/* Comparison Table */}
              <div className="bg-white rounded-lg border border-gray-200">
                <div className="px-4 py-3 border-b border-gray-200">
                  <h3 className="text-lg font-medium text-gray-900">Numerical Comparison</h3>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="text-left py-3 px-4 font-medium text-gray-700">Algorithm</th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">Accuracy</th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">MAE</th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">RMSE</th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">Trend</th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">Color</th>
                        <th className="text-center py-3 px-4 font-medium text-gray-700">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {forecasts
                        .sort((a, b) => b.accuracy - a.accuracy)
                        .map((forecast, index) => (
                          <tr key={forecast.configHash} className="border-b border-gray-100">
                            <td className="py-3 px-4">
                              <div className="flex items-center space-x-2">
                                <span className="font-medium text-gray-900">
                                  {forecast.selectedAlgorithm}
                                </span>
                                {index === 0 && <span className="text-lg">🏆</span>}
                              </div>
                            </td>
                            <td className="py-3 px-4 text-center">
                              <span className={`font-bold ${getAccuracyColor(forecast.accuracy)}`}>
                                {forecast.accuracy.toFixed(1)}%
                              </span>
                            </td>
                            <td className="py-3 px-4 text-center font-medium text-gray-900">
                              {forecast.mae.toFixed(2)}
                            </td>
                            <td className="py-3 px-4 text-center font-medium text-gray-900">
                              {forecast.rmse.toFixed(2)}
                            </td>
                            <td className="py-3 px-4 text-center">
                              <div className="flex items-center justify-center space-x-1">
                                {getTrendIcon(forecast.trend)}
                                <span className="text-sm font-medium capitalize">
                                  {forecast.trend}
                                </span>
                              </div>
                            </td>
                            <td className="py-3 px-4 text-center">
                              <div 
                                className="w-6 h-6 rounded-full mx-auto border-2 border-gray-300"
                                style={{ backgroundColor: colors[forecasts.indexOf(forecast) % colors.length] }}
                              ></div>
                            </td>
                            <td className="py-3 px-4 text-center">
                              <button
                                onClick={() => onRemoveForecast(forecast.configHash || '')}
                                className="text-red-600 hover:text-red-800 transition-colors"
                                title="Remove from comparison"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};