import React, { useState, useEffect } from 'react';
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
import { TrendingUp, Calendar, Target, AlertCircle } from 'lucide-react';
import { ApiService } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface AccuracyTrackerProps {
  configHash?: string;
}

interface AccuracyHistoryPoint {
  algorithm: string;
  forecast_date: string;
  accuracy: number;
  mae: number;
  rmse: number;
  actual_values: number[];
  predicted_values: number[];
}

export const AccuracyTracker: React.FC<AccuracyTrackerProps> = ({ configHash }) => {
  const [accuracyHistory, setAccuracyHistory] = useState<AccuracyHistoryPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [timeRange, setTimeRange] = useState<7 | 30 | 90>(30);
  const [selectedMetric, setSelectedMetric] = useState<'accuracy' | 'mae' | 'rmse'>('accuracy');

  useEffect(() => {
    if (configHash) {
      loadAccuracyHistory();
    }
  }, [configHash, timeRange]);

  const loadAccuracyHistory = async () => {
    if (!configHash) return;
    
    setLoading(true);

    // Resolve issue for Accuracy Tracker
    
    try {
      const history = await ApiService.getAccuracyHistory(configHash, timeRange);
      setAccuracyHistory(history);
    } catch (error) {
      console.error('Failed to load accuracy history:', error);
    } finally {
      setLoading(false);
    }
  };

  // Group data by algorithm
  const groupedData = accuracyHistory.reduce((acc, point) => {
    if (!acc[point.algorithm]) {
      acc[point.algorithm] = [];
    }
    acc[point.algorithm].push(point);
    return acc;
  }, {} as Record<string, AccuracyHistoryPoint[]>);

  // Generate colors for different algorithms
  const algorithmColors = [
    '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
    '#06B6D4', '#84CC16', '#F97316', '#EC4899', '#6B7280'
  ];

  const chartData = {
    labels: accuracyHistory.length > 0 
      ? [...new Set(accuracyHistory.map(point => 
          new Date(point.forecast_date).toLocaleDateString()
        ))].sort()
      : [],
    datasets: Object.entries(groupedData).map(([algorithm, points], index) => ({
      label: algorithm,
      data: points
        .sort((a, b) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime())
        .map(point => ({
          x: new Date(point.forecast_date).toLocaleDateString(),
          y: selectedMetric === 'accuracy' ? point.accuracy : 
             selectedMetric === 'mae' ? point.mae : point.rmse
        })),
      borderColor: algorithmColors[index % algorithmColors.length],
      backgroundColor: algorithmColors[index % algorithmColors.length] + '20',
      pointBackgroundColor: algorithmColors[index % algorithmColors.length],
      pointBorderColor: algorithmColors[index % algorithmColors.length],
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0.4,
      fill: false,
    }))
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
        text: `Forecast ${selectedMetric.toUpperCase()} Over Time`,
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
        callbacks: {
          label: (context) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            const unit = selectedMetric === 'accuracy' ? '%' : '';
            return `${label}: ${value.toFixed(2)}${unit}`;
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
        beginAtZero: selectedMetric === 'accuracy',
        max: selectedMetric === 'accuracy' ? 100 : undefined,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          callback: (value) => {
            const unit = selectedMetric === 'accuracy' ? '%' : '';
            return `${Number(value).toFixed(1)}${unit}`;
          }
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  // Calculate summary statistics
  const calculateSummaryStats = () => {
    if (accuracyHistory.length === 0) return null;

    const latestAccuracy = accuracyHistory[accuracyHistory.length - 1]?.accuracy || 0;
    const avgAccuracy = accuracyHistory.reduce((sum, point) => sum + point.accuracy, 0) / accuracyHistory.length;
    const accuracyTrend = accuracyHistory.length > 1 
      ? latestAccuracy - accuracyHistory[0].accuracy 
      : 0;

    return {
      latestAccuracy,
      avgAccuracy,
      accuracyTrend,
      totalForecasts: accuracyHistory.length
    };
  };

  const summaryStats = calculateSummaryStats();

  if (!configHash) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="text-center py-8">
          <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">Generate a forecast to start tracking accuracy over time</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Target className="w-6 h-6 text-blue-600" />
          <h3 className="text-xl font-semibold text-gray-900">Forecast Accuracy Tracking</h3>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Metric Selector */}
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value as any)}
            className="px-3 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="accuracy">Accuracy (%)</option>
            <option value="mae">MAE</option>
            <option value="rmse">RMSE</option>
          </select>
          
          {/* Time Range Selector */}
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(Number(e.target.value) as any)}
            className="px-3 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
        </div>
      </div>

      {/* Summary Statistics */}
      {summaryStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-blue-600">Latest Accuracy</p>
                <p className="text-2xl font-bold text-blue-900">
                  {summaryStats.latestAccuracy.toFixed(1)}%
                </p>
              </div>
              <Target className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-green-600">Average Accuracy</p>
                <p className="text-2xl font-bold text-green-900">
                  {summaryStats.avgAccuracy.toFixed(1)}%
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-purple-600">Accuracy Trend</p>
                <p className={`text-2xl font-bold ${
                  summaryStats.accuracyTrend >= 0 ? 'text-green-900' : 'text-red-900'
                }`}>
                  {summaryStats.accuracyTrend >= 0 ? '+' : ''}{summaryStats.accuracyTrend.toFixed(1)}%
                </p>
              </div>
              <div className={`w-8 h-8 ${
                summaryStats.accuracyTrend >= 0 ? 'text-green-500' : 'text-red-500'
              }`}>
                <TrendingUp className={summaryStats.accuracyTrend >= 0 ? '' : 'transform rotate-180'} />
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Forecasts</p>
                <p className="text-2xl font-bold text-gray-900">
                  {summaryStats.totalForecasts}
                </p>
              </div>
              <Calendar className="w-8 h-8 text-gray-500" />
            </div>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="h-96">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
          </div>
        ) : accuracyHistory.length > 0 ? (
          <Line data={chartData} options={chartOptions} />
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No accuracy history available for this configuration</p>
              <p className="text-sm text-gray-500 mt-2">
                Generate more forecasts to see accuracy trends over time
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Algorithm Performance Summary */}
      {Object.keys(groupedData).length > 0 && (
        <div className="mt-6">
          <h4 className="font-medium text-gray-900 mb-3">Algorithm Performance Summary</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(groupedData).map(([algorithm, points]) => {
              const avgAccuracy = points.reduce((sum, p) => sum + p.accuracy, 0) / points.length;
              const latestAccuracy = points[points.length - 1]?.accuracy || 0;
              const trend = points.length > 1 ? latestAccuracy - points[0].accuracy : 0;
              
              return (
                <div key={algorithm} className="bg-gray-50 rounded-lg p-3">
                  <h5 className="font-medium text-gray-900 text-sm mb-2">{algorithm}</h5>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Avg Accuracy:</span>
                      <span className="font-medium">{avgAccuracy.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Latest:</span>
                      <span className="font-medium">{latestAccuracy.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Trend:</span>
                      <span className={`font-medium ${trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {trend >= 0 ? '+' : ''}{trend.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};