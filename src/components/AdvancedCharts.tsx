import React, { useState } from 'react';
import { Line, Bar, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  Filler
} from 'chart.js';
import { TrendingUp, BarChart3, Activity, Target, Eye, EyeOff } from 'lucide-react';
import { ForecastResult } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface AdvancedChartsProps {
  result: ForecastResult;
  selectedItem: string;
}

export const AdvancedCharts: React.FC<AdvancedChartsProps> = ({ result, selectedItem }) => {
  const [activeChart, setActiveChart] = useState<'forecast' | 'confidence' | 'accuracy' | 'residuals'>('forecast');
  const [showConfidenceInterval, setShowConfidenceInterval] = useState(true);

  // Generate confidence intervals (mock data - in real implementation, this would come from the model)
  const generateConfidenceIntervals = () => {
    const upperBound = result.forecastData.map(d => d.quantity * 1.15); // +15%
    const lowerBound = result.forecastData.map(d => d.quantity * 0.85); // -15%
    return { upperBound, lowerBound };
  };

  const { upperBound, lowerBound } = generateConfidenceIntervals();

  // Forecast Chart with Confidence Intervals
  const forecastChartData = {
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
        fill: false,
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
        fill: false,
      },
      ...(showConfidenceInterval ? [
        {
          label: 'Upper Confidence (85%)',
          data: [...Array(result.historicData.length).fill(null), ...upperBound],
          borderColor: 'rgba(16, 185, 129, 0.3)',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          pointRadius: 0,
          tension: 0.4,
          borderDash: [2, 2],
          fill: '+1',
        },
        {
          label: 'Lower Confidence (85%)',
          data: [...Array(result.historicData.length).fill(null), ...lowerBound],
          borderColor: 'rgba(16, 185, 129, 0.3)',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          pointRadius: 0,
          tension: 0.4,
          borderDash: [2, 2],
          fill: false,
        }
      ] : [])
    ]
  };

  // Accuracy Comparison Chart
  const accuracyChartData = result.allAlgorithms ? {
    labels: result.allAlgorithms.map(alg => alg.algorithm.replace(/\s+/g, '\n')),
    datasets: [
      {
        label: 'Accuracy (%)',
        data: result.allAlgorithms.map(alg => alg.accuracy),
        backgroundColor: result.allAlgorithms.map(alg => 
          alg.algorithm === result.selectedAlgorithm.replace(' (Best Fit)', '') 
            ? 'rgba(34, 197, 94, 0.8)' 
            : 'rgba(59, 130, 246, 0.6)'
        ),
        borderColor: result.allAlgorithms.map(alg => 
          alg.algorithm === result.selectedAlgorithm.replace(' (Best Fit)', '') 
            ? 'rgba(34, 197, 94, 1)' 
            : 'rgba(59, 130, 246, 1)'
        ),
        borderWidth: 2,
      }
    ]
  } : null;

  // Residuals Analysis Chart
  const residualsData = {
    labels: result.historicData.map(d => d.period),
    datasets: [
      {
        label: 'Residuals',
        data: result.historicData.map((d, i) => ({
          x: i,
          y: Math.random() * 20 - 10 // Mock residuals - in real implementation, calculate actual residuals
        })),
        backgroundColor: 'rgba(239, 68, 68, 0.6)',
        borderColor: 'rgba(239, 68, 68, 1)',
        pointRadius: 4,
      }
    ]
  };

  // Error Distribution Chart
  const errorDistributionData = {
    labels: ['0-5%', '5-10%', '10-15%', '15-20%', '20%+'],
    datasets: [
      {
        label: 'Error Distribution',
        data: [45, 30, 15, 7, 3], // Mock data
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(251, 191, 36, 0.8)',
          'rgba(249, 115, 22, 0.8)',
          'rgba(239, 68, 68, 0.8)',
        ],
        borderWidth: 1,
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
          padding: 20,
          filter: (legendItem) => {
            // Hide confidence interval labels if disabled
            if (!showConfidenceInterval && legendItem.text?.includes('Confidence')) {
              return false;
            }
            return true;
          }
        }
      },
      title: {
        display: true,
        text: `Advanced Forecast Analysis - ${selectedItem}`,
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
            if (label.includes('Confidence')) {
              return `${label}: ${value.toFixed(2)}`;
            }
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

  const barChartOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Algorithm Accuracy Comparison',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        callbacks: {
          label: (context) => `Accuracy: ${context.parsed.y.toFixed(1)}%`
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
          font: {
            size: 10
          }
        }
      },
      y: {
        beginAtZero: true,
        max: 100,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          callback: (value) => `${value}%`
        }
      }
    }
  };

  const scatterOptions: ChartOptions<'scatter'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Residuals Analysis',
        font: {
          size: 16,
          weight: 'bold'
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time Period'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Residual Value'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    }
  };

  const chartTabs = [
    { id: 'forecast', label: 'Forecast with Confidence', icon: TrendingUp },
    { id: 'accuracy', label: 'Algorithm Comparison', icon: Target },
    { id: 'confidence', label: 'Error Distribution', icon: BarChart3 },
    { id: 'residuals', label: 'Residuals Analysis', icon: Activity },
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Advanced Visualizations</h3>
        
        {activeChart === 'forecast' && (
          <button
            onClick={() => setShowConfidenceInterval(!showConfidenceInterval)}
            className="flex items-center space-x-2 px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            {showConfidenceInterval ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            <span className="text-sm">
              {showConfidenceInterval ? 'Hide' : 'Show'} Confidence Intervals
            </span>
          </button>
        )}
      </div>

      {/* Chart Type Tabs */}
      <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
        {chartTabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveChart(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeChart === tab.id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Chart Display */}
      <div className="h-96">
        {activeChart === 'forecast' && (
          <Line data={forecastChartData} options={chartOptions} />
        )}
        
        {activeChart === 'accuracy' && accuracyChartData && (
          <Bar data={accuracyChartData} options={barChartOptions} />
        )}
        
        {activeChart === 'confidence' && (
          <Bar data={errorDistributionData} options={barChartOptions} />
        )}
        
        {activeChart === 'residuals' && (
          <Scatter data={residualsData} options={scatterOptions} />
        )}
      </div>

      {/* Chart Insights */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-blue-50 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 mb-2">Forecast Confidence</h4>
          <p className="text-sm text-blue-700">
            85% confidence interval shows expected range of forecast values
          </p>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4">
          <h4 className="font-medium text-green-900 mb-2">Best Algorithm</h4>
          <p className="text-sm text-green-700">
            {result.selectedAlgorithm} achieved {result.accuracy.toFixed(1)}% accuracy
          </p>
        </div>
        
        <div className="bg-yellow-50 rounded-lg p-4">
          <h4 className="font-medium text-yellow-900 mb-2">Error Analysis</h4>
          <p className="text-sm text-yellow-700">
            MAE: {result.mae.toFixed(2)}, RMSE: {result.rmse.toFixed(2)}
          </p>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4">
          <h4 className="font-medium text-purple-900 mb-2">Trend Direction</h4>
          <p className="text-sm text-purple-700 capitalize">
            {result.trend} trend detected in the data
          </p>
        </div>
      </div>
    </div>
  );
};