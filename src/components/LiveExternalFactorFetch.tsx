import React, { useState, useEffect } from 'react';
import { TrendingUp, Download, AlertCircle, CheckCircle, Info, Calendar, X, RefreshCw } from 'lucide-react';
import { ApiService, FredDataRequest, FredSeriesInfo } from '../services/api';

interface LiveExternalFactorFetchProps {
  onFetchSuccess: () => void;
}

export const LiveExternalFactorFetch: React.FC<LiveExternalFactorFetchProps> = ({ onFetchSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [seriesIds, setSeriesIds] = useState<string>('');
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [showSeriesInfo, setShowSeriesInfo] = useState(false);
  const [seriesInfo, setSeriesInfo] = useState<FredSeriesInfo | null>(null);
  const [loadingSeriesInfo, setLoadingSeriesInfo] = useState(false);

  // Set default date range (last 5 years)
  useEffect(() => {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setFullYear(endDate.getFullYear() - 5);
    
    setEndDate(endDate.toISOString().split('T')[0]);
    setStartDate(startDate.toISOString().split('T')[0]);
  }, []);

  const loadSeriesInfo = async () => {
    setLoadingSeriesInfo(true);
    try {
      const info = await ApiService.getFredSeriesInfo();
      setSeriesInfo(info);
    } catch (error) {
      console.error('Failed to load FRED series info:', error);
    } finally {
      setLoadingSeriesInfo(false);
    }
  };

  const handleShowSeriesInfo = () => {
    if (!seriesInfo) {
      loadSeriesInfo();
    }
    setShowSeriesInfo(true);
  };

  const handleFetchData = async () => {
    if (!seriesIds.trim()) {
      setError('Please enter at least one FRED series ID');
      return;
    }

    console.log('DEBUG: Starting FRED data fetch...');
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const seriesArray = seriesIds.split(',').map(id => id.trim()).filter(id => id);
      console.log('DEBUG: Series to fetch:', seriesArray);
      
      const request: FredDataRequest = {
        series_ids: seriesArray,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      };

      console.log('DEBUG: Sending request to backend:', request);
      const response = await ApiService.fetchFredData(request);
      console.log('DEBUG: Received response from backend:', response);
      setResult(response);
      onFetchSuccess();
    } catch (err) {
      console.error('DEBUG: Error during FRED data fetch:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch FRED data');
    } finally {
      setLoading(false);
    }
  };

  const addSeriesToInput = (seriesId: string) => {
    const currentSeries = seriesIds.split(',').map(id => id.trim()).filter(id => id);
    if (!currentSeries.includes(seriesId)) {
      const newSeries = currentSeries.length > 0 ? `${seriesIds}, ${seriesId}` : seriesId;
      setSeriesIds(newSeries);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Download className="w-6 h-6 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Fetch Live Economic Data</h3>
        </div>
        
        <button
          onClick={handleShowSeriesInfo}
          className="flex items-center space-x-2 px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors"
        >
          <Info className="w-4 h-4" />
          <span className="text-sm">Popular Series</span>
        </button>
      </div>

      <div className="space-y-4">
        {/* Series IDs Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            FRED Series IDs (comma-separated)
          </label>
          <textarea
            value={seriesIds}
            onChange={(e) => setSeriesIds(e.target.value)}
            placeholder="e.g., GDP, CPIAUCSL, UNRATE, FEDFUNDS"
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <p className="text-xs text-gray-500 mt-1">
            Enter FRED series IDs separated by commas. Visit{' '}
            <a href="https://fred.stlouisfed.org" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
              FRED website
            </a>{' '}
            to find more series.
          </p>
        </div>

        {/* Date Range */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Start Date (Optional)
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              End Date (Optional)
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* Fetch Button */}
        <button
          onClick={handleFetchData}
          disabled={loading || !seriesIds.trim()}
          className="w-full flex items-center justify-center px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <>
              <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
              Fetching Data...
            </>
          ) : (
            <>
              <Download className="w-5 h-5 mr-2" />
              Fetch Live Data
            </>
          )}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center">
            <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Success Result */}
      {result && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center mb-3">
            <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
            <h4 className="font-medium text-green-900">Data Fetch Completed</h4>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">{result.series_processed}</p>
              <p className="text-sm text-green-700">Series Processed</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">{result.inserted}</p>
              <p className="text-sm text-blue-700">Records Inserted</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-yellow-600">{result.duplicates}</p>
              <p className="text-sm text-yellow-700">Duplicates Skipped</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-purple-600">
                {result.series_details.filter((s: any) => s.status === 'success').length}
              </p>
              <p className="text-sm text-purple-700">Successful</p>
            </div>
          </div>

          {/* Series Details */}
          <div className="space-y-2">
            <h5 className="font-medium text-gray-900">Series Details:</h5>
            {result.series_details.map((series: any, index: number) => (
              <div
                key={index}
                className={`p-3 rounded-lg border ${
                  series.status === 'success'
                    ? 'bg-green-50 border-green-200'
                    : 'bg-red-50 border-red-200'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{series.series_id}</span>
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    series.status === 'success'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {series.status}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-1">{series.message}</p>
                {series.status === 'success' && (
                  <p className="text-xs text-gray-500 mt-1">
                    Inserted: {series.inserted} records
                    {series.duplicates > 0 && `, Duplicates: ${series.duplicates}`}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Series Info Modal */}
      {showSeriesInfo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl w-full max-w-4xl mx-4 max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Popular FRED Series</h3>
                <button
                  onClick={() => setShowSeriesInfo(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {loadingSeriesInfo ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="w-6 h-6 animate-spin text-blue-600" />
                </div>
              ) : seriesInfo ? (
                <div className="space-y-6">
                  {Object.entries(seriesInfo.series).map(([category, series]) => (
                    <div key={category}>
                      <h4 className="font-medium text-gray-900 mb-3">{category}</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {Object.entries(series).map(([seriesId, description]) => (
                          <div
                            key={seriesId}
                            className="p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors"
                            onClick={() => addSeriesToInput(seriesId)}
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium text-blue-600">{seriesId}</span>
                              <button className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                                Add
                              </button>
                            </div>
                            <p className="text-sm text-gray-600 mt-1">{description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                  
                  <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-sm text-blue-800">
                      <strong>Note:</strong> {seriesInfo.note}
                    </p>
                  </div>
                </div>
              ) : (
                <p className="text-gray-600">Failed to load series information.</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Information Box */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="font-medium text-blue-900 mb-2">About FRED Data</h4>
        <div className="text-sm text-blue-800 space-y-1">
          <p>• FRED (Federal Reserve Economic Data) provides free economic data from the St. Louis Fed</p>
          <p>• Data is automatically stored in your database and can be used as external factors in forecasting</p>
          <p>• Popular series include GDP, inflation rates, unemployment, interest rates, and more</p>
          <p>• <strong>Setup Required:</strong> Set FRED_API_KEY environment variable on your backend server</p>
        </div>
      </div>
    </div>
  );
};