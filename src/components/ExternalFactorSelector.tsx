import React, { useState, useEffect } from 'react';
import { TrendingUp, X, RefreshCw, CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react';
import { ApiService } from '../services/api';


interface ValidationResult {
  overall_status: 'good' | 'warning' | 'bad';
  factor_validations: {
    [factorName: string]: {
      status: 'good' | 'warning' | 'bad';
      coverage_percentage: number;
      message: string;
      total_missing?: number;
    };
  };
  recommendations: string[];
}

interface ExternalFactorSelectorProps {
  selectedFactors: string[];
  onFactorsChange: (factors: string[]) => void;
}

export const ExternalFactorSelector: React.FC<ExternalFactorSelectorProps> = ({
  selectedFactors,
  onFactorsChange
}) => {
  const [availableFactors, setAvailableFactors] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [validating, setValidating] = useState(false);

  const validateSelectedFactors = async () => {
    if (selectedFactors.length === 0) return;
    
    setValidating(true);
    try {
      const response = await fetch('/api/validate_external_factors', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(selectedFactors)
      });
      
      if (!response.ok) {
        throw new Error('Validation failed');
      }
      
      const result = await response.json();
      setValidationResult(result);
    } catch (error) {
      console.error('Validation error:', error);
      setValidationResult({
        overall_status: 'bad',
        factor_validations: {},
        recommendations: ['Validation failed. Please try again.']
      });
    } finally {
      setValidating(false);
    }
  };

  useEffect(() => {
    loadAvailableFactors();
  }, []);

  const loadAvailableFactors = async () => {
    setLoading(true);
    try {
      const response = await ApiService.getExternalFactors();
      setAvailableFactors(response.external_factors);
    } catch (error) {
      console.error('Failed to load external factors:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFactorToggle = (factor: string) => {
    if (selectedFactors.includes(factor)) {
      onFactorsChange(selectedFactors.filter(f => f !== factor));
    } else {
      onFactorsChange([...selectedFactors, factor]);
    }
  };

  const handleSelectAll = () => {
    onFactorsChange(availableFactors);
  };

  const handleClearAll = () => {
    onFactorsChange([]);
  };

  if (availableFactors.length === 0 && !loading) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-center">
          <TrendingUp className="w-5 h-5 text-yellow-600 mr-2" />
          <div>
            <p className="text-yellow-800 font-medium">No External Factors Available</p>
            <p className="text-yellow-700 text-sm mt-1">
              Upload external factor data to use as features in forecasting
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <TrendingUp className="w-5 h-5 text-purple-600" />
          <label className="block text-sm font-medium text-gray-700">
            External Factors (Optional)
          </label>
        </div>
        
        {availableFactors.length > 0 && (
          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={handleSelectAll}
              className="text-xs text-purple-600 hover:text-purple-700 font-medium"
            >
              Select All
            </button>
            <span className="text-gray-300">|</span>
            <button
              type="button"
              onClick={handleClearAll}
              className="text-xs text-gray-600 hover:text-gray-700 font-medium"
            >
              Clear All
            </button>
          </div>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin w-5 h-5 border-2 border-purple-500 border-t-transparent rounded-full"></div>
          <span className="ml-2 text-sm text-gray-600">Loading factors...</span>
        </div>
      ) : (
        <div className="space-y-2">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {availableFactors.map((factor) => (
              <label
                key={factor}
                className={`flex items-center p-3 rounded-lg border cursor-pointer transition-colors ${
                  selectedFactors.includes(factor)
                    ? 'bg-purple-50 border-purple-200 text-purple-900'
                    : 'bg-white border-gray-200 hover:bg-gray-50'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedFactors.includes(factor)}
                  onChange={() => handleFactorToggle(factor)}
                  className="sr-only"
                />
                <div className={`flex-shrink-0 w-4 h-4 rounded border-2 mr-3 flex items-center justify-center ${
                  selectedFactors.includes(factor)
                    ? 'bg-purple-600 border-purple-600'
                    : 'border-gray-300'
                }`}>
                  {selectedFactors.includes(factor) && (
                    <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
                <span className="text-sm font-medium">{factor}</span>
              </label>
            ))}
          </div>

          {selectedFactors.length > 0 && (
            <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-lg">
              <p className="text-sm font-medium text-purple-900 mb-2">
                Selected Factors ({selectedFactors.length}):
              </p>
              <div className="flex flex-wrap gap-2">
                {selectedFactors.map((factor) => (
                  <span
                    key={factor}
                    className="inline-flex items-center px-2 py-1 bg-purple-100 text-purple-800 text-xs font-medium rounded-full"
                  >
                    {factor}
                    <button
                      type="button"
                      onClick={() => handleFactorToggle(factor)}
                      className="ml-1 text-purple-600 hover:text-purple-800"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
              
              {/* Validation Button */}
              <div className="mt-3 flex items-center justify-between">
                <button
                  type="button"
                  onClick={validateSelectedFactors}
                  disabled={validating}
                  className="flex items-center space-x-2 px-3 py-1 bg-purple-600 text-white rounded text-sm hover:bg-purple-700 disabled:opacity-50"
                >
                  {validating ? (
                    <RefreshCw className="w-3 h-3 animate-spin" />
                  ) : (
                    <CheckCircle className="w-3 h-3" />
                  )}
                  <span>{validating ? 'Validating...' : 'Validate Coverage'}</span>
                </button>
              </div>
            </div>
          )}

          {/* Validation Results */}
          {validationResult && (
            <div className="mt-3">
              <div className={`border rounded-lg p-3 ${
                validationResult.overall_status === 'good' ? 'bg-green-50 border-green-200' :
                validationResult.overall_status === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                'bg-red-50 border-red-200'
              }`}>
                <div className="flex items-center mb-2">
                  {validationResult.overall_status === 'good' ? (
                    <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
                  ) : validationResult.overall_status === 'warning' ? (
                    <AlertTriangle className="w-4 h-4 text-yellow-600 mr-2" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-red-600 mr-2" />
                  )}
                  <h4 className={`text-sm font-medium ${
                    validationResult.overall_status === 'good' ? 'text-green-900' :
                    validationResult.overall_status === 'warning' ? 'text-yellow-900' :
                    'text-red-900'
                  }`}>
                    Date Coverage Validation
                  </h4>
                </div>
                
                {/* Individual Factor Results */}
                <div className="space-y-2 mb-3">
                  {Object.entries(validationResult.factor_validations).map(([factorName, result]) => (
                    <div key={factorName} className="text-xs">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{factorName}</span>
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          result.status === 'good' ? 'bg-green-100 text-green-800' :
                          result.status === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {result.coverage_percentage}% coverage
                        </span>
                      </div>
                      <p className="text-gray-600 mt-1">{result.message}</p>
                      {result.total_missing && result.total_missing > 0 && (
                        <p className="text-gray-500 mt-1">
                          {result.total_missing} missing dates
                        </p>
                      )}
                    </div>
                  ))}
                </div>
                
                {/* Recommendations */}
                {validationResult.recommendations.length > 0 && (
                  <div>
                    <p className="text-xs font-medium mb-1">Recommendations:</p>
                    <ul className="text-xs list-disc list-inside space-y-1">
                      {validationResult.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="text-xs text-gray-500">
        External factors will be used as additional features in machine learning algorithms to potentially improve forecast accuracy.
      </div>
    </div>
  );
};