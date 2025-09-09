import React from 'react';
import { TrendingUp, Calendar, Clock, Package, Users, MapPin, Target, Brain, Zap, Grid, List, X, Search, CheckSquare, Square } from 'lucide-react';
import { ForecastConfig } from '../services/api';
import { ExternalFactorSelector } from './ExternalFactorSelector';
import axios from 'axios';

interface ForecastConfigurationProps {
  config: ForecastConfig;
  onChange: (config: ForecastConfig) => void;
  productOptions: string[];
  customerOptions: string[];
  locationOptions: string[];
}

export const ForecastConfiguration: React.FC<ForecastConfigurationProps> = ({
  config,
  onChange,
  productOptions,
  customerOptions,
  locationOptions
}) => {
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [multiSelectMode, setMultiSelectMode] = React.useState(false);
  const [simpleMultiSelect, setSimpleMultiSelect] = React.useState(false);
  const [advancedMode, setAdvancedMode] = React.useState(false);

  // Search states for multi-select
  const [productSearch, setProductSearch] = React.useState('');
  const [customerSearch, setCustomerSearch] = React.useState('');
  const [locationSearch, setLocationSearch] = React.useState('');
  const [simpleSearch, setSimpleSearch] = React.useState('');

  const toggleAdvancedMode = () => {
    const newMode = !advancedMode;
    setAdvancedMode(newMode);
  };

  const algorithms = [
    // Statistical Methods
    { value: 'linear_regression', label: 'Linear Regression', icon: '📈', description: 'Simple trend-based forecasting', category: 'Statistical' },
    { value: 'polynomial_regression', label: 'Polynomial Regression', icon: '📊', description: 'Captures non-linear patterns', category: 'Statistical' },
    { value: 'exponential_smoothing', label: 'Exponential Smoothing', icon: '🌊', description: 'Weighted recent observations', category: 'Statistical' },
    { value: 'ses', label: 'Simple Exponential Smoothing', icon: '🌀', description: 'Optimized exponential smoothing', category: 'Statistical' },
    { value: 'holt_winters', label: 'Holt-Winters', icon: '❄️', description: 'Handles trend and seasonality', category: 'Statistical' },
    { value: 'damped_trend', label: 'Damped Trend', icon: '📉', description: 'Exponential smoothing with damped trend', category: 'Statistical' },
    { value: 'arima', label: 'ARIMA', icon: '🔄', description: 'Autoregressive integrated model', category: 'Statistical' },
    { value: 'sarima', label: 'SARIMA', icon: '🔄', description: 'Seasonal ARIMA with better seasonality', category: 'Statistical' },
    { value: 'theta_method', label: 'Theta Method', icon: '🎯', description: 'Simple but effective statistical method', category: 'Statistical' },
    { value: 'drift_method', label: 'Drift Method', icon: '📈', description: 'Linear trend extrapolation', category: 'Statistical' },
    { value: 'naive_seasonal', label: 'Naive Seasonal', icon: '🔁', description: 'Simple seasonal pattern repetition', category: 'Statistical' },
    { value: 'prophet_like', label: 'Prophet-like', icon: '🔮', description: 'Trend + seasonality decomposition', category: 'Statistical' },

    // Machine Learning Methods
    { value: 'random_forest', label: 'Random Forest', icon: '🌳', description: 'Machine learning ensemble', category: 'Machine Learning' },
    { value: 'xgboost', label: 'XGBoost', icon: '🚀', description: 'Gradient boosting algorithm', category: 'Machine Learning' },
    { value: 'svr', label: 'Support Vector Regression', icon: '🎯', description: 'SVM for regression tasks', category: 'Machine Learning' },
    { value: 'knn', label: 'K-Nearest Neighbors', icon: '👥', description: 'Instance-based learning', category: 'Machine Learning' },
    { value: 'gaussian_process', label: 'Gaussian Process', icon: '📊', description: 'Probabilistic approach with uncertainty', category: 'Machine Learning' },
    { value: 'neural_network', label: 'Neural Network (MLP)', icon: '🧠', description: 'Multi-layer perceptron', category: 'Machine Learning' },
    { value: 'lstm_like', label: 'LSTM-like Network', icon: '🔗', description: 'Neural network with memory', category: 'Machine Learning' },

    // Specialized Methods
    { value: 'seasonal_decomposition', label: 'Seasonal Decomposition', icon: '🗓️', description: 'Separates trend and seasonality', category: 'Specialized' },
    { value: 'moving_average', label: 'Moving Average', icon: '📉', description: 'Smoothed historical average', category: 'Specialized' },
    { value: 'croston', label: "Croston's Method", icon: '⚡', description: 'For intermittent/sparse demand', category: 'Specialized' },

    // Auto-Select (Featured)
    { value: 'best_fit', label: 'Best Fit (Auto-Select)', icon: '🎯', description: 'Automatically selects the best performing algorithm', category: 'Featured', featured: true },

    // Category-based selections
    { value: 'best_statistical', label: 'Best Statistical Method', icon: '📊', description: 'Automatically selects the best statistical algorithm', category: 'Category-Based', featured: true },
    { value: 'best_ml', label: 'Best Machine Learning Method', icon: '🤖', description: 'Automatically selects the best ML algorithm', category: 'Category-Based', featured: true },
    { value: 'best_specialized', label: 'Best Specialized Method', icon: '⚙️', description: 'Automatically selects the best specialized algorithm', category: 'Category-Based', featured: true },
  ];

  const handleChange = (field: keyof ForecastConfig, value: any) => {
    const newConfig = { ...config, [field]: value };

    // Reset selected item when forecast type changes
    if (field === 'forecastBy') {
      newConfig.selectedItem = '';
      newConfig.selectedItems = [];
      newConfig.selectedProduct = '';
      newConfig.selectedCustomer = '';
      newConfig.selectedLocation = '';
      newConfig.selectedProducts = [];
      newConfig.selectedCustomers = [];
      newConfig.selectedLocations = [];
      // Reset search when dimension changes
      setSimpleSearch('');
    }

    onChange(newConfig);
  };

  const handleMultiSelectChange = (field: 'selectedProducts' | 'selectedCustomers' | 'selectedLocations', value: string) => {
    const currentValues = config[field] || [];
    const newValues = currentValues.includes(value)
      ? currentValues.filter(v => v !== value)
      : [...currentValues, value];

    handleChange(field, newValues);
  };

  const handleSimpleMultiSelectChange = (value: string) => {
    const currentValues = config.selectedItems || [];
    const newValues = currentValues.includes(value)
      ? currentValues.filter(v => v !== value)
      : [...currentValues, value];

    handleChange('selectedItems', newValues);
  };

  // Select All functionality
  const handleSelectAll = (field: 'selectedProducts' | 'selectedCustomers' | 'selectedLocations', options: string[]) => {
    const currentValues = config[field] || [];
    const filteredOptions = getFilteredOptions(field, options);

    if (currentValues.length === filteredOptions.length) {
      // All are selected, so deselect all
      handleChange(field, []);
    } else {
      // Select all filtered options
      handleChange(field, filteredOptions);
    }
  };

  const handleSimpleSelectAll = () => {
    const options = getOptionsForType();
    const filteredOptions = getFilteredSimpleOptions(options);
    const currentValues = config.selectedItems || [];

    if (currentValues.length === filteredOptions.length) {
      // All are selected, so deselect all
      handleChange('selectedItems', []);
    } else {
      // Select all filtered options
      handleChange('selectedItems', filteredOptions);
    }
  };

  // Filter functions
  const getFilteredOptions = (field: 'selectedProducts' | 'selectedCustomers' | 'selectedLocations', options: string[]) => {
    const searchTerm = field === 'selectedProducts' ? productSearch : 
                     field === 'selectedCustomers' ? customerSearch : locationSearch;

    return options.filter(option => 
      option.toLowerCase().includes(searchTerm.toLowerCase())
    );
  };

  const getFilteredSimpleOptions = (options: string[]) => {
    return options.filter(option => 
      option.toLowerCase().includes(simpleSearch.toLowerCase())
    );
  };

  const toggleMultiSelectMode = () => {
    const newMode = !multiSelectMode;
    setMultiSelectMode(newMode);

    // Update config
    const newConfig = { 
      ...config, 
      multiSelect: newMode,
      // Clear conflicting selections when switching modes
      selectedItem: '',
      selectedItems: [],
      selectedProduct: '',
      selectedCustomer: '',
      selectedLocation: '',
      selectedProducts: [],
      selectedCustomers: [],
      selectedLocations: []
    };

    if (newMode) {
      // Switching to multi-select mode
      // Initialize empty arrays for multi-select
      newConfig.selectedProducts = [];
      newConfig.selectedCustomers = [];
      newConfig.selectedLocations = [];
      setShowAdvanced(true); // Force advanced mode for multi-select
      setSimpleMultiSelect(false); // Disable simple multi-select
      setShowAdvanced(true);
    } else {
      // Switching out of multi-select mode - reset all advanced states
      setShowAdvanced(false);
      setAdvancedMode(false);
      setSimpleMultiSelect(false);
    }

    // Clear search terms when switching modes
    setProductSearch('');
    setCustomerSearch('');
    setLocationSearch('');
    setSimpleSearch('');

    onChange(newConfig);
  };

  const toggleSimpleMultiSelect = () => {
    const newMode = !simpleMultiSelect;
    setSimpleMultiSelect(newMode);

    const newConfig = { ...config };

    if (newMode) {
      // Switching to simple multi-select
      newConfig.selectedItems = config.selectedItem ? [config.selectedItem] : [];
      newConfig.selectedItem = '';
      setMultiSelectMode(false); // Disable advanced multi-select
      setShowAdvanced(false); // Disable advanced mode
    } else {
      // Switching back to single select
      newConfig.selectedItems = [];
    }

    // Clear search when switching modes
    setSimpleSearch('');

    onChange(newConfig);
  };

  const getOptionsForType = () => {
    switch (config.forecastBy) {
      case 'product':
        return productOptions;
      case 'customer':
        return customerOptions;
      case 'location':
        return locationOptions;
      default:
        return [];
    }
  };

  const getForecastTypeIcon = () => {
    switch (config.forecastBy) {
      case 'product':
        return <Package className="w-4 h-4" />;
      case 'customer':
        return <Users className="w-4 h-4" />;
      case 'location':
        return <MapPin className="w-4 h-4" />;
      default:
        return null;
    }
  };

  const [filteredOptions, setFilteredOptions] = React.useState({
    products: productOptions,
    customers: customerOptions,
    locations: locationOptions
  });

  React.useEffect(() => {
    if (!config.multiSelect) {
      setFilteredOptions({
        products: productOptions,
        customers: customerOptions,
        locations: locationOptions
      });
      return;
    }

    const fetchFilteredOptions = async () => {
      try {
        const token = localStorage.getItem('access_token');
        const headers: any = {};
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }

        // Fetch products filtered by selected customers and locations
        const productsResponse = await axios.post('http://localhost:8000/database/filtered_options', {
          selectedCustomers: config.selectedCustomers || [],
          selectedLocations: config.selectedLocations || []
        }, { headers });

        // Fetch customers filtered by selected products and locations
        const customersResponse = await axios.post('http://localhost:8000/database/filtered_options', {
          selectedProducts: config.selectedProducts || [],
          selectedLocations: config.selectedLocations || []
        }, { headers });

        // Fetch locations filtered by selected products and customers
        const locationsResponse = await axios.post('http://localhost:8000/database/filtered_options', {
          selectedProducts: config.selectedProducts || [],
          selectedCustomers: config.selectedCustomers || []
        }, { headers });

        setFilteredOptions({
          products: productsResponse.data.products || [],
          customers: customersResponse.data.customers || [],
          locations: locationsResponse.data.locations || []
        });
      } catch (error) {
        console.error('Error fetching filtered options:', error);
        // Fallback to full options on error
        setFilteredOptions({
          products: productOptions,
          customers: customerOptions,
          locations: locationOptions
        });
      }
    };

    fetchFilteredOptions();
  }, [config.selectedProducts, config.selectedCustomers, config.selectedLocations, config.multiSelect, productOptions, customerOptions, locationOptions]);

  // Enhanced Multi-Select Component with Search and Select All
  const MultiSelectBox: React.FC<{
    title: string;
    icon: React.ReactNode;
    options: string[];
    selectedValues: string[];
    onToggle: (value: string) => void;
    onSelectAll: () => void;
    searchValue: string;
    onSearchChange: (value: string) => void;
    required?: boolean;
  }> = ({ title, icon, options, selectedValues, onToggle, onSelectAll, searchValue, onSearchChange, required = false }) => {
    const filteredOptions = options.filter(option => 
      option.toLowerCase().includes(searchValue.toLowerCase())
    );

    const allFilteredSelected = filteredOptions.length > 0 && filteredOptions.every(option => selectedValues.includes(option));
    const someFilteredSelected = filteredOptions.some(option => selectedValues.includes(option));

    return (
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {icon}
          {title}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>

        <div className="border border-gray-300 rounded-lg">
          {/* Search and Select All Header */}
          <div className="p-2 border-b border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between mb-1">
              <button
                type="button"
                onClick={onSelectAll}
                className="flex items-center space-x-1 text-xs font-medium text-purple-600 hover:text-purple-700 transition-colors"
              >
                {allFilteredSelected ? (
                  <CheckSquare className="w-3 h-3" />
                ) : someFilteredSelected ? (
                  <div className="w-3 h-3 border-2 border-purple-600 rounded bg-purple-100 flex items-center justify-center">
                    <div className="w-1 h-1 bg-purple-600 rounded"></div>
                  </div>
                ) : (
                  <Square className="w-3 h-3" />
                )}
                <span className="text-xs">
                  {allFilteredSelected ? 'Deselect All' : 'Select All'}
                  {searchValue && ` (${filteredOptions.length})`}
                </span>
              </button>

              <span className="text-xs text-gray-500">
                {selectedValues.length} of {options.length} selected
              </span>
            </div>

            {/* Search Input */}
            <div className="relative">
              <Search className="w-3 h-3 text-gray-400 absolute left-2 top-1/2 transform -translate-y-1/2" />
              <input
                type="text"
                placeholder={`Search ${title.toLowerCase()}...`}
                value={searchValue}
                onChange={(e) => onSearchChange(e.target.value)}
                className="w-full pl-6 pr-2 py-1.5 border border-gray-300 rounded text-xs focus:ring-1 focus:ring-purple-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Options List */}
          <div className="p-2 max-h-32 overflow-y-auto">
            {filteredOptions.length > 0 ? (
              <div className="space-y-1">
                {filteredOptions.map((option) => (
                  <label key={option} className="flex items-center space-x-1 cursor-pointer hover:bg-gray-50 p-1 rounded">
                    <input
                      type="checkbox"
                      checked={selectedValues.includes(option)}
                      onChange={() => onToggle(option)}
                      className="rounded border-gray-300 text-purple-600 focus:ring-purple-500 w-3 h-3"
                    />
                    <span className="text-xs text-gray-700 flex-1">{option}</span>
                  </label>
                ))}
              </div>
            ) : (
              <div className="text-center py-3">
                <Search className="w-6 h-6 text-gray-400 mx-auto mb-1" />
                <p className="text-xs text-gray-500">
                  {searchValue ? `No ${title.toLowerCase()} found matching "${searchValue}"` : `No ${title.toLowerCase()} available`}
                </p>
              </div>
            )}
          </div>

          {/* Selected Items Summary */}
          {selectedValues.length > 0 && (
            <div className="p-2 border-t border-gray-200 bg-gray-50">
              <div className="flex flex-wrap gap-1">
                {selectedValues.slice(0, 3).map((item) => (
                  <span
                    key={item}
                    className="inline-flex items-center px-1.5 py-0.5 bg-purple-100 text-purple-800 text-xs font-medium rounded-full"
                  >
                    {item}
                    <button
                      type="button"
                      onClick={() => onToggle(item)}
                      className="ml-1 text-purple-600 hover:text-purple-800"
                    >
                      <X className="w-2 h-2" />
                    </button>
                  </span>
                ))}
                {selectedValues.length > 3 && (
                  <span className="inline-flex items-center px-1.5 py-0.5 bg-gray-100 text-gray-600 text-xs font-medium rounded-full">
                    +{selectedValues.length - 3} more
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Algorithm Selection */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center mb-6">
          <Brain className="w-6 h-6 text-purple-600 mr-2" />
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Select Forecasting Algorithm</h2>
            <p className="text-sm text-gray-600 mt-1">Choose from 23 advanced algorithms or use Best Fit for automatic selection</p>
          </div>
        </div>

        {/* Best Fit (Featured) - Primary Recommendation */}
        <div className="mb-6">
          <div className="max-w-4xl mx-auto">
            {algorithms.filter(alg => alg.value === 'best_fit').map((algorithm) => (
              <div
                key={algorithm.value}
                className={`relative p-5 border-2 rounded-xl cursor-pointer transition-all duration-300 ${
                  config.algorithm === algorithm.value
                    ? 'border-purple-500 bg-gradient-to-br from-purple-50 to-indigo-50 shadow-lg'
                    : 'border-purple-300 bg-gradient-to-br from-purple-25 to-indigo-25 hover:border-purple-400 hover:shadow-md'
                }`}
                onClick={() => handleChange('algorithm', algorithm.value)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="text-2xl">
                      {algorithm.icon}
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-purple-900 flex items-center mb-1">
                        {algorithm.label}
                        <Zap className="w-5 h-5 ml-2 text-purple-600" />
                      </h3>
                      <p className="text-purple-700 text-sm leading-relaxed">
                        {algorithm.description}
                      </p>
                    </div>
                  </div>
                  {config.algorithm === algorithm.value && (
                    <div className="text-purple-500">
                      <div className="w-4 h-4 rounded-full bg-purple-500"></div>
                    </div>
                  )}
                </div>
                <div className="absolute -top-2 -right-2 bg-gradient-to-r from-purple-500 to-indigo-500 text-white text-xs px-3 py-1 rounded-full font-semibold shadow-lg">
                  ⭐ RECOMMENDED
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Category-Based Auto-Selection */}
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-3 text-center">
            Or Choose by Category
          </h3>
          <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-3">
            {algorithms.filter(alg => alg.category === 'Category-Based').map((algorithm) => (
              <div
                key={algorithm.value}
                className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-300 ${
                  config.algorithm === algorithm.value
                    ? 'border-indigo-500 bg-gradient-to-br from-indigo-50 to-blue-50 shadow-md'
                    : 'border-gray-200 bg-gradient-to-br from-gray-50 to-blue-25 hover:border-indigo-300 hover:shadow-sm'
                }`}
                onClick={() => handleChange('algorithm', algorithm.value)}
              >
                <div className="text-center">
                  <div className="text-xl mb-2">
                    {algorithm.icon}
                  </div>
                  <h4 className="font-semibold text-gray-900 mb-1 text-sm">
                    {algorithm.label}
                  </h4>
                  <p className="text-xs text-gray-600 leading-relaxed">
                    {algorithm.description}
                  </p>
                </div>
                {config.algorithm === algorithm.value && (
                  <div className="absolute top-2 right-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-500"></div>
                  </div>
                )}
                <div className="absolute -top-1 -right-1 bg-gradient-to-r from-indigo-500 to-blue-500 text-white text-xs px-2 py-0.5 rounded-full font-medium">
                  AUTO
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Individual Algorithm Categories */}
        <div className="space-y-5">
          {['Statistical', 'Machine Learning', 'Specialized'].map((category) => (
            <div key={category} className="max-w-6xl mx-auto">
              <h3 className="text-base font-semibold text-gray-900 mb-3 flex items-center justify-center">
                {category === 'Statistical' && <span className="mr-2 text-lg">📊</span>}
                {category === 'Machine Learning' && <span className="mr-2 text-lg">🤖</span>}
                {category === 'Specialized' && <span className="mr-2 text-lg">⚙️</span>}
                {category} Methods
                <span className="ml-2 text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
                  {algorithms.filter(alg => alg.category === category).length} algorithms
                </span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {algorithms.filter(alg => alg.category === category).map((algorithm) => (
                  <div
                    key={algorithm.value}
                    className={`relative p-3 border-2 rounded-lg cursor-pointer transition-all duration-300 group ${
                      config.algorithm === algorithm.value
                        ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-indigo-50 shadow-md'
                        : 'border-gray-200 hover:border-blue-300 hover:bg-gradient-to-br hover:from-gray-50 hover:to-blue-25 hover:shadow-sm'
                    }`}
                    onClick={() => handleChange('algorithm', algorithm.value)}
                  >
                    <div className="flex items-start space-x-2">
                      <div className="text-lg">
                        {algorithm.icon}
                      </div>
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900 text-xs mb-1">
                          {algorithm.label}
                        </h4>
                        <p className="text-xs text-gray-600 leading-tight">
                          {algorithm.description}
                        </p>
                      </div>
                    </div>
                    {config.algorithm === algorithm.value && (
                      <div className="absolute top-2 right-2">
                        <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {config.algorithm === 'best_fit' && (
          <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <Target className="w-5 h-5 text-purple-600" />
              <div>
                <h4 className="font-medium text-purple-900">Best Fit Mode</h4>
                <p className="text-sm text-purple-700 mt-1">
                  This will run all 23 algorithms and automatically select the one with the highest accuracy for your data.
                  You'll see results from all algorithms for comparison.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Data Selection */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center mb-6">
          <TrendingUp className="w-6 h-6 text-blue-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">Data Selection</h2>
        </div>

        {/* Advanced Multi-Dimension Selection Toggle */}
        <div className="mb-4">
          <div className="space-y-3">
            {/* Multi-Selection Mode Toggle */}
            <div className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
              <div className="flex items-center space-x-2">
                <Grid className="w-4 h-4 text-purple-600" />
                <div>
                  <h3 className="text-sm font-medium text-gray-900">Multi-Selection Forecasting</h3>
                  <p className="text-xs text-gray-600">Generate forecasts for multiple combinations</p>
                </div>
              </div>
              <button
                onClick={toggleMultiSelectMode}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  multiSelectMode 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-white text-purple-600 border border-purple-600 hover:bg-purple-50'
                }`}
              >
                {multiSelectMode ? 'Single Mode' : 'Multi Mode'}
              </button>
            </div>

            {/* Simple Multi-Select Toggle (only show if not in advanced multi-select mode) */}
            {!multiSelectMode && (
              <div className="flex items-center justify-between p-3 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
                <div className="flex items-center space-x-2">
                  <List className="w-4 h-4 text-green-600" />
                  <div>
                    <h3 className="text-sm font-medium text-gray-900">Simple Multi-Selection</h3>
                    <p className="text-xs text-gray-600">Select multiple items within the same dimension</p>
                  </div>
                </div>
                <button
                  onClick={toggleSimpleMultiSelect}
                  className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                    simpleMultiSelect 
                      ? 'bg-green-600 text-white' 
                      : 'bg-white text-green-600 border border-green-600 hover:bg-green-50'
                  }`}
                >
                  {simpleMultiSelect ? 'Single Select' : 'Multi Select'}
                </button>
              </div>
            )}
            {/* Advanced Mode Toggle (only show if not in multi-select mode) */}
            {!multiSelectMode && !simpleMultiSelect && (
              <div className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                <div className="flex items-center space-x-2">
                  <Target className="w-4 h-4 text-blue-600" />
                  <div>
                    <h3 className="text-sm font-medium text-gray-900">Precise Forecasting</h3>
                    <p className="text-xs text-gray-600">Select specific Product + Customer + Location combination</p>
                  </div>
                </div>
                <button
                  onClick={toggleAdvancedMode}
                  className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                    advancedMode 
                      ? 'bg-indigo-600 text-white' 
                      : 'bg-white text-indigo-600 border border-indigo-600 hover:bg-indigo-50'
                  }`}
                >
                  {advancedMode ? 'Flexible Mode' : 'Advanced Mode'}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* External Factors Section */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <ExternalFactorSelector
            selectedFactors={config.externalFactors || []}
            onFactorsChange={(factors) => onChange({ ...config, externalFactors: factors })}
          />
        </div>

        {(showAdvanced || multiSelectMode) ? (
          /* Advanced/Multi-Selection Mode */
          <div className="space-y-4">
            {multiSelectMode && (
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg border border-indigo-200">
                <div className="flex items-center space-x-3">
                  <Target className="w-5 h-5 text-indigo-600" />
                  <div>
                    <h3 className="font-medium text-gray-900">Advanced Mode (Precise Combinations)</h3>
                    <p className="text-sm text-gray-600">Generate forecasts for exact Product × Customer × Location combinations</p>
                  </div>
                </div>
                <button
                  onClick={toggleAdvancedMode}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    advancedMode 
                      ? 'bg-indigo-600 text-white' 
                      : 'bg-white text-indigo-600 border border-indigo-600 hover:bg-indigo-50'
                  }`}
                >
                  {advancedMode ? 'Flexible Mode' : 'Advanced Mode'}
                </button>
              </div>
            )}

            <div className="flex items-center space-x-2 mb-4">
              {advancedMode ? <Target className="w-5 h-5 text-indigo-600" /> : <Grid className="w-5 h-5 text-purple-600" />}
              <h3 className="text-lg font-semibold text-gray-900">
                {advancedMode ? 'Select Items for Precise Combinations' : 'Select Multiple Items (Flexible Combinations)'}
              </h3>
              {multiSelectMode && !advancedMode && (
                <p className="text-sm text-gray-600 ml-2">
                  Select at least 2 dimensions for combination forecasting
                </p>
              )}
              {advancedMode && (
                <p className="text-sm text-gray-600 ml-2">
                  All three dimensions required for precise forecasting
                </p>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Product Selection */}
              <MultiSelectBox
                title="Products"
                icon={<Package className="w-4 h-4 inline mr-1" />}
                options={filteredOptions.products}
                selectedValues={config.selectedProducts || []}
                onToggle={(value) => handleMultiSelectChange('selectedProducts', value)}
                onSelectAll={() => handleSelectAll('selectedProducts', filteredOptions.products)}
                searchValue={productSearch}
                onSearchChange={setProductSearch}
                required={advancedMode}
              />

              {/* Customer Selection */}
              <MultiSelectBox
                title="Customers"
                icon={<Users className="w-4 h-4 inline mr-1" />}
                options={filteredOptions.customers}
                selectedValues={config.selectedCustomers || []}
                onToggle={(value) => handleMultiSelectChange('selectedCustomers', value)}
                onSelectAll={() => handleSelectAll('selectedCustomers', filteredOptions.customers)}
                searchValue={customerSearch}
                onSearchChange={setCustomerSearch}
                required={advancedMode}
              />

              {/* Location Selection */}
              <MultiSelectBox
                title="Locations"
                icon={<MapPin className="w-4 h-4 inline mr-1" />}
                options={filteredOptions.locations}
                selectedValues={config.selectedLocations || []}
                onToggle={(value) => handleMultiSelectChange('selectedLocations', value)}
                onSelectAll={() => handleSelectAll('selectedLocations', filteredOptions.locations)}
                searchValue={locationSearch}
                onSearchChange={setLocationSearch}
                required={advancedMode}
              />
            </div>

            {multiSelectMode && !advancedMode && (
              <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                <div className="flex items-start space-x-3">
                  <Grid className="w-5 h-5 text-purple-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-purple-900">Multi-Selection Summary</h4>
                    <div className="text-sm text-purple-700 mt-1 space-y-1">
                      <p><strong>Products:</strong> {(config.selectedProducts || []).length} selected</p>
                      <p><strong>Customers:</strong> {(config.selectedCustomers || []).length} selected</p>
                      <p><strong>Locations:</strong> {(config.selectedLocations || []).length} selected</p>
                      <p><strong>Selected Dimensions:</strong> {
                        [(config.selectedProducts || []).length > 0 ? 'Products' : null,
                         (config.selectedCustomers || []).length > 0 ? 'Customers' : null,
                         (config.selectedLocations || []).length > 0 ? 'Locations' : null]
                        .filter(Boolean).join(', ') || 'None'
                      }</p>
                      <p><strong>Total Combinations:</strong> {
                        (() => {
                          const productCount = (config.selectedProducts || []).length || 1;
                          const customerCount = (config.selectedCustomers || []).length || 1;
                          const locationCount = (config.selectedLocations || []).length || 1;

                          let total = 0;
                          if ((config.selectedProducts || []).length > 0 && (config.selectedCustomers || []).length > 0 && (config.selectedLocations || []).length > 0) {
                            total = productCount * customerCount * locationCount;
                          } else if ((config.selectedProducts || []).length > 0 && (config.selectedCustomers || []).length > 0) {
                            total = productCount * customerCount;
                          } else if ((config.selectedProducts || []).length > 0 && (config.selectedLocations || []).length > 0) {
                            total = productCount * locationCount;
                          } else if ((config.selectedCustomers || []).length > 0 && (config.selectedLocations || []).length > 0) {
                            total = customerCount * locationCount;
                          }
                          return total;
                        })()
                      }</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {advancedMode && (
              <div className="p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
                <div className="flex items-start space-x-3">
                  <Target className="w-5 h-5 text-indigo-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-indigo-900">Advanced Mode Summary</h4>
                    <div className="text-sm text-indigo-700 mt-1 space-y-1">
                      <p><strong>Products:</strong> {(config.selectedProducts || []).length} selected</p>
                      <p><strong>Customers:</strong> {(config.selectedCustomers || []).length} selected</p>
                      <p><strong>Locations:</strong> {(config.selectedLocations || []).length} selected</p>
                      <p><strong>Precise Combinations:</strong> {
                        (config.selectedProducts || []).length * 
                        (config.selectedCustomers || []).length * 
                        (config.selectedLocations || []).length
                      }</p>
                      <p className="text-xs text-indigo-600 mt-2">
                        Each Product-Customer-Location combination will be forecasted separately without aggregation.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {!multiSelectMode && config.selectedProduct && config.selectedCustomer && config.selectedLocation && (
              <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-sm text-green-800">
                  <strong>Selected Combination:</strong> {config.selectedProduct} → {config.selectedCustomer} → {config.selectedLocation}
                </p>
                <p className="text-xs text-green-600 mt-1">
                  No aggregation will be applied - using exact data points for this combination
                </p>
              </div>
            )}
          </div>
        ) : simpleMultiSelect ? (
          /* Simple Multi-Select Mode */
          <div className="space-y-6">
            {/* Forecast By */}
            <div className='mt-6'>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Choose Forecast Dimension
              </label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {/* Product Option */}
                <div 
                  className={`relative p-3 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'product' 
                      ? 'border-blue-500 bg-blue-50 shadow-sm' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'product')}
                >
                  <div className="flex items-center space-x-2">
                    <div className={`p-1.5 rounded ${
                      config.forecastBy === 'product' ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <Package className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-gray-900">Products</h3>
                      <p className="text-xs text-gray-500">{productOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'product' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    </div>
                  )}
                </div>

                {/* Customer Option */}
                <div 
                  className={`relative p-3 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'customer' 
                      ? 'border-green-500 bg-green-50 shadow-sm' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'customer')}
                >
                  <div className="flex items-center space-x-2">
                    <div className={`p-1.5 rounded ${
                      config.forecastBy === 'customer' ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <Users className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-gray-900">Customers</h3>
                      <p className="text-xs text-gray-500">{customerOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'customer' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    </div>
                  )}
                </div>

                {/* Location Option */}
                <div 
                  className={`relative p-3 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'location' 
                      ? 'border-purple-500 bg-purple-50 shadow-sm' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'location')}
                >
                  <div className="flex items-center space-x-2">
                    <div className={`p-1.5 rounded ${
                      config.forecastBy === 'location' ? 'bg-purple-100 text-purple-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <MapPin className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-gray-900">Locations</h3>
                      <p className="text-xs text-gray-500">{locationOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'location' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Multi-Select Items with Search */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <div className="flex items-center space-x-2">
                  {getForecastTypeIcon()}
                  <span>Select Multiple {config.forecastBy.charAt(0).toUpperCase() + config.forecastBy.slice(1)}s</span>
                </div>
              </label>

              <div className="border border-gray-300 rounded-lg">
                {/* Search and Select All Header */}
                <div className="p-3 border-b border-gray-200 bg-gray-50">
                  <div className="flex items-center justify-between mb-2">
                    <button
                      type="button"
                      onClick={handleSimpleSelectAll}
                      className="flex items-center space-x-2 text-sm font-medium text-green-600 hover:text-green-700 transition-colors"
                    >
                      {(() => {
                        const filteredOptions = getFilteredSimpleOptions(getOptionsForType());
                        const allSelected = filteredOptions.length > 0 && filteredOptions.every(option => (config.selectedItems || []).includes(option));
                        const someSelected = filteredOptions.some(option => (config.selectedItems || []).includes(option));

                        return allSelected ? (
                          <CheckSquare className="w-4 h-4" />
                        ) : someSelected ? (
                          <div className="w-4 h-4 border-2 border-green-600 rounded bg-green-100 flex items-center justify-center">
                            <div className="w-2 h-2 bg-green-600 rounded"></div>
                          </div>
                        ) : (
                          <Square className="w-4 h-4" />
                        );
                      })()}
                      <span>
                        {(() => {
                          const filteredOptions = getFilteredSimpleOptions(getOptionsForType());
                          const allSelected = filteredOptions.length > 0 && filteredOptions.every(option => (config.selectedItems || []).includes(option));
                          return allSelected ? 'Deselect All' : 'Select All';
                        })()}
                        {simpleSearch && ` (${getFilteredSimpleOptions(getOptionsForType()).length})`}
                      </span>
                    </button>

                    <span className="text-xs text-gray-500">
                      {(config.selectedItems || []).length} of {getOptionsForType().length} selected
                    </span>
                  </div>

                  {/* Search Input */}
                  <div className="relative">
                    <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder={`Search ${config.forecastBy}s...`}
                      value={simpleSearch}
                      onChange={(e) => setSimpleSearch(e.target.value)}
                      className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    />
                  </div>
                </div>

                {/* Options List */}
                <div className="p-3 max-h-60 overflow-y-auto">
                  {(() => {
                    const filteredOptions = getFilteredSimpleOptions(getOptionsForType());

                    return filteredOptions.length > 0 ? (
                      <div className="space-y-2">
                        {filteredOptions.map((option) => (
                          <label key={option} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 p-1 rounded">
                            <input
                              type="checkbox"
                              checked={(config.selectedItems || []).includes(option)}
                              onChange={() => handleSimpleMultiSelectChange(option)}
                              className="rounded border-gray-300 text-green-600 focus:ring-green-500"
                            />
                            <span className="text-sm text-gray-700 flex-1">{option}</span>
                          </label>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-4">
                        <Search className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                        <p className="text-sm text-gray-500">
                          {simpleSearch ? `No ${config.forecastBy}s found matching "${simpleSearch}"` : `No ${config.forecastBy}s available`}
                        </p>
                      </div>
                    );
                  })()}
                </div>

                {/* Selected Items Summary */}
                {(config.selectedItems || []).length > 0 && (
                  <div className="p-3 border-t border-gray-200 bg-gray-50">
                    <div className="flex flex-wrap gap-1">
                      {(config.selectedItems || []).slice(0, 5).map((item) => (
                        <span
                          key={item}
                          className="inline-flex items-center px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full"
                        >
                          {item}
                          <button
                            type="button"
                            onClick={() => handleSimpleMultiSelectChange(item)}
                            className="ml-1 text-green-600 hover:text-green-800"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </span>
                      ))}
                      {(config.selectedItems || []).length > 5 && (
                        <span className="inline-flex items-center px-2 py-1 bg-gray-100 text-gray-600 text-xs font-medium rounded-full">
                          +{(config.selectedItems || []).length - 5} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {getOptionsForType().length === 0 && (
                <p className="mt-2 text-sm text-gray-500">
                  No {config.forecastBy}s available in data
                </p>
              )}
            </div>
          </div>
        ) : (
          /* Simple Single Dimension Selection */
          <div className="space-y-6">
            {/* Forecast By */}
            <div className='mt-6'>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Choose Forecast Dimension
              </label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Product Option */}
                <div 
                  className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'product' 
                      ? 'border-blue-500 bg-blue-50 shadow-md' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'product')}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      config.forecastBy === 'product' ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <Package className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Product</h3>
                      <p className="text-sm text-gray-500">{productOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'product' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    </div>
                  )}
                </div>

                {/* Customer Option */}
                <div 
                  className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'customer' 
                      ? 'border-green-500 bg-green-50 shadow-md' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'customer')}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      config.forecastBy === 'customer' ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <Users className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Customer</h3>
                      <p className="text-sm text-gray-500">{customerOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'customer' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                  )}
                </div>

                {/* Location Option */}
                <div 
                  className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                    config.forecastBy === 'location' 
                      ? 'border-purple-500 bg-purple-50 shadow-md' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => handleChange('forecastBy', 'location')}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      config.forecastBy === 'location' ? 'bg-purple-100 text-purple-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                      <MapPin className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Location</h3>
                      <p className="text-sm text-gray-500">{locationOptions.length} available</p>
                    </div>
                  </div>
                  {config.forecastBy === 'location' && (
                    <div className="absolute top-2 right-2">
                      <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Single Item Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <div className="flex items-center space-x-2">
                  {getForecastTypeIcon()}
                  <span>Select {config.forecastBy.charAt(0).toUpperCase() + config.forecastBy.slice(1)}</span>
                </div>
              </label>

              <select
                value={config.selectedItem || ''}
                onChange={(e) => handleChange('selectedItem', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
              >
                <option value="" disabled>
                  Select a {config.forecastBy}
                </option>
                {getOptionsForType().map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>

              {getOptionsForType().length === 0 && (
                <p className="mt-2 text-sm text-gray-500">
                  No {config.forecastBy}s available in data
                </p>
              )}

              {config.selectedItem && (
                <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-sm text-blue-800">
                    <strong>Selected:</strong> {config.selectedItem}
                  </p>
                  <p className="text-xs text-blue-600 mt-1">
                    Data will be aggregated across all other dimensions for this {config.forecastBy}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Time Configuration */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center mb-6">
          <Clock className="w-6 h-6 text-green-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">Time Configuration</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Time Interval */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Time Interval
            </label>
            <select
              value={config.interval}
              onChange={(e) => handleChange('interval', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
            >
              <option value="week">Weekly</option>
              <option value="month">Monthly</option>
              <option value="year">Yearly</option>
            </select>
          </div>

          {/* Historic Period */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Historic Periods
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={config.historicPeriod}
              onChange={(e) => handleChange('historicPeriod', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
            />
            <p className="mt-1 text-xs text-gray-500">
              Number of past {config.interval}s to analyze
            </p>
          </div>

          {/* Forecast Period */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Forecast Periods
            </label>
            <input
              type="number"
              min="1"
              max="52"
              value={config.forecastPeriod}
              onChange={(e) => handleChange('forecastPeriod', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
            />
            <p className="mt-1 text-xs text-gray-500">
              Number of future {config.interval}s to forecast
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};