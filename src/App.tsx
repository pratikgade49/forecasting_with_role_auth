import React, { useState, useEffect } from 'react';
import { BarChart3, Settings, Download, AlertCircle, CheckCircle, XCircle, Database, Upload, Eye, LogOut, User } from 'lucide-react';
import { TrendingUp } from 'lucide-react';
import { FileUpload } from './components/FileUpload';
import { ForecastConfiguration } from './components/ForecastConfiguration';
import { ForecastResults } from './components/ForecastResults';
import { MultiForecastResults } from './components/MultiForecastResults';
import { DatabaseStats } from './components/DatabaseStats';
import { DataViewer } from './components/DataViewer';
import { AuthModal } from './components/AuthModal';
import { ProcessLogViewer } from './components/ProcessLogViewer'; // Import ProcessLogViewer
import { SaveForecastModal } from './components/SaveForecastModal'; // Import SaveForecastModal
import { SavedForecastsManager } from './components/SavedForecastsManager'; // Import SavedForecastsManager
import { ApiService, ForecastConfig, ForecastResult, DatabaseStatsType, UserResponse } from './services/api';
''
import type { MultiForecastResult } from './services/api';
import { ConfigurationManager } from './components/ConfigurationManager'; // Import ConfigurationManager
import { ExternalFactorUpload } from './components/ExternalFactorUpload'; // Import ExternalFactorUpload
import { ModelCacheManager } from './components/ModelCacheManager';
import { LiveExternalFactorFetch } from './components/LiveExternalFactorFetch';
import { AdminDashboard } from './components/AdminDashboard';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState<UserResponse | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showAdminDashboard, setShowAdminDashboard] = useState(false);
  const [showDataViewer, setShowDataViewer] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [showUpload, setShowUpload] = useState(false);
  const [showExternalFactorUpload, setShowExternalFactorUpload] = useState(false); // New state for external factor upload
  const [showLiveFetch, setShowLiveFetch] = useState(false); // New state for live data fetch
  const [databaseStats, setDatabaseStats] = useState<DatabaseStatsType | null>(null);
  const [showModelCache, setShowModelCache] = useState(false);
  const [showProcessLogModal, setShowProcessLogModal] = useState(false);
  const [currentProcessLog, setCurrentProcessLog] = useState<string[]>([]);
  const [showSaveForecastModal, setShowSaveForecastModal] = useState(false);
  const [forecastConfigToSave, setForecastConfigToSave] = useState<ForecastConfig | null>(null);
  const [showSavedForecastsManager, setShowSavedForecastsManager] = useState(false);
  const [config, setConfig] = useState<ForecastConfig>({
    forecastBy: 'product',
    selectedItem: '',
    selectedProducts: [],
    selectedCustomers: [],
    selectedLocations: [],
    selectedItems: [],
    algorithm: 'best_fit',
    interval: 'month',
    historicPeriod: 12,
    forecastPeriod: 6,
    multiSelect: false
  });
  const [forecastResult, setForecastResult] = useState<ForecastResult | MultiForecastResult | null>(null);
  const [step, setStep] = useState<'configure' | 'results'>('configure');
  const [uniqueOptions, setUniqueOptions] = useState<{
    products: string[];
    customers: string[];
    locations: string[];
  }>({ products: [], customers: [], locations: [] });
  const [showConfigManager, setShowConfigManager] = useState(true); // <-- Added state for showConfigManager

  // Handle showing process log modal
  const handleShowProcessLog = (processLog: string[]) => {
    setCurrentProcessLog(processLog);
    setShowProcessLogModal(true);
  };

  const handleCloseProcessLog = () => {
    setShowProcessLogModal(false);
    setCurrentProcessLog([]);
  };

  // Handle saving forecast
  const handleSaveForecast = () => {
    setForecastConfigToSave(config);
    setShowSaveForecastModal(true);
  };

  // Handle viewing saved forecast
  const handleViewSavedForecast = (forecastData: ForecastResult | MultiForecastResult, forecastConfig: ForecastConfig) => {
    // Set the forecast result and config to display the saved forecast
    setForecastResult(forecastData);
    setConfig(forecastConfig);
    setStep('results');
    setShowSavedForecastsManager(false);
  };

  // Check backend status and load data on component mount
  useEffect(() => {
    checkBackendHealth();
    checkAuthStatus();
    loadDatabaseStats();
    loadUniqueOptions();
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      loadDatabaseStats();
      loadUniqueOptions();
    }
  }, [isAuthenticated]);
  const checkBackendHealth = async () => {
    try {
      await ApiService.checkHealth();
      setBackendStatus('online');
    } catch (error) {
      setBackendStatus('offline');
      setError('Backend server is not running. Please start the Python backend.');
    }
  };

  const checkAuthStatus = async () => {
    if (ApiService.isAuthenticated()) {
      try {
        const user = await ApiService.getCurrentUser();
        setCurrentUser(user);
        setIsAuthenticated(true);
      } catch (error) {
        ApiService.logout();
        setIsAuthenticated(false);
        setCurrentUser(null);
      }
    }
  };

  const handleAuthSuccess = async () => {
    try {
      const user = await ApiService.getCurrentUser();
      setCurrentUser(user);
      setIsAuthenticated(true);
      setShowAuthModal(false);
    } catch (error) {
      console.error('Failed to get user info:', error);
    }
  };

  const handleLogout = () => {
    ApiService.logout();
    setIsAuthenticated(false);
    setCurrentUser(null);
    setDatabaseStats(null);
    setUniqueOptions({ products: [], customers: [], locations: [] });
    setForecastResult(null);
    setStep('configure');
  };

  const loadDatabaseStats = async () => {
    try {
      const stats = await ApiService.getDatabaseStats();
      setDatabaseStats(stats);
    } catch (error) {
      console.error('Failed to load database stats:', error);
      if (error instanceof Error && error.message.includes('401')) {
        handleLogout();
      }
    }
  };

  const loadUniqueOptions = async () => {
    try {
      const options = await ApiService.getDatabaseOptions();
      setUniqueOptions(options);
    } catch (error) {
      console.error('Failed to load unique options:', error);
      if (error instanceof Error && error.message.includes('401')) {
        handleLogout();
      }
    }
  };

  const handleFileSelect = async (file: File) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await ApiService.uploadFile(file);
      setShowUpload(false);
      
      // Reload database stats and options
      await loadDatabaseStats();
      await loadUniqueOptions();
      
      // Show success message
      setError(null);
      alert(`File uploaded successfully!\nInserted: ${response.inserted} records\nDuplicates skipped: ${response.duplicates} records`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred processing the file');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateForecast = async () => {
    // Validate configuration based on mode
    if (config.multiSelect) {
      if (config.advancedMode) {
        // Advanced mode validation
        if (!config.selectedProducts || config.selectedProducts.length === 0 ||
            !config.selectedCustomers || config.selectedCustomers.length === 0 ||
            !config.selectedLocations || config.selectedLocations.length === 0) {
          setError('Advanced mode requires selection of Products, Customers, and Locations');
          return;
        }
      } else {
        // Multi-selection mode validation
        const selectedDimensions = [
          config.selectedProducts && config.selectedProducts.length > 0,
          config.selectedCustomers && config.selectedCustomers.length > 0,
          config.selectedLocations && config.selectedLocations.length > 0
        ].filter(Boolean).length;
        
        if (selectedDimensions < 2) {
          setError('Please select at least 2 dimensions (Product, Customer, or Location) for multi-selection forecasting');
          return;
        }
      }
    } else if (config.selectedItems && config.selectedItems.length > 1) {
      // Simple multi-select mode validation
      // No additional validation needed - selectedItems already validated
    } else {
      // Single selection mode validation
      const isAdvancedMode = config.selectedProduct || config.selectedCustomer || config.selectedLocation;
      
      if (isAdvancedMode) {
        if (!config.selectedProduct || !config.selectedCustomer || !config.selectedLocation) {
          setError('Please select Product, Customer, and Location for precise forecasting');
          return;
        }
      } else {
        if (!config.selectedItem) {
          setError('Please select an item to forecast');
          return;
        }
      }
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.generateForecast(config);
      setForecastResult(result);
      setStep('results');
      
      // Show notification that forecast was automatically saved
      alert('Forecast generated and automatically saved to your saved forecasts!');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred generating the forecast');
    } finally {
      setLoading(false);
      window.scrollTo(0, 0); // Scroll to top on forecast generation
    }
  };

  const isConfigurationValid = () => {
    if (config.multiSelect) {
      // Multi-selection mode validation
      const selectedDimensions = [
        config.selectedProducts && config.selectedProducts.length > 0,
        config.selectedCustomers && config.selectedCustomers.length > 0,
        config.selectedLocations && config.selectedLocations.length > 0
      ].filter(Boolean).length;
      return selectedDimensions >= 2;
    } else if (config.selectedItems && config.selectedItems.length > 1) {
      // Simple multi-select mode validation
      return true;
    } else {
      // Single selection mode validation
      const isAdvancedMode = config.selectedProduct || config.selectedCustomer || config.selectedLocation;
      
      if (isAdvancedMode) {
        return config.selectedProduct && config.selectedCustomer && config.selectedLocation;
      } else {
        return config.selectedItem;
      }
    }
  };

  const getSelectedItemDisplay = () => {
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
      
      // Calculate total combinations
      let totalCombinations = 0;
      const productCount = config.selectedProducts?.length || 0;
      const customerCount = config.selectedCustomers?.length || 0;
      const locationCount = config.selectedLocations?.length || 0;
      
      // Calculate combinations based on selected dimensions
      const selectedDimensions = [
        productCount > 0,
        customerCount > 0,
        locationCount > 0
      ].filter(Boolean).length;
      
      if (selectedDimensions >= 2) {
        if (productCount > 0 && customerCount > 0 && locationCount > 0) {
          totalCombinations = productCount * customerCount * locationCount;
        } else if (productCount > 0 && customerCount > 0) {
          totalCombinations = productCount * customerCount;
        } else if (productCount > 0 && locationCount > 0) {
          totalCombinations = productCount * locationCount;
        } else if (customerCount > 0 && locationCount > 0) {
          totalCombinations = customerCount * locationCount;
        }
      }
      
      return dimensions.length > 0 
        ? `${dimensions.join(' × ')} (${totalCombinations} combinations)`
        : 'No selections made';
    } else if (config.selectedItems && config.selectedItems.length > 1) {
      return `${config.selectedItems.length} ${config.forecastBy}s selected`;
    } else {
      if (config.selectedProduct && config.selectedCustomer && config.selectedLocation) {
        return `${config.selectedProduct} → ${config.selectedCustomer} → ${config.selectedLocation}`;
      } else if (config.selectedItem) {
        return config.selectedItem;
      }
    }
    return 'No selection';
  };

  const handleReset = () => {
    setConfig({
      forecastBy: 'product',
      selectedItem: '',
      selectedProducts: [],
      selectedCustomers: [],
      selectedLocations: [],
      selectedItems: [],
      algorithm: 'best_fit',
      interval: 'month',
      historicPeriod: 12,
      forecastPeriod: 6,
      multiSelect: false
    });
    setForecastResult(null);
    setStep('configure');
    setError(null);
  };

  const handleLoadConfiguration = (config: ForecastConfig) => {
    setConfig(config);
    setShowConfigManager(false);
  };

  // Always show the View Configurations button in the header
  // Handler to apply a configuration and generate forecast
  const handleApplyConfiguration = async (config: ForecastConfig) => {
    setConfig(config);
    setShowConfigManager(false);
    setLoading(true);
    setError(null);
    try {
      const result = await ApiService.generateForecast(config);
      setForecastResult(result);
      setStep('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred generating the forecast');
    } finally {
      setLoading(false);
    }
  };

  // Handler to update a configuration (assumes ConfigurationManager calls this with updated config)
  const handleUpdateConfiguration = async (updatedConfigObj: ForecastConfig) => {
    try {
      // Configuration update is handled in ConfigurationManager
      await loadUniqueOptions();
      await loadDatabaseStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred updating the configuration');
    }
  };

  // Handler to delete a configuration
  const handleDeleteConfiguration = async (configId: number) => {
    try {
      // Configuration deletion is handled in ConfigurationManager
      await loadUniqueOptions();
      await loadDatabaseStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred deleting the configuration');
    }
  };

  const hasData = databaseStats && databaseStats.totalRecords > 0;

  // Check if user needs approval
  const needsApproval = currentUser && !currentUser.is_approved && !currentUser.is_admin;
  const isAdmin = currentUser?.is_admin || false;

  // Show auth modal if not authenticated and backend is online
  if (backendStatus === 'online' && !isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 flex items-center justify-center">
        <div className="text-center">
          <BarChart3 className="w-16 h-16 text-blue-600 mx-auto mb-6" />
          <h1 className="text-3xl font-bold text-gray-900 mb-4">Multi-variant Forecasting Tool</h1>
          <p className="text-gray-600 mb-8">Please sign in to access the forecasting application</p>
          <button
            onClick={() => setShowAuthModal(true)}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Sign In
          </button>
        </div>
        
        <AuthModal
          isOpen={showAuthModal}
          onClose={() => setShowAuthModal(false)}
          onSuccess={handleAuthSuccess}
        />
      </div>
    );
  }

  // Show pending approval message if user needs approval
  if (backendStatus === 'online' && isAuthenticated && needsApproval) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-yellow-50 via-white to-orange-50 flex items-center justify-center">
        <div className="max-w-md mx-auto text-center bg-white rounded-xl shadow-lg p-8">
          <div className="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <User className="w-8 h-8 text-yellow-600" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Account Pending Approval</h1>
          <p className="text-gray-600 mb-6">
            Your account has been created successfully, but it requires approval from an administrator before you can access the forecasting application.
          </p>
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <p className="text-sm text-yellow-800">
              <strong>Account Details:</strong><br />
              Username: {currentUser.username}<br />
              Email: {currentUser.email}<br />
              Status: Pending Approval
            </p>
          </div>
          <p className="text-sm text-gray-500 mb-6">
            Please contact your system administrator or wait for approval. You will be able to access the application once your account is approved.
          </p>
          <button
            onClick={handleLogout}
            className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            Sign Out
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <BarChart3 className="w-8 h-8 text-blue-600" />
              <h1 className="text-xl font-bold text-gray-900">Multi-variant Forecasting Tool</h1>
              
              {/* Backend Status Indicator */}
              <div className="flex items-center space-x-2 ml-4">
                {backendStatus === 'checking' && (
                  <div className="flex items-center space-x-1 text-yellow-600">
                    <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
                    <span className="text-xs">Checking backend...</span>
                  </div>
                )}
                {backendStatus === 'online' && (
                  <div className="flex items-center space-x-1 text-green-600">
                    <CheckCircle className="w-4 h-4" />
                    <span className="text-xs">Backend Online</span>
                  </div>
                )}
                {backendStatus === 'offline' && (
                  <div className="flex items-center space-x-1 text-red-600">
                    <XCircle className="w-4 h-4" />
                    <span className="text-xs">Backend Offline</span>
                  </div>
                )}
              </div>
              </div>
            <div className='flex space-x-3'>
              {/* Admin Dashboard Button */}
              {isAdmin && (
                <button
                  onClick={() => setShowAdminDashboard(true)}
                  className="inline-flex items-center px-4 py-2 border border-purple-600 rounded-lg text-sm font-medium text-purple-600 bg-white hover:bg-purple-50 transition-colors"
                >
                  <Settings className="w-4 h-4 mr-2" />
                  Admin Dashboard
                </button>
              )}
              
              {/* Logout Button */}
              <button
                onClick={handleLogout}
                className="inline-flex items-center px-4 py-2 border border-red-600 rounded-lg text-sm font-medium text-red-600 bg-white hover:bg-red-50 transition-colors"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </button>
              {/* User Info */}
              {currentUser && (
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <User className="w-4 h-4" />
                  <span>
                    {currentUser.full_name || currentUser.username}
                    {isAdmin && <span className="ml-1 text-purple-600 font-medium">(Admin)</span>}
                  </span>
                </div>
              )}
              </div>
            </div>
          {/* </div> */}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Backend Offline Warning */}
            <div className="flex items-center space-x-4 mb-6">
              {/* {currentUser && (
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <User className="w-4 h-4" />
                  <span>{currentUser.full_name || currentUser.username}</span>
                </div>
              )} */}
              
              {/* View Data Button */}
              {hasData && (
                <button
                  onClick={() => setShowDataViewer(true)}
                  disabled={backendStatus === 'offline'}
                  className="inline-flex items-center px-4 py-2 border border-green-600 rounded-lg text-sm font-medium text-green-600 bg-white hover:bg-green-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Eye className="w-4 h-4 mr-2" />
                  View Data
                </button>
              )}
              
              {/* View Configurations Button - always visible */}
              {hasData && (
                <button
                  onClick={() => setShowConfigManager(true)}
                  disabled={backendStatus === 'offline'}
                  className="inline-flex items-center px-4 py-2 border border-purple-600 rounded-lg text-sm font-medium text-purple-600 bg-white hover:bg-purple-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Settings className="w-4 h-4 mr-2" />
                  Configurations
                </button>
              )}
              
              {/* Upload Data Button */}
              <button
                onClick={() => setShowUpload(!showUpload)}
                disabled={backendStatus === 'offline'}
                className="inline-flex items-center px-4 py-2 border border-blue-600 rounded-lg text-sm font-medium text-blue-600 bg-white hover:bg-blue-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload Data
              </button>

              {/* Upload External Factors Button */}
              <button
                onClick={() => setShowExternalFactorUpload(!showExternalFactorUpload)}
                disabled={backendStatus === 'offline'}
                className="inline-flex items-center px-4 py-2 border border-purple-600 rounded-lg text-sm font-medium text-purple-600 bg-white hover:bg-purple-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ml-2"
              >
                <TrendingUp className="w-4 h-4 mr-2" />
                Upload External Factors
              </button>

               {/* Fetch Live External Factors Button */}
              <button
                onClick={() => {
                  console.log('DEBUG: "Fetch Live External Factors" button clicked!');
                  console.log('Current showLiveFetch state:', showLiveFetch);
                  setShowLiveFetch(!showLiveFetch);
                  console.log('New showLiveFetch state should be:', !showLiveFetch);
                }}
                disabled={backendStatus === 'offline'}
                className="inline-flex items-center px-4 py-2 border border-blue-600 rounded-lg text-sm font-medium text-blue-600 bg-white hover:bg-blue-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ml-2"
              >
                <Download className="w-4 h-4 mr-2" />
                Fetch Live External Factors
              </button>
              
              {/* Model Cache Manager Button */}
              <button
                onClick={() => setShowModelCache(!showModelCache)}
                disabled={backendStatus === 'offline'}
                className="inline-flex items-center px-4 py-2 border border-gray-600 rounded-lg text-sm font-medium text-gray-600 bg-white hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Database className="w-4 h-4 mr-2" />
                Model Cache
              </button>
              
              {/* View Saved Forecasts Button */}
              <button
                onClick={() => setShowSavedForecastsManager(true)}
                className="inline-flex items-center px-4 py-2 border border-indigo-600 rounded-lg text-sm font-medium text-indigo-600 bg-white hover:bg-indigo-50 transition-colors"
              >
                <Database className="w-4 h-4 mr-2" />
                View Saved Forecasts
              </button>
              
              {step !== 'configure' && (
                <button
                  onClick={handleReset}
                  className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors"
                >
                  <Settings className="w-4 h-4 mr-2" />
                  New Analysis
                </button>
              )}
            </div>
        {backendStatus === 'offline' && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <XCircle className="w-5 h-5 text-red-500 mr-2" />
              <div>
                <p className="text-red-700 font-medium">Backend Server Required</p>
                <p className="text-red-600 text-sm mt-1">
                  Please start the Python backend server by running: <code className="bg-red-100 px-1 rounded">python backend/main.py</code>
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}

        {/* Upload Section */}
        {showUpload && (
          <div className="mb-8 bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-6">
              <Database className="w-6 h-6 text-blue-600 mr-2" />
              <h2 className="text-xl font-semibold text-gray-900">Upload Data to Database</h2>
            </div>
            
            <FileUpload 
              onFileSelect={handleFileSelect} 
              loading={loading} 
              error={error}
              disabled={backendStatus === 'offline'}
            />
            
            <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-blue-900 mb-4">Required Columns</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Date</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Quantity</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                  <span>Product</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                  <span>Customer</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                  <span>Location</span>
                </div>
              </div>
              <p className="text-xs text-blue-700 mt-4">
                <strong>Required:</strong> Date, Quantity | <strong>Optional:</strong> Product, Customer, Location (at least one required)
                <br />
                <strong>Note:</strong> Duplicate records will be automatically skipped based on Product + Customer + Location + Date combination.
              </p>
            </div>
          </div>
        )}

        {/* External Factor Upload Section */}
        {showExternalFactorUpload && (
          <div className="mb-8 bg-white rounded-xl shadow-lg p-6">
            <ExternalFactorUpload
              onUploadSuccess={() => {
                alert('External factors uploaded successfully!');
                setShowExternalFactorUpload(false);
              }}
            />
          </div>
        )}


        {/* Live External Factor Fetch Section - ADD THIS BLOCK */}
        {showLiveFetch && (
          <div className="mb-8">
            <LiveExternalFactorFetch
              onFetchSuccess={() => {
                alert('FRED data fetched and saved successfully!');
                setShowLiveFetch(false); // Optionally close after successful fetch
              }}
            />
          </div>
        )}


        {/* Model Cache Manager Section */}
        {showModelCache && (
          <div className="mb-8">
            <ModelCacheManager />
          </div>
        )}

        {/* Database Statistics */}
        {databaseStats && (
          <DatabaseStats stats={databaseStats} />
        )}

        {/* No Data Warning */}
        {!hasData && backendStatus === 'online' && (
          <div className="mb-6 p-6 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center">
              <AlertCircle className="w-6 h-6 text-yellow-500 mr-3" />
              <div>
                <p className="text-yellow-800 font-medium text-lg">No Data Available</p>
                <p className="text-yellow-700 mt-1">
                  Please upload an Excel or CSV file to get started with forecasting. Click the "Upload Data" button above.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Configuration Manager */}
        {hasData && showConfigManager && (
          <ConfigurationManager
            config={config}
            onLoadConfiguration={handleLoadConfiguration}
            onApply={handleApplyConfiguration} // Apply and forecast
            onUpdate={handleUpdateConfiguration} // Update config
            onDelete={handleDeleteConfiguration} // Delete config
            onClose={() => setShowConfigManager(false)}
          />
        )}

        {/* Data Viewer */}
        <DataViewer
          isOpen={showDataViewer}
          onClose={() => setShowDataViewer(false)}
          productOptions={uniqueOptions.products}
          customerOptions={uniqueOptions.customers}
          locationOptions={uniqueOptions.locations}
        />
        {/* Content based on current step */}
        {hasData && step === 'configure' && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Configure Your Forecast</h2>
              <p className="text-gray-600 mb-8">
                Set up your forecasting parameters and select the item to analyze from your database.
              </p>
            </div>
            
            <ForecastConfiguration
              config={config}
              onChange={setConfig}
              productOptions={uniqueOptions.products}
              customerOptions={uniqueOptions.customers}
              locationOptions={uniqueOptions.locations}
            />
            
            <div className="text-center">
              <button
                onClick={handleGenerateForecast}
                disabled={!isConfigurationValid() || loading}
                className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? (
                  <div className="animate-spin w-5 h-5 mr-2 border-2 border-white border-t-transparent rounded-full"></div>
                ) : (
                  <Settings className="w-5 h-5 mr-2" />
                )}
                {loading ? 'Generating...' : 'Generate Forecast'}
              </button>
            </div>
          </div>
        )}

        {hasData && step === 'results' && forecastResult && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Forecast Results</h2>
              <p className="text-gray-600 mb-8">
                Analysis complete! Review your forecast data and insights below.
              </p>
            </div>
            
            {'results' in forecastResult ? (
              <MultiForecastResults 
                result={forecastResult as MultiForecastResult}
                onShowProcessLog={handleShowProcessLog}
                onSaveForecast={handleSaveForecast}
              />
            ) : (
              <ForecastResults
                result={forecastResult as ForecastResult}
                forecastBy={config.forecastBy}
                selectedItem={getSelectedItemDisplay()}
                onShowProcessLog={handleShowProcessLog}
              />
            )}
          </div>
        )}


        {/* Process Log Viewer */}
        <ProcessLogViewer
          isOpen={showProcessLogModal}
          onClose={handleCloseProcessLog}
          processLog={currentProcessLog}
          title="Forecast Generation Process Log"
        />

        {/* Save Forecast Modal */}
        {showSaveForecastModal && forecastResult && forecastConfigToSave && (
          <SaveForecastModal
            isOpen={showSaveForecastModal}
            onClose={() => setShowSaveForecastModal(false)}
            forecastResult={forecastResult}
            forecastConfig={forecastConfigToSave}
            onSaveSuccess={() => {
              setShowSaveForecastModal(false);
              alert('Forecast saved successfully!');
              loadDatabaseStats();
              loadUniqueOptions();
            }}
          />
        )}

        {/* Saved Forecasts Manager */}
        <SavedForecastsManager
          isOpen={showSavedForecastsManager}
          onClose={() => setShowSavedForecastsManager(false)}
          onViewForecast={handleViewSavedForecast}
        />

        {/* Admin Dashboard */}
        {isAdmin && (
          <AdminDashboard
            isOpen={showAdminDashboard}
            onClose={() => setShowAdminDashboard(false)}
          />
        )}
      </main>
    </div>
  );
}

export default App;