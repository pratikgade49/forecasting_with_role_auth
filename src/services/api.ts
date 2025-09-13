// const API_BASE_URL = 'http://10.213.1.29:8000';
export const API_BASE_URL = 'http://localhost:8000';

export interface UploadResponse {
  message: string;
  inserted: number;
  duplicates: number;
  totalRecords: number;
  filename: string;
}

export interface DatabaseStatsType {
  totalRecords: number;
  dateRange: {
    start: string;
    end: string;
  };
  uniqueProducts: number;
  uniqueCustomers: number;
  uniqueLocations: number;
}

export interface ForecastConfig {
  forecastBy: string;
  selectionKeyId?: number;
  selectedProducts?: string[];  // New multi-select fields
  selectedCustomers?: string[];
  selectedLocations?: string[];
  selectedItems?: string[];  // For simple mode multi-select
  algorithm: string;
  interval: string;
  historicPeriod: number;
  forecastPeriod: number;
  multiSelect?: boolean;  // Flag for multi-selection mode
  advancedMode?: boolean;  // Flag for advanced mode (precise combinations)
  externalFactors?: string[];  // Add external factors support
}

export interface DataPoint {
  date: string;
  quantity: number;
  period: string;
}

export interface AlgorithmResult {
  algorithm: string;
  accuracy: number;
  mae: number;
  rmse: number;
  historicData: DataPoint[];
  forecastData: DataPoint[];
  trend: string;
}

export interface ForecastResult {
  combination?: { [key: string]: string };
  selectedAlgorithm: string;
  accuracy: number;
  mae: number;
  rmse: number;
  historicData: DataPoint[];
  forecastData: DataPoint[];
  trend: string;
  allAlgorithms?: AlgorithmResult[];
  configHash?: string;
  processLog?: string[];
}

export interface MultiForecastResult {
  results: ForecastResult[];
  totalCombinations: number;
  summary: {
    averageAccuracy: number;
    bestCombination: {
      combination: { [key: string]: string };
      accuracy: number;
    };
    worstCombination: {
      combination: { [key: string]: string };
      accuracy: number;
    };
    successfulCombinations: number;
    failedCombinations: number;
    failedDetails: Array<{
      combination: string;
      error: string;
    }>;
  };
  processLog?: string[];
}

export interface SaveConfigRequest {
  name: string;
  description?: string;
  config: ForecastConfig;
}

export interface SavedConfiguration {
  id: number;
  name: string;
  description?: string;
  config: ForecastConfig;
  createdAt: string;
  updatedAt: string;
}

export interface ConfigurationResponse {
  id: number;
  name: string;
  description?: string;
  config: ForecastConfig;
  createdAt: string;
  updatedAt: string;
}

export interface UserCreate {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}

export interface UserLogin {
  username: string;
  password: string;
}

export interface UserResponse {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_approved?: boolean;
  is_admin?: boolean;
  created_at: string;
}

export interface Token {
  access_token: string;
  token_type: string;
  user: UserResponse;
}

export interface DataViewRequest {
  product?: string;
  customer?: string;
  location?: string;
  start_date?: string;
  end_date?: string;
  page?: number;
  page_size?: number;
}

export interface DataViewResponse {
  data: Array<{
    id: number;
    product?: string;
    quantity: number;
    product_group?: string;
    product_hierarchy?: string;
    location?: string;
    location_region?: string;
    customer?: string;
    customer_group?: string;
    customer_region?: string;
    ship_to_party?: string;
    sold_to_party?: string;
    uom?: string;
    date: string;
    unit_price?: number;
    created_at: string;
    updated_at: string;
  }>;
  total_records: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ExternalFactorUploadResponse {
  message: string;
  inserted: number;
  duplicates: number;
  totalRecords: number;
  filename: string;
}

export interface ModelCacheInfo {
  model_hash: string;
  algorithm: string;
  accuracy: number;
  created_at: string;
  last_used: string;
  use_count: number;
}

export interface FredDataRequest {
  series_ids: string[];
  start_date?: string;
  end_date?: string;
}

export interface FredDataResponse {
  message: string;
  inserted: number;
  duplicates: number;
  series_processed: number;
  series_details: Array<{
    series_id: string;
    status: string;
    message: string;
    inserted: number;
    duplicates?: number;
  }>;
}

export interface FredSeriesInfo {
  message: string;
  series: Record<string, Record<string, string>>;
  note: string;
}

export interface SavedForecast {
  id: number;
  user_id: number;
  name: string;
  description?: string;
  forecast_config: ForecastConfig;
  forecast_data: ForecastResult | MultiForecastResult;
  created_at: string;
  updated_at: string;
}

export interface SaveForecastRequest {
  name: string;
  description?: string;
  forecast_config: ForecastConfig;
  forecast_data: ForecastResult | MultiForecastResult;
}

export interface SavedForecastResponse {
  id: number;
  user_id: number;
  name: string;
  description?: string;
  forecast_config: ForecastConfig;
  forecast_data: ForecastResult | MultiForecastResult;
  created_at: string;
  updated_at: string;
}

export interface ScheduledForecastCreate {
  name: string;
  description?: string;
  forecast_config: ForecastConfig;
  frequency: 'daily' | 'weekly' | 'monthly';
  start_date: string;
  end_date?: string;
}

export interface ScheduledForecastUpdate {
  name?: string;
  description?: string;
  frequency?: 'daily' | 'weekly' | 'monthly';
  start_date?: string;
  end_date?: string;
  status?: 'active' | 'paused' | 'completed' | 'failed';
}

export interface ScheduledForecast {
  id: number;
  user_id: number;
  name: string;
  description?: string;
  forecast_config: ForecastConfig;
  frequency: string;
  start_date: string;
  end_date?: string;
  next_run: string;
  last_run?: string;
  status: string;
  run_count: number;
  success_count: number;
  failure_count: number;
  last_error?: string;
  created_at: string;
  updated_at: string;
}

export interface ForecastExecution {
  id: number;
  scheduled_forecast_id: number;
  execution_time: string;
  status: string;
  duration_seconds?: number;
  result_summary?: any;
  error_message?: string;
  created_at: string;
}

export interface FactorCoverageValidation {
    [factorName: string]: {
        coverage: number;
        message: string;
    };
}

export interface SchedulerStatus {
  running: boolean;
  check_interval: number;
}

export interface ForecastSelectionKey {
  id: number;
  product?: string;
  customer?: string;
  location?: string;
  created_at: string;
  updated_at: string;
}

export interface ResolveSelectionKeyRequest {
  product?: string;
  customer?: string;
  location?: string;
}

export interface ResolveSelectionKeyResponse {
  selection_key_id: number;
  selection_key: ForecastSelectionKey;
  created: boolean;
}
export class ApiService {
  private static getAuthHeaders(): HeadersInit {
    const token = localStorage.getItem('access_token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
  }

  static async register(userData: UserCreate): Promise<UserResponse> {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Registration failed');
    }

    return response.json();
  }

  static async login(credentials: UserLogin): Promise<Token> {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(credentials),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const token = await response.json();
    localStorage.setItem('access_token', token.access_token);
    return token;
  }

  static async getCurrentUser(): Promise<UserResponse> {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to get user info');
    }

    return response.json();
  }

  // Admin: list users
  static async listUsers(): Promise<UserResponse[]> {
    const response = await fetch(`${API_BASE_URL}/admin/users`, {
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Failed to list users');
    }
    return response.json();
  }

  // Admin: approve a user
  static async approveUser(userId: number): Promise<UserResponse> {
    const response = await fetch(`${API_BASE_URL}/admin/users/${userId}/approve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Failed to approve user');
    }
    return response.json();
  }

  // Admin: deactivate/reactivate user
  static async setUserActive(userId: number, active: boolean): Promise<UserResponse> {
    const response = await fetch(`${API_BASE_URL}/admin/users/${userId}/active`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify({ is_active: active }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Failed to update user status');
    }
    return response.json();
  }

  static logout(): void {
    localStorage.removeItem('access_token');
  }

  static isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  }

  static async saveForecast(request: SaveForecastRequest): Promise<SavedForecast> {
    const response = await fetch(`${API_BASE_URL}/saved_forecasts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to save forecast');
    }

    return response.json();
  }

  static async getSavedForecasts(): Promise<SavedForecast[]> {
    const response = await fetch(`${API_BASE_URL}/saved_forecasts`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch saved forecasts');
    }

    return response.json();
  }

  static async deleteSavedForecast(id: number): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/saved_forecasts/${id}`, {
      method: 'DELETE',
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete saved forecast');
    }
  }

  // Scheduled Forecasts API methods
  static async createScheduledForecast(request: ScheduledForecastCreate): Promise<ScheduledForecast> {
    const response = await fetch(`${API_BASE_URL}/scheduled_forecasts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create scheduled forecast');
    }

    return response.json();
  }

  static async getScheduledForecasts(): Promise<ScheduledForecast[]> {
    const response = await fetch(`${API_BASE_URL}/scheduled_forecasts`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch scheduled forecasts');
    }

    return response.json();
  }

  static async getScheduledForecast(id: number): Promise<ScheduledForecast> {
    const response = await fetch(`${API_BASE_URL}/scheduled_forecasts/${id}`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch scheduled forecast');
    }

    return response.json();
  }

  static async updateScheduledForecast(id: number, request: ScheduledForecastUpdate): Promise<ScheduledForecast> {
    const response = await fetch(`${API_BASE_URL}/scheduled_forecasts/${id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update scheduled forecast');
    }

    return response.json();
  }

  static async deleteScheduledForecast(id: number): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/scheduled_forecasts/${id}`, {
      method: 'DELETE',
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete scheduled forecast');
    }
  }

  static async getForecastExecutions(scheduledForecastId: number): Promise<ForecastExecution[]> {
    const response = await fetch(`${API_BASE_URL}/scheduled_forecasts/${scheduledForecastId}/executions`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch forecast executions');
    }

    return response.json();
  }

  static async getSchedulerStatus(): Promise<SchedulerStatus> {
    const response = await fetch(`${API_BASE_URL}/scheduler/status`);
    if (!response.ok) {
      throw new Error('Failed to fetch scheduler status');
    }
    return response.json();
  }

  static async resolveSelectionKey(request: ResolveSelectionKeyRequest): Promise<ResolveSelectionKeyResponse> {
    const response = await fetch(`${API_BASE_URL}/forecast_selection_keys/resolve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to resolve selection key');
    }

    return response.json();
  }

  static async getSelectionKey(id: number): Promise<ForecastSelectionKey> {
    const response = await fetch(`${API_BASE_URL}/forecast_selection_keys/${id}`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get selection key');
    }

    return response.json();
  }
  static async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      headers: {
        ...this.getAuthHeaders(),
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  }

  static async getDatabaseStats(): Promise<DatabaseStatsType> {
    const response = await fetch(`${API_BASE_URL}/database/stats`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch database stats');
    }

    return response.json();
  }

  static async getCustomerByProduct(id: string[]): Promise<{
    customers: string[];
  }> {
    const response = await fetch(`${API_BASE_URL}/get_customer_by_product/${id}`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch customers by product');
    }

    return response.json();
  }

  static async getDatabaseOptions(): Promise<{
    products: string[];
    customers: string[];
    locations: string[];
  }> {
    const response = await fetch(`${API_BASE_URL}/database/options`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch database options');
    }

    return response.json();
  }

  static async getFilteredOptions(filters: {
    selectedProducts?: string[];
    selectedCustomers?: string[];
    selectedLocations?: string[];
  }): Promise<{
    products: string[];
    customers: string[];
    locations: string[];
  }> {
    const response = await fetch(`${API_BASE_URL}/database/filtered_options`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(filters),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch filtered options');
    }

    return response.json();
  }

  static async getExternalFactors(): Promise<{ external_factors: string[] }> {
    const response = await fetch(`${API_BASE_URL}/external_factors`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch external factors');
    }

    return response.json();
  }

  static async uploadExternalFactors(file: File): Promise<ExternalFactorUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload_external_factors`, {
      method: 'POST',
      headers: {
        ...this.getAuthHeaders(),
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  }

  static async fetchFredData(request: FredDataRequest): Promise<FredDataResponse> {
    const response = await fetch(`${API_BASE_URL}/fetch_fred_data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch FRED data');
    }

    return response.json();
  }

  static async getFredSeriesInfo(): Promise<FredSeriesInfo> {
    const response = await fetch(`${API_BASE_URL}/fred_series_info`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch FRED series info');
    }

    return response.json();
  }

  static async validateExternalFactorCoverage(factorNames: string[]): Promise<FactorCoverageValidation> {
    const response = await fetch(`${API_BASE_URL}/external_factors/validate_coverage`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(factorNames),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Validation failed');
    }

    return response.json();
  }

  static async viewDatabaseData(request: DataViewRequest): Promise<DataViewResponse> {
    const response = await fetch(`${API_BASE_URL}/database/view`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch data');
    }

    return response.json();
  }

  static async getConfigurations(): Promise<{ configurations: SavedConfiguration[] }> {
    const response = await fetch(`${API_BASE_URL}/configurations`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch configurations');
    }

    return response.json();
  }

  static async saveConfiguration(request: SaveConfigRequest): Promise<{ message: string; id: number; name: string }> {
    const response = await fetch(`${API_BASE_URL}/configurations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to save configuration');
    }

    return response.json();
  }

  static async getConfiguration(id: number): Promise<SavedConfiguration> {
    const response = await fetch(`${API_BASE_URL}/configurations/${id}`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch configuration');
    }

    return response.json();
  }

  static async updateConfiguration(id: number, request: SaveConfigRequest): Promise<{ message: string; id: number; name: string }> {
    const response = await fetch(`${API_BASE_URL}/configurations/${id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update configuration');
    }

    return response.json();
  }

  static async deleteConfiguration(id: number): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/configurations/${id}`, {
      method: 'DELETE',
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to delete configuration');
    }

    return response.json();
  }

  static async generateForecast(config: ForecastConfig): Promise<ForecastResult> {
    console.log('API: Sending forecast request with config:', config);

    const response = await fetch(`${API_BASE_URL}/forecast`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(config),
    });

    console.log('API: Response status:', response.status);

    if (!response.ok) {
      const error = await response.json();
      console.error('API: Error response:', error);
      throw new Error(error.detail || 'Forecast generation failed');
    }

    const result = await response.json();
    console.log('API: Received result:', result);
    return result;
  }

  static async checkHealth(): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/`);

    if (!response.ok) {
      throw new Error('Backend is not responding');
    }

    return response.json();
  }

  static async getAlgorithms(): Promise<{ algorithms: Record<string, string> }> {
    const response = await fetch(`${API_BASE_URL}/algorithms`);

    if (!response.ok) {
      throw new Error('Failed to fetch algorithms');
    }

    return response.json();
  }

  static async getModelCacheInfo(): Promise<ModelCacheInfo[]> {
    const response = await fetch(`${API_BASE_URL}/model_cache_info`, {
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch model cache info');
    }

    return response.json();
  }

  static async clearModelCache(): Promise<{ message: string; cleared_count: number }> {
    const response = await fetch(`${API_BASE_URL}/clear_model_cache`, {
      method: 'POST',
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to clear model cache');
    }

    return response.json();
  }

  static async clearAllModelCache(): Promise<{ message: string; cleared_count: number }> {
    const response = await fetch(`${API_BASE_URL}/clear_all_model_cache`, {
      method: 'POST',
      headers: {
        ...this.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to clear all model cache');
    }

    return response.json();
  }
}