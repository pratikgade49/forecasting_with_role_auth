# Advanced Multi-variant Forecasting API with MySQL

## 🚀 Features

### 8 Powerful Forecasting Algorithms
1. **Linear Regression** - Simple trend-based forecasting
2. **Polynomial Regression** - Captures non-linear patterns
3. **Exponential Smoothing** - Weighted recent observations
4. **Holt-Winters** - Handles trend and seasonality
5. **ARIMA** - Autoregressive integrated model
6. **Random Forest** - Machine learning ensemble
7. **Seasonal Decomposition** - Separates trend and seasonality
8. **Moving Average** - Smoothed historical average

### 🎯 Best Fit Mode
- Automatically runs all 8 algorithms
- Selects the best performing algorithm based on accuracy
- Provides comparison of all algorithms
- Shows detailed performance metrics

### 🗄️ MySQL Database Integration
- Persistent data storage
- Automatic duplicate prevention
- Fast data retrieval and filtering
- Database statistics and monitoring

## 📊 Installation & Setup

### Prerequisites
- Python >= 3.8 , <= 3.12
- PostgreSQL Server 17.5
- pip package manager

### Quick Start

1. **Prerequisites**
   - Install PostgreSQL 17.5
   - Create database: `CREATE DATABASE forecasting_db;`
   - Grant permissions to your PostgreSQL user

2. **Install Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Setup Database Tables**
   ```bash
   # Create tables (database must already exist)
   python setup_database.py
   ```

4. **Configure Database (Optional)**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your PostgreSQL credentials
   nano .env
   ```

5. **Start the Server**
   ```bash
   python main.py
   ```

6. **Access the Application**
   - Backend API: http://localhost:8000
   - Frontend: http://localhost:5173

## 🗄️ Database Configuration

### Default Settings
- Host: localhost
- Port: 3306
- User: root
- Password: password
- Database: forecasting_db

### Environment Variables
Create a `.env` file in the backend directory:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=forecasting_db
```

### Database Schema

The application creates these tables if they don't exist:

#### forecast_data table:

```sql
CREATE TABLE forecast_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    product VARCHAR(255),
    quantity DECIMAL(15,2) NOT NULL,
    product_group VARCHAR(255),
    product_hierarchy VARCHAR(255),
    location VARCHAR(255),
    location_region VARCHAR(255),
    customer VARCHAR(255),
    customer_group VARCHAR(255),
    customer_region VARCHAR(255),
    ship_to_party VARCHAR(255),
    sold_to_party VARCHAR(255),
    uom VARCHAR(50),
    date DATE NOT NULL,
    unit_price DECIMAL(15,2),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_forecast_record (product, customer, location, date)
);
```

#### forecast_configurations table:

```sql
CREATE TABLE forecast_configurations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    forecast_by VARCHAR(50) NOT NULL,
    selected_item VARCHAR(255),
    selected_product VARCHAR(255),
    selected_customer VARCHAR(255),
    selected_location VARCHAR(255),
    algorithm VARCHAR(100) NOT NULL DEFAULT 'best_fit',
    interval VARCHAR(20) NOT NULL DEFAULT 'month',
    historic_period INT NOT NULL DEFAULT 12,
    forecast_period INT NOT NULL DEFAULT 6,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_config_name (name)
);
```

## 🔧 API Endpoints

### Health Check
```
GET /
```

### Database Statistics
```
GET /database/stats
```

### Database Options
```
GET /database/options
```

### Upload Data File
```
POST /upload
Content-Type: multipart/form-data
```

### Generate Forecast
```
POST /forecast
Content-Type: application/json
```

### Download Forecast Excel
```
POST /download_forecast_excel
Content-Type: application/json
```

## 📈 Data Upload Process

### Supported File Formats
- CSV files (.csv)
- Excel files (.xlsx, .xls)

### Required Columns
- **Date**: Date/time information
- **Quantity**: Numeric values to forecast

### Optional Columns
- **Product**: Product identifier
- **Customer**: Customer identifier  
- **Location**: Location identifier
- **Product Group**: Product grouping
- **Product Hierarchy**: Product hierarchy
- **Location Region**: Location region
- **Customer Group**: Customer grouping
- **Customer Region**: Customer region
- **Ship To Party**: Shipping party
- **Sold To Party**: Selling party
- **UoM**: Unit of measure
- **Unit Price**: Price per unit

### Duplicate Prevention
The system automatically prevents duplicate records based on the combination of:
- Product + Customer + Location + Date

Duplicate records are skipped during upload and reported in the response.

## 🎯 Forecasting Workflow

### 1. Upload Data
- Upload Excel/CSV files through the web interface
- Data is automatically stored in PostgreSQL database
- Duplicates are prevented and reported

### 2. Configure Forecast
- Select forecasting dimension (Product/Customer/Location)
- Choose specific items or combinations
- Set time intervals and periods
- Select algorithm or use Best Fit mode

### 3. Generate Results
- View forecast charts and metrics
- Compare algorithm performances (Best Fit mode)
- Download results as Excel files

## 📊 Algorithm Performance Metrics

### Accuracy
- Calculated as: `100 - MAPE`
- Range: 0-100%
- Higher is better

### MAE (Mean Absolute Error)
- Average absolute difference
- Lower is better
- Same units as data

### RMSE (Root Mean Square Error)
- Square root of mean squared errors
- Lower is better
- Penalizes large errors more

## 🔧 Configuration Options

### Data Selection Modes

#### Simple Mode
- Select single dimension (Product/Customer/Location)
- Aggregates all data for selected item

#### Advanced Mode
- Select exact combination of Product + Customer + Location
- Uses precise data points without aggregation

### Time Configuration
- **Intervals**: Weekly, Monthly, Yearly
- **Historic Period**: Number of past periods to analyze
- **Forecast Period**: Number of future periods to predict

### Algorithm Selection
- Choose specific algorithm
- Use "Best Fit" for automatic selection with ensemble

## 🚀 Usage Examples

### Basic Forecast
```python
config = {
    "forecastBy": "product",
    "selectedItem": "Product A",
    "algorithm": "linear_regression",
    "interval": "month",
    "historicPeriod": 12,
    "forecastPeriod": 6
}
```

### Best Fit Mode
```python
config = {
    "forecastBy": "product",
    "selectedItem": "Product A",
    "algorithm": "best_fit",  # Auto-select best algorithm
    "interval": "month",
    "historicPeriod": 12,
    "forecastPeriod": 6
}
```

### Advanced Mode
```python
config = {
    "selectedProduct": "Product A",
    "selectedCustomer": "Customer X",
    "selectedLocation": "Location Y",
    "algorithm": "holt_winters",
    "interval": "week",
    "historicPeriod": 24,
    "forecastPeriod": 12
}
```

## 🔍 Error Handling

The API provides detailed error messages for:
- Database connection issues
- Invalid file formats
- Missing required columns
- Insufficient data
- Algorithm failures
- Configuration errors

## 🔧 Troubleshooting

### Database Issues
1. **Connection Failed**: Check PostgreSQL server status, credentials, and ensure database exists
2. **Permission Denied**: Ensure user has CREATE TABLE/INSERT/SELECT privileges
3. **Tables Not Found**: Run `python setup_database.py`
4. **Database Not Found**: Create database manually: `CREATE DATABASE forecasting_db;`

### Data Upload Issues
1. **Invalid Format**: Ensure file is CSV or Excel format
2. **Missing Columns**: Verify Date and Quantity columns exist
3. **Date Parsing**: Check date format consistency

### Forecasting Issues
1. **No Data Found**: Verify filter criteria and data availability
2. **Insufficient Data**: Ensure enough historical data points
3. **Algorithm Failure**: Try Best Fit mode for automatic selection

## 📊 Performance Tips

- Use appropriate time intervals for your data
- Ensure sufficient historical data (12+ periods recommended)
- Consider data seasonality when selecting algorithms
- Use Best Fit mode when unsure about algorithm choice
- Validate results with domain knowledge

## 🔄 Data Management

### Viewing Database Stats
The application provides real-time database statistics including:
- Total number of records
- Date range of data
- Number of unique products, customers, and locations

### Data Persistence
- All uploaded data is permanently stored in PostgreSQL
- No need to re-upload files for subsequent analyses
- Data can be queried and filtered efficiently

### Backup and Recovery
- Regular PostgreSQL backups recommended
- Export data using standard PostgreSQL tools
- Database schema can be recreated using setup script