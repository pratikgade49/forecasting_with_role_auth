import React, { useState, useEffect } from 'react';
import { Eye, Filter, Download, ChevronLeft, ChevronRight, Calendar, Package, Users, MapPin, X } from 'lucide-react';
import { ApiService, DataViewRequest, DataViewResponse } from '../services/api';

interface DataViewerProps {
  isOpen: boolean;
  onClose: () => void;
  productOptions: string[];
  customerOptions: string[];
  locationOptions: string[];
}

export const DataViewer: React.FC<DataViewerProps> = ({
  isOpen,
  onClose,
  productOptions,
  customerOptions,
  locationOptions
}) => {
  const [data, setData] = useState<DataViewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [filters, setFilters] = useState<DataViewRequest>({
    product: '',
    customer: '',
    location: '',
    start_date: '',
    end_date: '',
    page: 1,
    page_size: 50
  });

  const [showFilters, setShowFilters] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen, filters.page]);

  const loadData = async () => {
    setLoading(true);
    setError(null);

    try {
      const request: DataViewRequest = {
        ...filters,
        product: filters.product || undefined,
        customer: filters.customer || undefined,
        location: filters.location || undefined,
        start_date: filters.start_date || undefined,
        end_date: filters.end_date || undefined
      };

      const response = await ApiService.viewDatabaseData(request);
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (field: keyof DataViewRequest, value: string | number) => {
    setFilters(prev => ({
      ...prev, // Add the spread operator here
      [field]: value,
      page: field !== 'page' ? 1 : value
    }));
  };
  const applyFilters = () => {
    setFilters(prev => ({ ...prev, page: 1 }));
    loadData();
    setShowFilters(false);
  };

  const clearFilters = () => {
    setFilters({
      product: '',
      customer: '',
      location: '',
      start_date: '',
      end_date: '',
      page: 1,
      page_size: 50
    });
  };

  const exportToCSV = () => {
    if (!data || data.data.length === 0) return;

    const headers = [
      'ID', 'Product', 'Customer', 'Location', 'Date', 'Quantity', 
      'UoM', 'Unit Price', 'Product Group', 'Product Hierarchy',
      'Location Region', 'Customer Group', 'Customer Region'
    ];

    const csvContent = [
      headers.join(','),
      ...data.data.map(row => [
        row.id,
        `"${row.product || ''}"`,
        `"${row.customer || ''}"`,
        `"${row.location || ''}"`,
        row.date,
        row.quantity,
        `"${row.uom || ''}"`,
        row.unit_price || '',
        `"${row.product_group || ''}"`,
        `"${row.product_hierarchy || ''}"`,
        `"${row.location_region || ''}"`,
        `"${row.customer_group || ''}"`,
        `"${row.customer_region || ''}"`
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forecast_data_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-7xl mx-4 h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <Eye className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">Database Data Viewer</h2>
            {data && (
              <span className="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full">
                {data.total_records.toLocaleString()} records
              </span>
            )}
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              <Filter className="w-4 h-4" />
              <span>Filters</span>
            </button>
            
            <button
              onClick={exportToCSV}
              disabled={!data || data.data.length === 0}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Download className="w-4 h-4" />
              <span>Export CSV</span>
            </button>
            
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <div className="p-6 bg-gray-50 border-b border-gray-200">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <Package className="w-4 h-4 inline mr-1" />
                  Product
                </label>
                <select
                  value={filters.product}
                  onChange={(e) => handleFilterChange('product', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">All Products</option>
                  {productOptions.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <Users className="w-4 h-4 inline mr-1" />
                  Customer
                </label>
                <select
                  value={filters.customer}
                  onChange={(e) => handleFilterChange('customer', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">All Customers</option>
                  {customerOptions.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <MapPin className="w-4 h-4 inline mr-1" />
                  Location
                </label>
                <select
                  value={filters.location}
                  onChange={(e) => handleFilterChange('location', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">All Locations</option>
                  {locationOptions.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <Calendar className="w-4 h-4 inline mr-1" />
                  Start Date
                </label>
                <input
                  type="date"
                  value={filters.start_date}
                  onChange={(e) => handleFilterChange('start_date', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  <Calendar className="w-4 h-4 inline mr-1" />
                  End Date
                </label>
                <input
                  type="date"
                  value={filters.end_date}
                  onChange={(e) => handleFilterChange('end_date', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
            
            <div className="flex items-center justify-end space-x-3 mt-4">
              <button
                onClick={clearFilters}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
              >
                Clear Filters
              </button>
              <button
                onClick={applyFilters}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Apply Filters
              </button>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {loading ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
            </div>
          ) : error ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <p className="text-red-600 mb-4">{error}</p>
                <button
                  onClick={loadData}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : data && data.data.length > 0 ? (
            <>
              {/* Table */}
              <div className="flex-1 overflow-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="text-left py-3 px-4 font-medium text-gray-700 border-b">ID</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-700 border-b">Product</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-700 border-b">Customer</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-700 border-b">Location</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-700 border-b">Date</th>
                      <th className="text-right py-3 px-4 font-medium text-gray-700 border-b">Quantity</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-700 border-b">UoM</th>
                      <th className="text-right py-3 px-4 font-medium text-gray-700 border-b">Unit Price</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.data.map((row, index) => (
                      <tr key={row.id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="py-3 px-4 border-b text-gray-600">{row.id}</td>
                        <td className="py-3 px-4 border-b text-gray-900">{row.product || '-'}</td>
                        <td className="py-3 px-4 border-b text-gray-900">{row.customer || '-'}</td>
                        <td className="py-3 px-4 border-b text-gray-900">{row.location || '-'}</td>
                        <td className="py-3 px-4 border-b text-gray-600">{row.date}</td>
                        <td className="py-3 px-4 border-b text-right font-medium text-gray-900">
                          {row.quantity.toLocaleString()}
                        </td>
                        <td className="py-3 px-4 border-b text-gray-600">{row.uom || '-'}</td>
                        <td className="py-3 px-4 border-b text-right text-gray-900">
                          {row.unit_price ? `$${row.unit_price.toFixed(2)}` : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="flex items-center justify-between p-4 border-t border-gray-200 bg-white">
                <div className="text-sm text-gray-600">
                  Showing {((data.page - 1) * data.page_size) + 1} to {Math.min(data.page * data.page_size, data.total_records)} of {data.total_records.toLocaleString()} results
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleFilterChange('page', Math.max(1, data.page - 1))}
                    disabled={data.page <= 1}
                    className="flex items-center px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <ChevronLeft className="w-4 h-4 mr-1" />
                    Previous
                  </button>
                  
                  <span className="text-sm text-gray-600">
                    Page {data.page} of {data.total_pages}
                  </span>
                  
                  <button
                    onClick={() => handleFilterChange('page', Math.min(data.total_pages, data.page + 1))}
                    disabled={data.page >= data.total_pages}
                    className="flex items-center px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                    <ChevronRight className="w-4 h-4 ml-1" />
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <Eye className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 mb-4">No data found matching your criteria</p>
                <button
                  onClick={clearFilters}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Clear Filters
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};