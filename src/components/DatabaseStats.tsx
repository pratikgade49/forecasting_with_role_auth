import React from 'react';
import { Database, Calendar, Package, Users, MapPin, BarChart3 } from 'lucide-react';
import { DatabaseStatsType } from '../services/api';

interface DatabaseStatsProps {
  stats: DatabaseStatsType;
}

export const DatabaseStats: React.FC<DatabaseStatsProps> = ({ stats }) => {
  return (
    <div className="mb-8 bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center mb-6">
        <Database className="w-6 h-6 text-green-600 mr-2" />
        <h2 className="text-xl font-semibold text-gray-900">Database Overview</h2>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        {/* Total Records */}
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-600">Total Records</p>
              <p className="text-2xl font-bold text-blue-900">{stats.totalRecords.toLocaleString()}</p>
            </div>
            <BarChart3 className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        {/* Date Range */}
        <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-600">Date Range</p>
              <p className="text-xs font-bold text-green-900">{stats.dateRange.start}</p>
              <p className="text-xs font-bold text-green-900">to {stats.dateRange.end}</p>
            </div>
            <Calendar className="w-8 h-8 text-green-500" />
          </div>
        </div>

        {/* Unique Products */}
        <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-600">Products</p>
              <p className="text-2xl font-bold text-purple-900">{stats.uniqueProducts}</p>
            </div>
            <Package className="w-8 h-8 text-purple-500" />
          </div>
        </div>

        {/* Unique Customers */}
        <div className="bg-gradient-to-r from-orange-50 to-orange-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-orange-600">Customers</p>
              <p className="text-2xl font-bold text-orange-900">{stats.uniqueCustomers}</p>
            </div>
            <Users className="w-8 h-8 text-orange-500" />
          </div>
        </div>

        {/* Unique Locations */}
        <div className="bg-gradient-to-r from-red-50 to-red-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-red-600">Locations</p>
              <p className="text-2xl font-bold text-red-900">{stats.uniqueLocations}</p>
            </div>
            <MapPin className="w-8 h-8 text-red-500" />
          </div>
        </div>
      </div>
      
      {stats.totalRecords > 0 && (
        <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-sm text-green-800">
            ✅ Database is ready for forecasting! You can now configure and generate forecasts using your uploaded data.
          </p>
        </div>
      )}
    </div>
  );
};