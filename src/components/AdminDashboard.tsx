import React, { useState, useEffect } from 'react';
import { X, Users, UserCheck, UserX, Shield, Clock, Mail, Calendar, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react';
import { ApiService, UserResponse } from '../services/api';

interface AdminDashboardProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AdminDashboard: React.FC<AdminDashboardProps> = ({ isOpen, onClose }) => {
  const [users, setUsers] = useState<UserResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<{ [key: number]: string }>({});

  useEffect(() => {
    if (isOpen) {
      loadUsers();
    }
  }, [isOpen]);

  const loadUsers = async () => {
    setLoading(true);
    setError(null);
    try {
      const userList = await ApiService.listUsers();
      setUsers(userList);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  const handleApproveUser = async (userId: number, username: string) => {
    if (!confirm(`Are you sure you want to approve user "${username}"?`)) {
      return;
    }

    setActionLoading(prev => ({ ...prev, [userId]: 'approving' }));
    try {
      await ApiService.approveUser(userId);
      await loadUsers(); // Refresh the list
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve user');
    } finally {
      setActionLoading(prev => {
        const newState = { ...prev };
        delete newState[userId];
        return newState;
      });
    }
  };

  const handleToggleUserActive = async (userId: number, username: string, currentStatus: boolean) => {
    const action = currentStatus ? 'deactivate' : 'activate';
    if (!confirm(`Are you sure you want to ${action} user "${username}"?`)) {
      return;
    }

    setActionLoading(prev => ({ ...prev, [userId]: action }));
    try {
      await ApiService.setUserActive(userId, !currentStatus);
      await loadUsers(); // Refresh the list
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action} user`);
    } finally {
      setActionLoading(prev => {
        const newState = { ...prev };
        delete newState[userId];
        return newState;
      });
    }
  };

  const getUserStatusBadge = (user: UserResponse) => {
    if (!user.is_active) {
      return <span className="px-2 py-1 bg-red-100 text-red-800 text-xs font-medium rounded-full">Inactive</span>;
    }
    if (!user.is_approved && !user.is_admin) {
      return <span className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs font-medium rounded-full">Pending Approval</span>;
    }
    if (user.is_admin) {
      return <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs font-medium rounded-full">Admin</span>;
    }
    return <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">Active</span>;
  };

  const getUserStats = () => {
    const total = users.length;
    const active = users.filter(u => u.is_active).length;
    const pending = users.filter(u => !u.is_approved && !u.is_admin && u.is_active).length;
    const admins = users.filter(u => u.is_admin).length;
    const inactive = users.filter(u => !u.is_active).length;

    return { total, active, pending, admins, inactive };
  };

  if (!isOpen) return null;

  const stats = getUserStats();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-6xl mx-4 h-[85vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <Shield className="w-6 h-6 text-purple-600" />
            <h2 className="text-xl font-semibold text-gray-900">Admin Dashboard</h2>
            <span className="bg-purple-100 text-purple-800 text-sm px-3 py-1 rounded-full">
              {stats.total} users
            </span>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={loadUsers}
              disabled={loading}
              className="flex items-center space-x-2 px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              <span className="text-sm">Refresh</span>
            </button>
            
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="p-6 border-b border-gray-200">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-600">Total Users</p>
                  <p className="text-2xl font-bold text-blue-900">{stats.total}</p>
                </div>
                <Users className="w-6 h-6 text-blue-500" />
              </div>
            </div>

            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-600">Active</p>
                  <p className="text-2xl font-bold text-green-900">{stats.active}</p>
                </div>
                <UserCheck className="w-6 h-6 text-green-500" />
              </div>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-yellow-600">Pending</p>
                  <p className="text-2xl font-bold text-yellow-900">{stats.pending}</p>
                </div>
                <Clock className="w-6 h-6 text-yellow-500" />
              </div>
            </div>

            <div className="bg-purple-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-600">Admins</p>
                  <p className="text-2xl font-bold text-purple-900">{stats.admins}</p>
                </div>
                <Shield className="w-6 h-6 text-purple-500" />
              </div>
            </div>

            <div className="bg-red-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-red-600">Inactive</p>
                  <p className="text-2xl font-bold text-red-900">{stats.inactive}</p>
                </div>
                <UserX className="w-6 h-6 text-red-500" />
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <AlertCircle className="w-4 h-4 text-red-500 mr-2" />
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Users Table */}
        <div className="flex-1 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full"></div>
            </div>
          ) : users.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Users className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Users Found</h3>
                <p className="text-gray-600">No users are registered in the system.</p>
              </div>
            </div>
          ) : (
            <div className="overflow-y-auto h-full">
              <table className="w-full">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="text-left py-3 px-6 font-medium text-gray-700 border-b">User</th>
                    <th className="text-left py-3 px-6 font-medium text-gray-700 border-b">Email</th>
                    <th className="text-center py-3 px-6 font-medium text-gray-700 border-b">Status</th>
                    <th className="text-center py-3 px-6 font-medium text-gray-700 border-b">Created</th>
                    <th className="text-center py-3 px-6 font-medium text-gray-700 border-b">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user, index) => (
                    <tr key={user.id} className={`border-b border-gray-100 ${
                      index % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                    }`}>
                      <td className="py-4 px-6">
                        <div className="flex items-center space-x-3">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium ${
                            user.is_admin ? 'bg-purple-500' : 'bg-blue-500'
                          }`}>
                            {user.username.charAt(0).toUpperCase()}
                          </div>
                          <div>
                            <p className="font-medium text-gray-900">{user.username}</p>
                            {user.full_name && (
                              <p className="text-sm text-gray-500">{user.full_name}</p>
                            )}
                          </div>
                        </div>
                      </td>
                      
                      <td className="py-4 px-6">
                        <div className="flex items-center space-x-2">
                          <Mail className="w-4 h-4 text-gray-400" />
                          <span className="text-gray-900">{user.email}</span>
                        </div>
                      </td>
                      
                      <td className="py-4 px-6 text-center">
                        {getUserStatusBadge(user)}
                      </td>
                      
                      <td className="py-4 px-6 text-center">
                        <div className="flex items-center justify-center space-x-1 text-sm text-gray-500">
                          <Calendar className="w-4 h-4" />
                          <span>{new Date(user.created_at).toLocaleDateString()}</span>
                        </div>
                      </td>
                      
                      <td className="py-4 px-6">
                        <div className="flex items-center justify-center space-x-2">
                          {/* Approve Button */}
                          {!user.is_approved && !user.is_admin && user.is_active && (
                            <button
                              onClick={() => handleApproveUser(user.id, user.username)}
                              disabled={actionLoading[user.id] === 'approving'}
                              className="flex items-center space-x-1 px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50 transition-colors"
                            >
                              {actionLoading[user.id] === 'approving' ? (
                                <RefreshCw className="w-3 h-3 animate-spin" />
                              ) : (
                                <CheckCircle className="w-3 h-3" />
                              )}
                              <span>Approve</span>
                            </button>
                          )}
                          
                          {/* Activate/Deactivate Button */}
                          {!user.is_admin && (
                            <button
                              onClick={() => handleToggleUserActive(user.id, user.username, user.is_active)}
                              disabled={actionLoading[user.id] === 'activate' || actionLoading[user.id] === 'deactivate'}
                              className={`flex items-center space-x-1 px-3 py-1 rounded text-sm transition-colors disabled:opacity-50 ${
                                user.is_active
                                  ? 'bg-red-600 text-white hover:bg-red-700'
                                  : 'bg-blue-600 text-white hover:bg-blue-700'
                              }`}
                            >
                              {(actionLoading[user.id] === 'activate' || actionLoading[user.id] === 'deactivate') ? (
                                <RefreshCw className="w-3 h-3 animate-spin" />
                              ) : user.is_active ? (
                                <UserX className="w-3 h-3" />
                              ) : (
                                <UserCheck className="w-3 h-3" />
                              )}
                              <span>{user.is_active ? 'Deactivate' : 'Activate'}</span>
                            </button>
                          )}
                          
                          {/* Admin Badge */}
                          {user.is_admin && (
                            <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded text-sm font-medium">
                              Admin User
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div>
              Showing {users.length} users
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Active</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                <span>Pending</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                <span>Inactive</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span>Admin</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};