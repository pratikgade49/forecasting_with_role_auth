import React, { useState, useEffect } from 'react';
import axios from 'axios';

const api = axios.create({
  baseURL: 'https://your-api-url.com/api',
  headers: {
    Authorization: `Bearer ${localStorage.getItem('token')}`,
  },
});

interface User {
  id: number;
  username: string;
  // Add any other user fields you need here
}

interface Props {
  // Add any props you need here
}

const AdminDashboard: React.FC<Props> = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('/api/users')
      .then(response => {
        setUsers(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error(error);
      });
  }, []);

  const handleApproveUser = (userId: number) => {
    axios.post(`/api/users/${userId}/approve`)
      .then(response => {
        console.log(response.data);
      })
      .catch(error => {
        console.error(error);
      });
  };

  return (
    <div>
      <h2>Admin Dashboard</h2>
      <ul>
        {users.map((user) => (
          <li key={user.id}>
            <span>{user.username}</span>
            <button onClick={() => handleApproveUser(user.id)}>Approve</button>
          </li>
        ))}
      </ul>
      {loading && <p>Loading...</p>}
    </div>
  );
};

export default AdminDashboard;