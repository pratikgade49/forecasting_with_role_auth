import React from 'react';

interface Props {
  // Add any props you need here
}

const PendingApprovalMessage: React.FC<Props> = () => {
  return (
    <div>
      <h2>Pending Approval</h2>
      <p>Your account is pending approval. Please wait for an admin to review your account.</p>
    </div>
  );
};

export default PendingApprovalMessage;