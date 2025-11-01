// src/pages/AuthPage.jsx
import React, { useState } from 'react';
import AuthCard from '../components/AuthCard';
import ProcessModal from '../components/ProcessModal';

export default function AuthPage() {
  const [openProcess, setOpenProcess] = useState(false);

  return (
    <div style={{ padding: '20px 0', minHeight: '80vh', fontFamily: 'Inter, system-ui, Arial' }}>
      <AuthCard />
      <div style={{ textAlign: 'center', marginTop: 10 }}>
        <button onClick={() => setOpenProcess(true)} style={{ padding: '10px 14px', borderRadius:8 }}>Open Processing Modal</button>
      </div>

      <ProcessModal open={openProcess} onClose={() => setOpenProcess(false)} />
    </div>
  );
}
