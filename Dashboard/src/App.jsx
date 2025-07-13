import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import styles from './App.module.css';
import Sidebar from './layout/Sidebar';
import Dashboard from './features/dashboard/Dashboard';
import ApplianceDetails from './features/appliance-details/ApplianceDetails';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className={styles.container}>
      {!sidebarOpen && (
        <button
          className={styles.sidebarToggle}
          onClick={() => setSidebarOpen(true)}
          aria-label="Open sidebar"
        >
          ☰
        </button>
      )}
      {sidebarOpen && <Sidebar onClose={() => setSidebarOpen(false)} />}
      <main className={`${styles.mainContent} ${!sidebarOpen ? styles.sidebarClosedPadding : ''}`}>
        <Routes>
          <Route path="/" element={<ApplianceDetails />} />
          <Route path="/comparison" element={<Dashboard />} />
          <Route path="/appliance-details" element={<ApplianceDetails />} />
        </Routes>
      </main>
    </div>
  );
}
