import { useState, useEffect } from 'react';
import styles from './Dashboard.module.css';
import AreaChartComponent from './components/AreaChartComponent';
import MetricsTable from './components/MetricsTable';
import ApplianceDropdown from '../../generic-components/ApplianceDropdown';
import { fetchRecentDownsampledData } from '../../api/appliance-api';

export default function Dashboard() {
  const [showLSTM, setShowLSTM] = useState(true);
  const [showActual, setShowActual] = useState(true);
  const [selectedDate, setSelectedDate] = useState('');
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedAppliance, setSelectedAppliance] = useState(null);
  const [metricsReady, setMetricsReady] = useState(false);
  const [loadingForecast, setLoadingForecast] = useState(false);

  useEffect(() => {
    setMetricsReady(false);
  }, [selectedAppliance, selectedDate]);

  useEffect(() => {
    if (!selectedAppliance) return;

    fetchRecentDownsampledData(selectedAppliance.value)
      .then(data => {
        const uniqueDates = [...new Set(
          data.map(row => new Date(row.timestamp).toISOString().split('T')[0])
        )];
        const allDates = ['All', ...uniqueDates];
        setAvailableDates(allDates);

        if (!selectedDate || !allDates.includes(selectedDate)) {
          setSelectedDate(allDates[0]);
        }
      })
      .catch(err => console.error('Eroare la fetch dates:', err));
  }, [selectedAppliance]);

  return (
    <div className={styles.dashboard}>
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.cardTitle}>Comparison between LSTM/KAN/Baseline/Actual</span>
          <div style={{ marginTop: '2%', marginRight: '5%', width: 'fit-content' }}>
            <label style={{ color: '#e0e6f3', marginRight: '8px' }}>Select date:</label>
            <select
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              style={{
                padding: '6px 12px',
                background: '#2d3650',
                color: '#e0e6f3',
                border: 'none',
                borderRadius: '8px',
                fontSize: '1rem'
              }}
            >
              {availableDates.map(date => (
                <option key={date} value={date}>{date}</option>
              ))}
            </select>
          </div>
        </div>
        {loadingForecast && (
          <div style={{ textAlign: 'center', marginBottom: '12px' }}>
            <div className={styles.spinner}></div>
            <div style={{ color: '#e0e6f3', marginTop: '8px', fontStyle: 'italic' }}>
              Loading forecast...
            </div>
          </div>
        )}
        <div className={styles.chartContainer}>
          <AreaChartComponent
            showLSTM={showLSTM}
            showActual={showActual}
            setShowLSTM={setShowLSTM}
            setShowActual={setShowActual}
            selectedDate={selectedDate}
            channelId={selectedAppliance?.value}
            onForecastReady={() => setMetricsReady(true)}
            setLoadingForecast={setLoadingForecast}
          />
        </div>
      </div>
      <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: '100px',
          marginTop: '32px',
          marginLeft: '2%',
          width: '96%'
        }}>
        <ApplianceDropdown
          selected={selectedAppliance}
          onChange={setSelectedAppliance}
        />
        <MetricsTable
          channelId={selectedAppliance?.value}
          selectedDate={selectedDate}
          ready={metricsReady}
        />
      </div>
    </div>
  );
}