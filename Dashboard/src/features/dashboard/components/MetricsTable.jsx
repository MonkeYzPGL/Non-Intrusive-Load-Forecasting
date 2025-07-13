import { useEffect, useState } from 'react';
import { fetchForecastMetricsLSTM, fetchForecastMetricsKAN } from '../../../api/forecast-api';

export default function MetricsTable({ channelId, selectedDate, ready }) {
  const [metricsLSTM, setMetricsLSTM] = useState(null);
  const [metricsKAN, setMetricsKAN] = useState(null);

  useEffect(() => {
    if (!channelId || !selectedDate || selectedDate === 'All' || !ready) {
      setMetricsLSTM(null);
      setMetricsKAN(null);
      return;
    }

    Promise.all([
      fetchForecastMetricsLSTM(channelId, selectedDate),
      fetchForecastMetricsKAN(channelId, selectedDate)
    ])
      .then(([lstm, kan]) => {
        setMetricsLSTM(lstm);
        setMetricsKAN(kan);
      })
      .catch(err => {
        console.error('Eroare la fetch metrici:', err.message);
        setMetricsLSTM(null);
        setMetricsKAN(null);
      });
  }, [channelId, selectedDate, ready]);

  const commonKeys = metricsLSTM && metricsKAN
    ? Object.keys(metricsLSTM).filter(key => key in metricsKAN)
    : [];

  const isBetter = (metric, val1, val2) => {
    const lowerIsBetter = ['MAE', 'MAPE', 'MSE', 'RMSE', 'SMAPE'];
    if (lowerIsBetter.includes(metric)) {
      return val1 < val2 ? 'LSTM' : 'KAN';
    } else {
      return val1 > val2 ? 'LSTM' : 'KAN';
    }
  };

  return (
    <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '20px',
        justifyContent: 'flex-start',
        color: '#e0e6f3',
      }}>
      {!metricsLSTM || !metricsKAN ? (
        <p>No metrics available for the selected date.</p>
      ) : (
        commonKeys.map(metric => {
          const valLSTM = parseFloat(metricsLSTM[metric]);
          const valKAN = parseFloat(metricsKAN[metric]);
          const better = isBetter(metric, valLSTM, valKAN);

          return (
            <div key={metric} style={{
              background: '#2d3650',
              borderRadius: '12px',
              padding: '16px 24px',
              minWidth: '220px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              flex: '1 1 220px'
            }}>
              <div style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '12px' }}>{metric}</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.95rem' }}>
                <div style={{ color: better === 'LSTM' ? '#22c55e' : '#e0e6f3' }}>
                  <div style={{ fontSize: '0.8rem' }}>LSTM</div>
                  <div>{valLSTM.toFixed(4)}</div>
                </div>
                <div style={{ color: better === 'KAN' ? '#22c55e' : '#e0e6f3' }}>
                  <div style={{ fontSize: '0.8rem' }}>KAN</div>
                  <div>{valKAN.toFixed(4)}</div>
                </div>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
