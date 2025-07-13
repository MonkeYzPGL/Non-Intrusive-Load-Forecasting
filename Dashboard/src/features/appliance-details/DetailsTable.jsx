import { useEffect, useState } from 'react';
import { fetchChannelDetails, fetchDailyConsumption } from '../../api/appliance-api';

export default function DetailsTable({ channelId, selectedDate }) {
  const [data, setData] = useState({});
  const [dailyConsumption, setDailyConsumption] = useState(null);

  useEffect(() => {
    if (!channelId) return;

    fetchChannelDetails(channelId)
      .then(rows => setData(rows[0] || {}))
      .catch(err => console.error('Error loading metrics:', err));
  }, [channelId]);

  useEffect(() => {
    if (!channelId || !selectedDate) return;

    fetchDailyConsumption(channelId, selectedDate)
      .then(res => setDailyConsumption(res.total_consumption || null))
      .catch(err => console.error('Error fetching daily consumption:', err));

  }, [channelId, selectedDate]);

  return (
    <div style={{
      background: '#232b3e',
      padding: '24px',
      marginTop: '10%',
      borderRadius: '16px',
      marginBottom: '10%',
      maxWidth: '600px',
      width: '100%',
      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
      flex: '1'
    }}>
      <table style={{
        width: '100%',
        borderCollapse: 'separate',
        borderSpacing: '0 12px',
        color: '#e0e6f3',
        fontSize: '1rem'
      }}>
        <tbody>
          {Object.entries(data).map(([key, value]) => (
            <tr key={key}>
              <td style={{
                fontWeight: 600,
                padding: '8px 16px',
                width: '40%',
                color: '#b0b8c9',
                verticalAlign: 'top'
              }}>
                {key}:
              </td>
              <td style={{ padding: '8px 16px', color: '#fff' }}>
                {isNaN(value) ? value : Number(value).toFixed(2)}
              </td>
            </tr>
          ))}

          {dailyConsumption !== null && (
            <tr>
              <td style={{
                fontWeight: 600,
                padding: '8px 16px',
                color: '#b0b8c9',
              }}>
                selected_date_consum:
              </td>
              <td style={{ padding: '8px 16px', color: '#fff' }}>
                {Number(dailyConsumption).toFixed(2)} Wh
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
