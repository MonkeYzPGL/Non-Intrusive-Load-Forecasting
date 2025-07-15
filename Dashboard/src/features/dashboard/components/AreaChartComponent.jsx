import { useEffect, useState } from 'react';
import {AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend} from 'recharts';
import { fetchPredictionLSTMForDay, fetchBaselinePreview, fetchPredictionKANForDay } from '../../../api/forecast-api';

const renderLegend = ({ payload, showLSTM, setShowLSTM, showKAN, setShowKAN, showActual, setShowActual, showBaseline, setShowBaseline }) => {
  return (
    <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <li style={{ marginRight: '20px' }}>
        <label style={{ color: '#f43f5e', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showLSTM}
            onChange={(e) => setShowLSTM(e.target.checked)}
            style={{ marginRight: '5px', accentColor: '#f43f5e' }}
          />
          LSTM
        </label>
      </li>
      <li style={{ marginRight: '20px' }}>
        <label style={{ color: '#bb9922', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
          <input type="checkbox" checked={showKAN} onChange={(e) => setShowKAN(e.target.checked)} style={{ marginRight: '5px', accentColor: '#bb9922' }} />
          KAN
        </label>
      </li>
      <li style={{ marginRight: '20px' }}>
        <label style={{ color: '#3b82f6', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showActual}
            onChange={(e) => setShowActual(e.target.checked)}
            style={{ marginRight: '5px', accentColor: '#3b82f6' }}
          />
          Actual
        </label>
      </li>
      <li>
        <label style={{ color: '#10b981', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showBaseline}
            onChange={(e) => setShowBaseline(e.target.checked)}
            style={{ marginRight: '5px', accentColor: '#10b981' }}
          />
          Baseline
        </label>
      </li>
    </ul>
  );
};

function formatLabel(timestamp) {
  const date = new Date(timestamp);
  return `${date.getHours()}:00\n${date.getDate()}/${date.getMonth() + 1}`;
}

export default function AreaChartComponent({ showLSTM, showActual, setShowLSTM, setShowActual, selectedDate, channelId, onForecastReady, setLoadingForecast }) {

  const [data, setData] = useState([]);
  const [showBaseline, setShowBaseline] = useState(true);
  const [showKAN, setShowKAN] = useState(true);

  useEffect(() => {
    async function loadData() {
      if (!channelId) return;

      try {
        const baselineRaw = await fetchBaselinePreview(channelId);

        let LSTMMap = {};
        let KANMap = {};

        if (selectedDate && selectedDate !== 'All') {
          try {
            if (setLoadingForecast) setLoadingForecast(true);

            const [LSTMRaw, KANRaw] = await Promise.all([
              fetchPredictionLSTMForDay(channelId, selectedDate),
              fetchPredictionKANForDay(channelId, selectedDate)
            ]);

            LSTMMap = Object.fromEntries(
              LSTMRaw.map(row => {
                const iso = new Date(row.timestamp).toISOString();
                const value = row.predicted_power ?? row.total_predicted ?? null;
                return [iso, value];
              })
            );

            KANMap = Object.fromEntries(
              KANRaw.map(row => {
                const iso = new Date(row.timestamp).toISOString();
                const value = row.predicted_power ?? row.total_predicted ?? null;
                return [iso, value];
              })
            );

            if (Object.keys(LSTMMap).length > 0 && Object.keys(KANMap).length > 0 && onForecastReady) {
                onForecastReady();
              }
          } catch (err) {
            console.warn('Predictie indisponibila:', err.message);
          }
          if (setLoadingForecast) setLoadingForecast(false);
        }

        const parsedData = baselineRaw.map(row => {
          const iso = new Date(row.timestamp).toISOString();
          return {
            timestamp: row.timestamp,
            actual: row.actual ?? row.power ?? null,
            baseline: row.prediction ?? null,
            LSTM: LSTMMap[iso] ?? null,
            KAN: KANMap[iso] ?? null
          };
        });

        const filtered = selectedDate && selectedDate !== 'All'
          ? parsedData.filter(row =>
              new Date(row.timestamp).toISOString().split('T')[0] === selectedDate)
          : parsedData;

        setData(filtered);
      } catch (error) {
        console.error("Eroare la incarcare date:", error);
        setData([]);
      }
    }

    loadData();
  }, [selectedDate, channelId]);


  return (
    <div style={{ width: '100%', height: '100%', minHeight: 0 }}>
      {data.length === 0 ? (
        <div style={{
          color: '#e0e6f3',
          textAlign: 'center',
          marginTop: '40px',
          fontSize: '1.1rem'
        }}>
          No data available for the selected date.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 24, right: 32, left: 0, bottom: 0 }}
          >
          <defs>
            <linearGradient id="colorPredictor" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.7}/>
              <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.1}/>
            </linearGradient>
            <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.7}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
            </linearGradient>
            <linearGradient id="colorBaseline" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.7} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="colorKAN" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#bb9922" stopOpacity={0.7} />
              <stop offset="95%" stopColor="#bb9922" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#2d3650" vertical={false} />
          <XAxis
            dataKey="timestamp"
            tick={{ fill: '#b0b8c9', fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            interval={Math.ceil(data.length / 6)}
            tickFormatter={formatLabel}
          />
          <YAxis tick={{ fill: '#b0b8c9' }} tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{ background: '#232b3e', border: 'none', borderRadius: 8, color: '#fff' }}
            labelStyle={{ color: '#b0b8c9' }}
            formatter={(value, name) => {
              if (name === 'LSTM') return [value, 'LSTM'];
              if (name === 'KAN') return [value, 'KAN'];
              if (name === 'actual') return [value, 'Actual'];
              if (name === 'baseline') return [value, 'Baseline'];
              return [value, name];
            }}
            labelFormatter={label => `Time: ${label}`}
          />
          <Legend
            verticalAlign="top"
            height={36}
            content={({ payload }) => renderLegend({
              payload,
              showLSTM,
              setShowLSTM,
              showKAN,
              setShowKAN,
              showActual,
              setShowActual,
              showBaseline,
              setShowBaseline
            })}
          />
          {showActual && (
            <Area
              type="monotone"
              dataKey="actual"
              stroke="#3b82f6"
              fillOpacity={1}
              fill="url(#colorActual)"
              strokeWidth={3}
              dot={false}
              name="Actual"
            />
          )}
          {showBaseline && (
            <Area
              type="monotone"
              dataKey="baseline"
              stroke="#10b981"
              fillOpacity={1}
              fill="url(#colorBaseline)"
              strokeWidth={3}
              dot={false}
              name="Baseline"
            />
          )}
          {showLSTM && (
            <Area
              type="monotone"
              dataKey="LSTM"
              stroke="#f43f5e"
              fillOpacity={1}
              fill="url(#colorPredictor)"
              strokeWidth={3}
              dot={false}
              name="LSTM"
            />
          )}
          {showKAN && (
            <Area
              type="monotone"
              dataKey="KAN"
              stroke="#bb9922"
              fillOpacity={1}
              fill="url(#colorKAN)"
              strokeWidth={3}
              dot={false}
              name="KAN"
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
      )}
    </div>
  );
} 