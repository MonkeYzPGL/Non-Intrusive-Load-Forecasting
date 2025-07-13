import { getHistogramUrl } from '../../api/appliance-api';

export default function HistogramImage({ channelId }) {
  return (
    <div style={{
        background: '#232b3e',
        borderRadius: '16px',
        padding: '24px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
        maxWidth: '700px',
        flex: '1'
      }}>
        <img
          src={getHistogramUrl(channelId)}
          alt="Histogram"
          style={{
            width: '100%',
            borderRadius: '8px',
            display: 'block'
          }}
        />
      </div>
  );
}
