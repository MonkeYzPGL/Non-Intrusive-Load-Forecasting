const BASE_URL = 'http://localhost:5000/forecast';

export async function fetchBaselinePreview(channelId) {
  const url = `${BASE_URL}/baseline/seasonal/preview/${channelId}`;

  const response = await fetch(url);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error?.error || 'Failed to fetch baseline preview');
  }

  return await response.json();
}

export async function fetchPredictionLSTMForDay(channelId, dateStr) {
  const url = `${BASE_URL}/predictieLSTM/${channelId}/${dateStr}`;

  const response = await fetch(url);
  console.log(response);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error?.error || 'Failed to fetch prediction for LSTM for day');
  }

  return await response.json();
}

export async function fetchForecastMetricsLSTM(channelId, dateStr) {
  const response = await fetch(`${BASE_URL}/metricsLSTM/${channelId}/${dateStr}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error?.error || 'Failed to fetch forecast metrics for LSTM');
  }

  return await response.json();
}

export async function fetchPredictionKANForDay(channelId, dateStr) {
  const url = `${BASE_URL}/predictieKAN/${channelId}/${dateStr}`;

  const response = await fetch(url);
  console.log(response);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error?.error || 'Failed to fetch prediction for KAN for day');
  }

  return await response.json();
}

export async function fetchForecastMetricsKAN(channelId, dateStr) {
  const response = await fetch(`${BASE_URL}/metricsKAN/${channelId}/${dateStr}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error?.error || 'Failed to fetch forecast metrics for KAN');
  }

  return await response.json();
}