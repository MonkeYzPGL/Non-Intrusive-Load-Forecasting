import Papa from 'papaparse';

const BASE_URL = 'http://localhost:5000/details';

export async function fetchChannelDetails(channelId) {
  const response = await fetch(`${BASE_URL}/get_details/${channelId}`);
  const text = await response.text();

  return new Promise((resolve, reject) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => resolve(result.data),
      error: (err) => reject(err)
    });
  });
}

export function getHistogramUrl(channelId) {
  return `${BASE_URL}/histogram/${channelId}`;
}

export function getACFUrl(channelId){
  return `${BASE_URL}/acf/${channelId}`;
}

export async function fetchChannelCSV(channelId) {
  const response = await fetch(`${BASE_URL}/csv/${channelId}`);

  if (!response.ok) {
    throw new Error(`Error fetching CSV for channel ${channelId}: ${response.statusText}`);
  }

  const text = await response.text();
  return new Promise((resolve, reject) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => resolve(result.data),
      error: (err) => reject(err)
    });
  });
}

export async function fetchDailyConsumption(channelId, dateStr) {
  const response = await fetch(`${BASE_URL}/consumption/${channelId}/${dateStr}`);

  if (!response.ok) {
    throw new Error(`Error fetching daily consumption: ${response.statusText}`);
  }

  return await response.json();
}

export async function fetchLabels() {
  const response = await fetch(`${BASE_URL}/labels`);
  if (!response.ok) {
    throw new Error(`Error fetching labels: ${response.statusText}`);
  }
  return await response.json();
}

export async function fetchRecentDownsampledData(channelId) {
  const response = await fetch(`${BASE_URL}/downsampled_json/${channelId}`);
  if (!response.ok) {
    throw new Error(`Error fetching downsampled data for channel ${channelId}: ${response.statusText}`);
  }
  return await response.json();
}

