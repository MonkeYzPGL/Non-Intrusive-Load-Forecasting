import { useEffect, useState } from 'react';
import styles from './ApplianceDetails.module.css';
import ApplianceDropdown from '../../generic-components/ApplianceDropdown';
import DetailsTable from './DetailsTable';
import HistogramImage from './HistogramImage';
import ACFImage from './ACFImage';
import { fetchChannelCSV } from '../../api/appliance-api';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

const ApplianceDetails = () => {
  const [selectedAppliance, setSelectedAppliance] = useState(null);
  const [selectedDate, setSelectedDate] = useState('');
  const [availableDates, setAvailableDates] = useState([]);
  const [dateRange, setDateRange] = useState({ min: null, max: null });

  useEffect(() => {
    if (!selectedAppliance) return;

    fetchChannelCSV(selectedAppliance.value)
      .then(data => {
        const uniqueDates = [...new Set(
          data.map(row => new Date(row.timestamp).toISOString().split('T')[0])
        )];
        setAvailableDates(uniqueDates);
        setSelectedDate(uniqueDates[0] || '');
        const sortedDates = [...uniqueDates].sort();
        setDateRange({
          min: new Date(sortedDates[0]),
          max: new Date(sortedDates[sortedDates.length - 1])
        });
      })
      .catch(err => console.error('Eroare CSV:', err));
  }, [selectedAppliance]);

  return (
    <div className={styles.container}>
      <div className={styles.selectionRow}>
        <ApplianceDropdown
          selected={selectedAppliance}
          onChange={setSelectedAppliance}
        />

        {availableDates.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', marginLeft: '10px' }}>
            <label style={{ color: '#000', marginBottom: '8px', fontWeight: 600 }}>Select date:</label>
            <DatePicker
              selected={selectedDate ? new Date(selectedDate) : null}
              onChange={(date) => {
                const iso = date.toISOString().split('T')[0];
                setSelectedDate(iso);
              }}
              includeDates={availableDates.map(dateStr => new Date(dateStr))}
              minDate={dateRange.min}
              maxDate={dateRange.max}
              dateFormat="yyyy-MM-dd"
              placeholderText="Select a date"
              calendarClassName={styles.customCalendar}
              popperPlacement="bottom"
              showMonthDropdown
              showYearDropdown
              dropdownMode="select"
              customInput={
                <input
                  readOnly
                  className={styles.customDatePicker}
                />
              }
            />
          </div>
        )}
      </div>

      <h1 className={styles.title}>
        {selectedAppliance
          ? `Now showing the details for ${selectedAppliance.label}`
          : 'Please select an appliance to view details'}
      </h1>

      {selectedAppliance && selectedDate && (
        <>
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'flex-start',
          gap: '48px',
          flexWrap: 'wrap',
          padding: '0 24px'
        }}>
          <DetailsTable
            channelId={selectedAppliance.value}
            selectedDate={selectedDate}
          />
          <div className={styles.graphs}>
          <HistogramImage channelId={selectedAppliance.value} />
          <ACFImage channelId={selectedAppliance.value} />
        </div>
        </div>
        </>
      )}
    </div>
  );
};

export default ApplianceDetails;
