import { useEffect, useState } from 'react';
import Select from 'react-select';
import { fetchLabels } from '../api/appliance-api';

export default function ApplianceDropdown({ selected, onChange }) {
  const [options, setOptions] = useState([]);

  useEffect(() => {
    fetchLabels()
      .then(labelsObj => {
        const parsed = Object.entries(labelsObj).map(([id, label]) => ({
          value: parseInt(id, 10),
          label: label
        }));
        setOptions(parsed);
        if (!selected && parsed.length > 0) {
          onChange?.(parsed[0]);
        }
      })
      .catch(err => console.error('Failed to fetch labels:', err));
  }, []);

  const customStyles = {
    control: (base) => ({
      ...base,
      backgroundColor: '#2d3650',
      color: '#e0e6f3',
      border: 'none',
      borderRadius: '8px',
      padding: '2px 4px',
      boxShadow: 'none',
      width: '250px'
    }),
    singleValue: (base) => ({
      ...base,
      color: '#e0e6f3',
    }),
    option: (base, { isFocused }) => ({
      ...base,
      backgroundColor: isFocused ? '#3b4662' : '#2d3650',
      color: '#e0e6f3',
      cursor: 'pointer',
    }),
    menu: (base) => ({
      ...base,
      backgroundColor: '#2d3650',
      zIndex: 9999,
      width: '250px',
    }),
    menuList: (base) => ({
      ...base,
      maxHeight: '200px',
      overflowY: 'auto'
    }),
    menuPortal: (base) => ({
      ...base,
      zIndex: 9999,
      position: 'fixed'
    }),
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
    }}>
      <label style={{ color: '#000000', marginBottom: '8px', fontWeight: 600 }}>Select appliance:</label>
      <Select
        options={options}
        value={selected}
        onChange={onChange}
        styles={customStyles}
        
        menuPlacement="auto"
        menuPosition="absolute"
        menuPortalTarget={document.body}
        menuShouldBlockScroll={true}
      />
    </div>
  );
}
