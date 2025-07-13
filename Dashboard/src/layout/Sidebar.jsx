import { Link, useLocation } from 'react-router-dom';
import styles from './Sidebar.module.css';

export default function Sidebar({ onClose }) {
  const location = useLocation();

  return (
    <aside className={styles.sidebar}>
      {onClose && (
        <button className={styles.closeBtn} onClick={onClose} aria-label="Close sidebar">✕</button>
      )}
      <div className={styles.logo}>⚡ NILF</div>
      <nav className={styles.menu}>
        <Link 
          to="/" 
          className={`${styles.menuItem} ${location.pathname === '/' ? styles.active : ''}`}
        >
          Details
        </Link>
        <Link 
          to="/comparison" 
          className={`${styles.menuItem} ${location.pathname === '/comparison' ? styles.active : ''}`}
        >
          Comparison
        </Link>
      </nav>
    </aside>
  );
}
