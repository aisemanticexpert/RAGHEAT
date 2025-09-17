import React, { useState, useEffect } from 'react';
import { TrendingUp, Activity, Globe, Clock } from 'lucide-react';
import './Header.css';

const Header = () => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState('OPEN');

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
      
      // Simple market hours check (9:30 AM - 4:00 PM EST)
      const now = new Date();
      const hours = now.getHours();
      const isWeekday = now.getDay() >= 1 && now.getDay() <= 5;
      
      if (isWeekday && hours >= 9 && hours < 16) {
        setMarketStatus('OPEN');
      } else {
        setMarketStatus('CLOSED');
      }
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <header className="header">
      <div className="header-container">
        <div className="brand-section">
          <div className="logo">
            <TrendingUp size={32} className="logo-icon" />
            <span className="brand-name">RAGHEAT</span>
          </div>
          <div className="tagline">
            Real-time AI-Guided Heat Analysis & Trading
          </div>
        </div>

        <div className="header-center">
          <div className="status-indicators">
            <div className={`market-status ${marketStatus.toLowerCase()}`}>
              <Activity size={16} />
              <span>Market {marketStatus}</span>
            </div>
            <div className="live-indicator">
              <div className="pulse-dot"></div>
              <span>LIVE DATA</span>
            </div>
          </div>
        </div>

        <div className="header-right">
          <div className="time-section">
            <Clock size={16} />
            <div className="time-display">
              <div className="current-time">
                {currentTime.toLocaleTimeString('en-US', {
                  hour12: false,
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit'
                })}
              </div>
              <div className="current-date">
                {currentTime.toLocaleDateString('en-US', {
                  weekday: 'short',
                  month: 'short',
                  day: 'numeric'
                })}
              </div>
            </div>
          </div>
          <div className="global-indicator">
            <Globe size={16} />
            <span>Global Markets</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;