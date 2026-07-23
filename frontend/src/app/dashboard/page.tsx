'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { fetchCityAirQuality, fetchCityForecast, fetchLiveWeather, fetchLiveWeatherHourly } from '@/lib/api';
import { useCity } from '@/lib/city-context';
import type { CityAirQuality, ForecastData, HourlyForecast, DailyForecast } from '@/types';

const CITY_INFO: Record<string, { country: string; province: string }> = {
  Lusaka: { country: 'Zambia', province: 'Lusaka Province' },
  Ndola: { country: 'Zambia', province: 'Copperbelt Province' },
  Kitwe: { country: 'Zambia', province: 'Copperbelt Province' },
};

const pollutantLabels: Record<string, { label: string; unit: string; whoAnnual?: number; who24h?: number }> = {
  pm25: { label: 'PM2.5', unit: 'µg/m³', whoAnnual: 5, who24h: 15 },
  pm10: { label: 'PM10', unit: 'µg/m³', whoAnnual: 15, who24h: 45 },
  no2: { label: 'NO₂', unit: 'µg/m³', whoAnnual: 10, who24h: 25 },
  so2: { label: 'SO₂', unit: 'µg/m³', whoAnnual: 40, who24h: 40 },
  co: { label: 'CO', unit: 'mg/m³', whoAnnual: 4, who24h: 4 },
  o3: { label: 'O₃', unit: 'µg/m³', whoAnnual: 100, who24h: 100 },
};

function getCategoryColor(category: string): string {
  switch (category) {
    case 'Good': return '#22c55e';
    case 'Moderate': return '#eab308';
    case 'Unhealthy': return '#f97316';
    case 'Very Unhealthy': return '#ef4444';
    case 'Hazardous': return '#7f1d1d';
    default: return '#eab308';
  }
}

function getCategoryBg(category: string): string {
  switch (category) {
    case 'Good': return '#f0fdf4';
    case 'Moderate': return '#fefce8';
    case 'Unhealthy': return '#fff7ed';
    case 'Very Unhealthy': return '#fef2f2';
    case 'Hazardous': return '#450a0a';
    default: return '#fefce8';
  }
}

function getAQIValue(pm25: number) {
  if (pm25 <= 12.0) return Math.round((pm25 / 12.0) * 50);
  if (pm25 <= 35.4) return Math.round(50 + ((pm25 - 12.0) / (35.4 - 12.0)) * 50);
  if (pm25 <= 55.4) return Math.round(100 + ((pm25 - 35.4) / (55.4 - 35.4)) * 50);
  if (pm25 <= 150.4) return Math.round(150 + ((pm25 - 55.4) / (150.4 - 35.4)) * 100);
  return Math.round(200 + ((pm25 - 150.4) / (500 - 150.4)) * 300);
}

function formatHour(timestamp: string): string {
  const d = new Date(timestamp);
  return d.toLocaleTimeString('en-US', { hour: 'numeric', hour12: true });
}

function formatDayName(dateStr: string): string {
  const d = new Date(dateStr + 'T12:00:00');
  const today = new Date();
  today.setHours(12, 0, 0, 0);
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === tomorrow.toDateString()) return 'Tomorrow';
  return d.toLocaleDateString('en-US', { weekday: 'short' });
}

function TempIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M14 14.76V3.5a2.5 2.5 0 00-5 0v11.26a4.5 4.5 0 105 0z" />
    </svg>
  );
}

function WindIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M9.59 4.59A2 2 0 1111 8H2m10.59 11.41A2 2 0 1014 16H2m15.73-8.27A2.5 2.5 0 1119.5 12H2" />
    </svg>
  );
}

function HumidityIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M12 2.69l5.66 5.66a8 8 0 11-11.31 0z" />
    </svg>
  );
}

function WeatherMini({ temp, humidity, wind }: { temp: number; humidity: number; wind: number }) {
  return (
    <div className="forecast-weather-row">
      <span className="forecast-weather-item"><TempIcon /> {temp}°</span>
      <span className="forecast-weather-item"><WindIcon /> {wind} km/h</span>
      <span className="forecast-weather-item"><HumidityIcon /> {humidity}%</span>
    </div>
  );
}

export default function Dashboard() {
  const { selectedCity } = useCity();
  const [data, setData] = useState<CityAirQuality | null>(null);
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [liveWeather, setLiveWeather] = useState<{ temperature: number; humidity: number; wind_speed: number } | null>(null);
  const [forecastWeatherMap, setForecastWeatherMap] = useState<Record<string, { temperature: number; humidity: number; wind_speed: number }> | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState('');

  const loadData = useCallback(async () => {
    setLoading(true);
    const [aqResult, forecastResult] = await Promise.all([
      fetchCityAirQuality(selectedCity),
      fetchCityForecast(selectedCity, 3),
    ]);
    setData(aqResult);
    setForecast(forecastResult);

    if (aqResult && aqResult.readings.temperature === 0 && aqResult.readings.humidity === 0) {
      const wx = await fetchLiveWeather(selectedCity);
      if (wx) setLiveWeather(wx);
    } else {
      setLiveWeather(null);
    }

    const hasZeroWeather = forecastResult?.hourly?.some(h => h.temperature === 0 && h.humidity === 0) ?? false;
    if (hasZeroWeather) {
      const wxMap = await fetchLiveWeatherHourly(selectedCity, 3);
      if (wxMap) setForecastWeatherMap(wxMap);
    } else {
      setForecastWeatherMap(null);
    }

    if (aqResult) setLastUpdate(new Date().toLocaleTimeString());
    setLoading(false);
  }, [selectedCity]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 120000);
    return () => clearInterval(interval);
  }, [loadData]);

  const hourlyNow = forecast?.hourly?.slice(0, 48).map(h => {
    if (forecastWeatherMap && h.temperature === 0) {
      const wx = forecastWeatherMap[h.timestamp];
      if (wx) {
        return { ...h, temperature: wx.temperature, humidity: wx.humidity, wind_speed: wx.wind_speed };
      }
    }
    return h;
  }) || [];
  const dailyData = (forecast?.daily || []).map(d => {
    if (forecastWeatherMap && d.temp_max === 0) {
      const dayHours = Object.entries(forecastWeatherMap)
        .filter(([ts]) => ts.startsWith(d.date))
        .map(([, wx]) => wx);
      if (dayHours.length > 0) {
        const temps = dayHours.map(h => h.temperature);
        const humids = dayHours.map(h => h.humidity);
        const winds = dayHours.map(h => h.wind_speed);
        return {
          ...d,
          temp_max: Math.max(...temps),
          temp_min: Math.min(...temps),
          humidity_avg: Math.round(humids.reduce((a, b) => a + b, 0) / humids.length),
          wind_max: Math.max(...winds),
        };
      }
    }
    return d;
  });
  const aqiValue = hourlyNow.length > 0 ? hourlyNow[0].aqi : (data ? getAQIValue(data.readings.pm25) : 0);
  const categoryColor = hourlyNow.length > 0 ? getCategoryColor(hourlyNow[0].category) : (data ? getCategoryColor(data.category) : '#eab308');
  const cityInfo = CITY_INFO[selectedCity] || { country: 'Zambia', province: '' };
  const currentPm25 = hourlyNow.length > 0 ? hourlyNow[0].pm25 : (data?.readings.pm25 ?? 0);
  const currentTemp = hourlyNow.length > 0 ? hourlyNow[0].temperature : (liveWeather?.temperature ?? data?.readings.temperature ?? 0);
  const currentHumidity = hourlyNow.length > 0 ? hourlyNow[0].humidity : (liveWeather?.humidity ?? data?.readings.humidity ?? 0);
  const currentWind = hourlyNow.length > 0 ? hourlyNow[0].wind_speed : (liveWeather?.wind_speed ?? data?.readings.wind_speed ?? 0);
  const currentCategory = hourlyNow.length > 0 ? hourlyNow[0].category : (data?.category ?? 'Moderate');

  return (
    <div className="dash-container">
      {/* Breadcrumb */}
      <nav className="breadcrumb">
        <span className="breadcrumb-item">World</span>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-item">{cityInfo.country}</span>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-item">{cityInfo.province}</span>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-current">{selectedCity}</span>
      </nav>

      {loading && !data ? (
        <div className="home-loading">
          <div className="loading-spinner-lg" />
          <p>Checking air quality in {selectedCity}...</p>
        </div>
      ) : data && data.status === 'ok' ? (
        <>
          {/* AQI Hero */}
          <section className="aqi-hero-card">
            <div className="aqi-hero-left">
              <div className="aqi-hero-big-number" style={{ color: categoryColor }}>{aqiValue}</div>
              <div className="aqi-hero-us-label">US AQI⁺</div>
              <div className="aqi-hero-badge" style={{ background: categoryColor }}>{currentCategory}</div>
            </div>
            <div className="aqi-hero-divider" />
            <div className="aqi-hero-right">
              <div className="aqi-hero-pollutant">
                <span className="aqi-hero-pollutant-label">Main pollutant</span>
                <span className="aqi-hero-pollutant-value">PM2.5 — {currentPm25} µg/m³</span>
              </div>
              <WeatherMini temp={currentTemp} humidity={currentHumidity} wind={currentWind} />
              <div className="aqi-hero-advice" style={{ color: categoryColor }}>{data.health?.message ?? 'Air quality data loading...'}</div>
            </div>
          </section>

          {/* Attribution */}
          <div className="dash-attribution">
            Data from Open-Meteo & OpenAQ • {lastUpdate && `Updated ${lastUpdate}`}
          </div>

          {/* Hourly Forecast */}
          {hourlyNow.length > 0 && (
            <section className="dash-section animate-fade-in-up delay-100">
              <h2 className="dash-section-title">Hourly forecast</h2>
              <p className="dash-section-sub">{selectedCity} air quality index (AQI⁺) forecast</p>
              <div className="hourly-scroll">
                {hourlyNow.map((h, i) => (
                  <div key={h.timestamp} className={`hourly-card ${i === 0 ? 'hourly-card-now' : ''}`}>
                    <span className="hourly-time">{i === 0 ? 'Now' : formatHour(h.timestamp)}</span>
                    <span className="hourly-aqi" style={{ color: getCategoryColor(h.category) }}>{h.aqi}</span>
                    <span className="hourly-temp"><TempIcon /> {h.temperature}°</span>
                    <span className="hourly-wind"><WindIcon /> {h.wind_speed}</span>
                    <span className="hourly-humid"><HumidityIcon /> {h.humidity}%</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Daily Forecast */}
          {dailyData.length > 0 && (
            <section className="dash-section animate-fade-in-up delay-200">
              <h2 className="dash-section-title">Daily forecast</h2>
              <p className="dash-section-sub">{selectedCity} air quality index (AQI⁺) forecast</p>
              <div className="daily-grid">
                {dailyData.map((d) => (
                  <div key={d.date} className="daily-card">
                    <span className="daily-day">{formatDayName(d.date)}</span>
                    <span className="daily-aqi" style={{ color: getCategoryColor(d.category) }}>{d.aqi_max}</span>
                    <span className="daily-temp-range">{d.temp_max}° / {d.temp_min}°</span>
                    <WeatherMini temp={d.temp_max} humidity={d.humidity_avg} wind={d.wind_max} />
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Air Pollutants Detail */}
          <section className="dash-section animate-fade-in-up delay-300">
            <h2 className="dash-section-title">Air pollutants</h2>
            <p className="dash-section-sub">What is the current air quality in {selectedCity}?</p>

            {/* PM2.5 Focus Card */}
            <div className="pollutant-focus-card">
              <div className="pollutant-focus-header">
                <div>
                  <span className="pollutant-focus-name">PM2.5</span>
                  <span className="pollutant-focus-desc">Fine particles (≤ 2.5 µm)</span>
                </div>
                <div className="pollutant-focus-value">
                  <span className="pollutant-focus-number">{currentPm25}</span>
                  <span className="pollutant-focus-unit">µg/m³</span>
                </div>
              </div>
              <div className="pollutant-bar-container">
                <div className="pollutant-bar-track">
                  <div
                    className="pollutant-bar-fill"
                    style={{
                      width: `${Math.min((currentPm25 / 150) * 100, 100)}%`,
                      background: getCategoryColor(getAQIValue(currentPm25) <= 50 ? 'Good' : getAQIValue(currentPm25) <= 100 ? 'Moderate' : 'Unhealthy'),
                    }}
                  />
                </div>
                <div className="pollutant-bar-guidelines">
                  <div className="pollutant-bar-guideline">
                    <span className="pollutant-bar-guideline-line" style={{ left: `${(5 / 150) * 100}%` }} />
                    <span className="pollutant-bar-guideline-label">WHO annual<br/>5 µg/m³</span>
                  </div>
                  <div className="pollutant-bar-guideline">
                    <span className="pollutant-bar-guideline-line" style={{ left: `${(15 / 150) * 100}%` }} />
                    <span className="pollutant-bar-guideline-label">WHO 24h<br/>15 µg/m³</span>
                  </div>
                </div>
                <div className="pollutant-bar-scale">
                  <span>0</span>
                  <span>50</span>
                  <span>100</span>
                  <span>150 µg/m³</span>
                </div>
              </div>
              <p className="pollutant-who-note">
                PM2.5 concentration is currently <strong>{(currentPm25 / 5).toFixed(1)}×</strong> the World Health Organization annual PM2.5 guideline value.
              </p>
            </div>

            {/* Other Pollutants Grid */}
            <div className="pollutant-grid">
              {Object.entries(pollutantLabels).filter(([k]) => k !== 'pm25').map(([key, info]) => {
                const value = data.readings[key as keyof typeof data.readings];
                const whoVal = info.whoAnnual || 100;
                const ratio = value / whoVal;
                return (
                  <div key={key} className="pollutant-mini-card">
                    <div className="pollutant-mini-header">
                      <span className="pollutant-mini-name">{info.label}</span>
                      <span className="pollutant-mini-value">{value} <span className="pollutant-mini-unit">{info.unit}</span></span>
                    </div>
                    <div className="pollutant-mini-bar">
                      <div
                        className="pollutant-mini-bar-fill"
                        style={{
                          width: `${Math.min(ratio * 100, 100)}%`,
                          background: ratio > 1 ? '#ef4444' : ratio > 0.7 ? '#f97316' : '#22c55e',
                        }}
                      />
                    </div>
                    <span className="pollutant-mini-who">WHO: {whoVal} {info.unit}</span>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Health Recommendations */}
          <section className="dash-section animate-fade-in-up delay-400">
            <h2 className="dash-section-title">Health recommendations</h2>
            <div className="health-grid">
              <div className="health-card">
                <div className="health-icon health-icon-exercise">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M18 8a2 2 0 110-4 2 2 0 010 4z" /><path d="M22 14l-4.5-3-3 5-4.5-3L6 18" />
                  </svg>
                </div>
                <span className="health-label">Exercise</span>
                <span className="health-action">{aqiValue > 100 ? 'Avoid outdoor exercise' : 'Outdoor exercise is fine'}</span>
              </div>
              <div className="health-card">
                <div className="health-icon health-icon-window">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 12h18" /><path d="M12 3v18" />
                  </svg>
                </div>
                <span className="health-label">Windows</span>
                <span className="health-action">{aqiValue > 100 ? 'Close your windows to avoid dirty outdoor air' : 'Windows can remain open'}</span>
              </div>
              <div className="health-card">
                <div className="health-icon health-icon-mask">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M6.09 13.28a8.96 8.96 0 010 1.44M17.91 13.28a8.96 8.96 0 000 1.44" /><path d="M12 2a8 8 0 00-8 8c0 3.5 2 6.5 5 8v2h6v-2c3-1.5 5-4.5 5-8a8 8 0 00-8-8z" />
                  </svg>
                </div>
                <span className="health-label">Mask</span>
                <span className="health-action">{aqiValue > 100 ? 'Wear a mask outdoors' : 'No mask needed outdoors'}</span>
              </div>
              <div className="health-card">
                <div className="health-icon health-icon-purifier">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  </svg>
                </div>
                <span className="health-label">Air Purifier</span>
                <span className="health-action">{aqiValue > 100 ? 'Run an air purifier indoors' : 'Air purifier not required'}</span>
              </div>
            </div>
          </section>
        </>
      ) : (
        <div className="home-error">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 8v4M12 16h.01" />
          </svg>
          <h2>Could not load air quality data</h2>
          <p>Please check your connection and try again.</p>
          <button className="home-refresh-btn" onClick={loadData}>Try Again</button>
        </div>
      )}

      <footer className="home-footer">
        <div className="flag-accent" />
        <p>AirQ Zambia &mdash; Keeping communities informed about air quality</p>
      </footer>
    </div>
  );
}
