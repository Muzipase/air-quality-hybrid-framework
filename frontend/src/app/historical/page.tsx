'use client';

import React, { useState, useMemo } from 'react';
import { useCity } from '@/lib/city-context';
import { fetchCityHistorical } from '@/lib/api';
import type { HistoricalData, HistoricalDay } from '@/lib/api';

const CITIES = ['Lusaka', 'Ndola', 'Kitwe'];

const POLLUTANT_INFO: Record<string, { label: string; unit: string; color: string }> = {
  pm25: { label: 'PM2.5', unit: 'µg/m³', color: '#3b82f6' },
  pm10: { label: 'PM10', unit: 'µg/m³', color: '#8b5cf6' },
  no2:  { label: 'NO₂', unit: 'µg/m³', color: '#f97316' },
  o3:   { label: 'O₃', unit: 'µg/m³', color: '#22c55e' },
  so2:  { label: 'SO₂', unit: 'µg/m³', color: '#ef4444' },
  co:   { label: 'CO', unit: 'mg/m³', color: '#6366f1' },
};

const PRESETS: { label: string; start: string; end: string }[] = (() => {
  const today = new Date();
  const fmt = (d: Date) => d.toISOString().split('T')[0];
  const end = fmt(new Date(today.getFullYear(), today.getMonth(), today.getDate() - 1));
  return [
    { label: 'Last 3 Months', start: fmt(new Date(today.getFullYear(), today.getMonth() - 3, today.getDate())), end },
    { label: 'Last 6 Months', start: fmt(new Date(today.getFullYear(), today.getMonth() - 6, today.getDate())), end },
    { label: 'Last 12 Months', start: fmt(new Date(today.getFullYear(), today.getMonth() - 12, today.getDate())), end },
    { label: 'All Time', start: '2022-08-15', end },
  ];
})();

function computeMovingAverage(data: (number | null)[], window: number): (number | null)[] {
  const result: (number | null)[] = [];
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = data.slice(start, i + 1).filter((v) => v !== null && v !== undefined);
    result.push(slice.length > 0 ? Math.round((slice.reduce((a, b) => a + b, 0) / slice.length) * 100) / 100 : null);
  }
  return result;
}

// ==================== SVG Line Chart ====================

interface ChartProps {
  daily: HistoricalDay[];
  pollutants: string[];
  showMovingAvg: boolean;
  yLabel: string;
}

function LineChart({ daily, pollutants, showMovingAvg, yLabel }: ChartProps) {
  const W = 900, H = 340, pad = { top: 20, right: 20, bottom: 40, left: 55 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  const { xScale, yMin, yMax, series } = useMemo(() => {
    const dates = daily.map((d) => d.date);
    let allVals: number[] = [];
    const seriesMap: Record<string, (number | null)[]> = {};

    for (const p of pollutants) {
      const raw = daily.map((d) => {
        const v = (d as unknown as Record<string, unknown>)[p];
        return v != null ? Number(v) : null;
      });
      const ma = showMovingAvg ? computeMovingAverage(raw, 7) : raw;
      seriesMap[p] = ma;
      allVals = allVals.concat(ma.filter((v): v is number => v !== null));
    }

    const yMin = Math.max(0, Math.floor(Math.min(...allVals) * 0.9));
    const yMax = Math.ceil(Math.max(...allVals) * 1.1) || 10;
    const xScale = (i: number) => (i / Math.max(dates.length - 1, 1)) * cw;

    return { xScale, yMin, yMax, series: seriesMap };
  }, [daily, pollutants, showMovingAvg]);

  const yScale = (v: number) => ch - ((v - yMin) / (yMax - yMin)) * ch;

  const tickCount = 6;
  const yTicks = Array.from({ length: tickCount }, (_, i) => yMin + (i * (yMax - yMin)) / (tickCount - 1));

  const xLabelStep = Math.max(1, Math.floor(daily.length / 8));
  const xLabels: { idx: number; label: string }[] = [];
  for (let i = 0; i < daily.length; i += xLabelStep) {
    const d = daily[i].date;
    xLabels.push({ idx: i, label: d.slice(5) });
  }

  const buildPath = (vals: (number | null)[]): string => {
    let started = false;
    const segs: string[] = [];
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (v == null) { started = false; continue; }
      const x = xScale(i);
      const y = yScale(v);
      if (!started) { segs.push(`M ${x} ${y}`); started = true; }
      else { segs.push(`L ${x} ${y}`); }
    }
    return segs.join(' ');
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="hist-chart-svg" preserveAspectRatio="xMidYMid meet">
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={pad.left} y1={pad.top + yScale(t) - yScale(yMax) + ch} x2={W - pad.right} y2={pad.top + yScale(t) - yScale(yMax) + ch} stroke="var(--border)" strokeDasharray="4 4" />
          <text x={pad.left - 8} y={pad.top + yScale(t) - yScale(yMax) + ch + 4} textAnchor="end" fontSize={11} fill="var(--text-muted)">
            {Math.round(t)}
          </text>
        </g>
      ))}

      {xLabels.map(({ idx, label }) => (
        <text key={idx} x={pad.left + xScale(idx)} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
          {label}
        </text>
      ))}

      {pollutants.map((p) => {
        const vals = series[p];
        if (!vals) return null;
        const color = POLLUTANT_INFO[p]?.color || '#999';
        return (
          <path key={p} d={buildPath(vals)} fill="none" stroke={color} strokeWidth={showMovingAvg ? 2.5 : 1.5} strokeLinecap="round" strokeLinejoin="round" />
        );
      })}

      <text x={pad.left - 8} y={8} textAnchor="start" fontSize={11} fill="var(--text-muted)">
        {yLabel}
      </text>
    </svg>
  );
}

// ==================== Main Page ====================

export default function HistoricalPage() {
  const { selectedCity, setSelectedCity } = useCity();
  const [startDate, setStartDate] = useState(PRESETS[2].start);
  const [endDate, setEndDate] = useState(PRESETS[2].end);
  const [activePreset, setActivePreset] = useState(2);
  const [selectedPollutants, setSelectedPollutants] = useState<string[]>(['pm25', 'pm10']);
  const [showMovingAvg, setShowMovingAvg] = useState(true);
  const [compareCities, setCompareCities] = useState(false);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<HistoricalData | null>(null);

  const handlePreset = (idx: number) => {
    setActivePreset(idx);
    setStartDate(PRESETS[idx].start);
    setEndDate(PRESETS[idx].end);
  };

  const togglePollutant = (p: string) => {
    setSelectedPollutants((prev) => (prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p]));
  };

  const loadData = async () => {
    setLoading(true);
    try {
      const result = await fetchCityHistorical(selectedCity, startDate, endDate, selectedPollutants);
      setData(result);
    } catch {
      setData(null);
    }
    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') loadData();
  };

  return (
    <div className="hist-page">
      <div className="hist-header">
        <div>
          <h1 className="hist-title">Historical Air Quality Trends</h1>
          <p className="hist-subtitle">Aug 2022 &ndash; present</p>
        </div>
      </div>

      {/* Controls */}
      <div className="hist-controls">
        {/* City selector */}
        <div className="hist-control-group">
          <label className="hist-label">City</label>
          <div className="hist-city-pills">
            {CITIES.map((c) => (
              <button key={c} className={`hist-pill ${selectedCity === c ? 'active' : ''}`} onClick={() => setSelectedCity(c)}>
                {c}
              </button>
            ))}
          </div>
        </div>

        {/* Date range presets */}
        <div className="hist-control-group">
          <label className="hist-label">Date Range</label>
          <div className="hist-city-pills">
            {PRESETS.map((p, i) => (
              <button key={p.label} className={`hist-pill ${activePreset === i ? 'active' : ''}`} onClick={() => handlePreset(i)}>
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Custom dates */}
        <div className="hist-control-group hist-date-row">
          <div className="hist-date-field">
            <label className="hist-label">From</label>
            <input type="date" className="hist-date-input" value={startDate} min="2022-08-15" onChange={(e) => { setStartDate(e.target.value); setActivePreset(-1); }} onKeyDown={handleKeyDown} />
          </div>
          <div className="hist-date-field">
            <label className="hist-label">To</label>
            <input type="date" className="hist-date-input" value={endDate} min={startDate} onChange={(e) => { setEndDate(e.target.value); setActivePreset(-1); }} onKeyDown={handleKeyDown} />
          </div>
        </div>

        {/* Pollutant toggles */}
        <div className="hist-control-group">
          <label className="hist-label">Pollutants</label>
          <div className="hist-city-pills">
            {Object.entries(POLLUTANT_INFO).map(([key, info]) => (
              <button key={key} className={`hist-pill poll-pill ${selectedPollutants.includes(key) ? 'active' : ''}`} style={selectedPollutants.includes(key) ? { borderColor: info.color, color: info.color, background: `${info.color}15` } : undefined} onClick={() => togglePollutant(key)}>
                {info.label}
              </button>
            ))}
          </div>
        </div>

        {/* Options */}
        <div className="hist-control-group hist-options-row">
          <label className="hist-checkbox">
            <input type="checkbox" checked={showMovingAvg} onChange={(e) => setShowMovingAvg(e.target.checked)} />
            7-day moving average
          </label>
          <label className="hist-checkbox">
            <input type="checkbox" checked={compareCities} onChange={(e) => setCompareCities(e.target.checked)} />
            Compare all cities
          </label>
        </div>

        {/* Fetch button */}
        <button className="hist-fetch-btn" onClick={loadData} disabled={loading || selectedPollutants.length === 0}>
          {loading ? 'Loading...' : 'Load Data'}
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="hist-loading">
          <div className="hist-spinner" />
          <p>Fetching historical data from Open-Meteo...</p>
        </div>
      )}

      {/* No data yet */}
      {!loading && !data && (
        <div className="hist-empty">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round">
            <path d="M3 3v18h18" />
            <path d="M7 16l4-5 4 3 5-7" />
          </svg>
          <p>Select a city, date range, and pollutants, then click &quot;Load Data&quot;.</p>
        </div>
      )}

      {/* No data from API */}
      {!loading && data && data.status === 'no_data' && (
        <div className="hist-empty">
          <p>{data.message || 'No data available for this range.'}</p>
        </div>
      )}

      {/* Charts */}
      {!loading && data && data.daily && data.daily.length > 0 && (
        <>
          {compareCities ? (
            <div className="hist-charts-grid">
              {CITIES.map((city) => (
                <CompareChart key={city} city={city} startDate={startDate} endDate={endDate} pollutants={selectedPollutants} showMovingAvg={showMovingAvg} />
              ))}
            </div>
          ) : (
            <div className="hist-chart-section">
              <div className="hist-chart-wrap">
                <LineChart daily={data.daily} pollutants={selectedPollutants} showMovingAvg={showMovingAvg} yLabel="Concentration" />
              </div>

              {/* Legend */}
              <div className="hist-legend">
                {selectedPollutants.map((p) => {
                  const info = POLLUTANT_INFO[p];
                  return (
                    <div key={p} className="hist-legend-item">
                      <span className="hist-legend-dot" style={{ background: info.color }} />
                      {info.label} ({info.unit})
                    </div>
                  );
                })}
                {showMovingAvg && <span className="hist-legend-note">Lines show 7-day moving average</span>}
              </div>

              {/* Stats cards */}
              {data.stats && (
                <div className="hist-stats-grid">
                  {selectedPollutants.map((p) => {
                    const s = data.stats[p];
                    const info = POLLUTANT_INFO[p];
                    if (!s) return null;
                    return (
                      <div key={p} className="hist-stat-card">
                        <div className="hist-stat-header" style={{ borderLeftColor: info.color }}>
                          <span className="hist-stat-name">{info.label}</span>
                          <span className={`hist-stat-trend trend-${s.trend}`}>
                            {s.trend === 'increasing' ? '↑' : s.trend === 'decreasing' ? '↓' : '→'} {s.trend}
                          </span>
                        </div>
                        <div className="hist-stat-body">
                          <div className="hist-stat-val">
                            <span className="hist-stat-num">{s.mean}</span>
                            <span className="hist-stat-unit">avg {info.unit}</span>
                          </div>
                          <div className="hist-stat-range">
                            <span>Min: {s.min}</span>
                            <span>Max: {s.max}</span>
                          </div>
                          <div className="hist-stat-days">{s.count} days of data</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ==================== Compare Chart (single city) ====================

function CompareChart({ city, startDate, endDate, pollutants, showMovingAvg }: { city: string; startDate: string; endDate: string; pollutants: string[]; showMovingAvg: boolean }) {
  const [data, setData] = useState<HistoricalData | null>(null);
  const [loading, setLoading] = useState(true);

  React.useEffect(() => {
    setLoading(true);
    fetchCityHistorical(city, startDate, endDate, pollutants).then((d) => {
      setData(d);
      setLoading(false);
    });
  }, [city, startDate, endDate, pollutants.join(',')]);

  if (loading) return <div className="hist-compare-loading"><div className="hist-spinner" /><p>Loading {city}...</p></div>;
  if (!data || !data.daily || data.daily.length === 0) return <div className="hist-compare-empty"><p>No data for {city}</p></div>;

  return (
    <div className="hist-compare-card">
      <h3 className="hist-compare-title">{city}</h3>
      <div className="hist-chart-wrap">
        <LineChart daily={data.daily} pollutants={pollutants} showMovingAvg={showMovingAvg} yLabel="µg/m³" />
      </div>
      {data.stats && (
        <div className="hist-compare-stats">
          {pollutants.map((p) => {
            const s = data.stats[p];
            const info = POLLUTANT_INFO[p];
            if (!s) return null;
            return (
              <span key={p} className="hist-compare-stat" style={{ color: info.color }}>
                {info.label}: avg {s.mean} {s.trend === 'increasing' ? '↑' : s.trend === 'decreasing' ? '↓' : '→'}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
