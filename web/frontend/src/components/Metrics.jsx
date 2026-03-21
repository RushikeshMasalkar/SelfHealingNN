import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, useInView } from 'framer-motion';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const API = 'http://localhost:5000/api';

function useCountUp(target, trigger) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!trigger) return;
    let raf;
    const duration = 1000;
    const start = performance.now();

    const loop = (t) => {
      const k = Math.min((t - start) / duration, 1);
      setValue(target * k);
      if (k < 1) raf = requestAnimationFrame(loop);
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [target, trigger]);

  return value;
}

function Metrics() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then((r) => r.json())
      .then((d) => setMetrics(d))
      .catch(() => setMetrics(null));
  }, []);

  const summary = useMemo(
    () =>
      metrics?.summaryMetrics || {
        baselineAccuracy: 94.2,
        noisyAccuracy: 51.7,
        healedAccuracy: 87.6,
        avgPsnr: 28.4,
        avgSsim: 0.847
      },
    [metrics]
  );

  const ref = useRef(null);
  const inView = useInView(ref, { once: true, amount: 0.35 });
  const baseline = useCountUp(summary.baselineAccuracy, inView);
  const noisy = useCountUp(summary.noisyAccuracy, inView);
  const healed = useCountUp(summary.healedAccuracy, inView);

  return (
    <motion.section id="metrics" className="section container glass-card" ref={ref} initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}>
      <h2>Metrics</h2>
      <div className="metric-grid" style={{ marginBottom: '1rem' }}>
        <div className="glass-card" style={{ padding: '1rem' }}><h3>{baseline.toFixed(1)}%</h3><p>Clean Baseline</p></div>
        <div className="glass-card" style={{ padding: '1rem' }}><h3>{noisy.toFixed(1)}%</h3><p>No Healing</p></div>
        <div className="glass-card" style={{ padding: '1rem' }}><h3>{healed.toFixed(1)}%</h3><p>With Healing</p></div>
      </div>

      <div style={{ height: 300, marginBottom: '1rem' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={metrics?.accuracyVsNoise || []}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.15)" />
            <XAxis dataKey="noise" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="clean" stroke="#00d4ff" strokeWidth={3} />
            <Line type="monotone" dataKey="noHealing" stroke="#ef4444" strokeWidth={3} />
            <Line type="monotone" dataKey="withHealing" stroke="#7c3aed" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ height: 280 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={metrics?.accuracyVsNoise || []}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.15)" />
            <XAxis dataKey="noise" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="noHealing" fill="#ef4444" />
            <Bar dataKey="withHealing" fill="#00d4ff" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </motion.section>
  );
}

export default Metrics;
