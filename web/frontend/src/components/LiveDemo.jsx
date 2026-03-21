import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

const API = 'http://localhost:5000/api';

function LiveDemo() {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = 900;
    canvas.height = 300;
    ctx.fillStyle = '#0f1a31';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
  }, []);

  const pointerPos = (event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const src = event.touches ? event.touches[0] : event;
    return {
      x: ((src.clientX - rect.left) / rect.width) * canvasRef.current.width,
      y: ((src.clientY - rect.top) / rect.height) * canvasRef.current.height
    };
  };

  const start = (e) => {
    setDrawing(true);
    const ctx = canvasRef.current.getContext('2d');
    const { x, y } = pointerPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e) => {
    if (!drawing) return;
    const ctx = canvasRef.current.getContext('2d');
    const { x, y } = pointerPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stop = () => setDrawing(false);

  const clear = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.fillStyle = '#0f1a31';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setResult(null);
  };

  const classify = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/classify`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      const data = await res.json();
      setResult(data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.section
      id="demo"
      className="section container glass-card"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
    >
      <h2>Live Demo</h2>
      <p>Draw a pattern to simulate a corrupted sample and run the backend classify endpoint.</p>
      <canvas
        ref={canvasRef}
        className="drawpad"
        onMouseDown={start}
        onMouseMove={draw}
        onMouseUp={stop}
        onMouseLeave={stop}
        onTouchStart={start}
        onTouchMove={draw}
        onTouchEnd={stop}
      />
      <div className="btn-row">
        <button className="primary" onClick={classify} disabled={loading}>{loading ? 'Processing...' : 'Run Classification'}</button>
        <button className="secondary" onClick={clear}>Clear Pad</button>
      </div>

      {result && (
        <div style={{ marginTop: '1rem' }}>
          <p>
            Prediction: <strong>{result.prediction.class_name}</strong> ({(result.confidence * 100).toFixed(2)}%)
          </p>
          <p>
            PSNR: {result.psnr} dB | SSIM: {result.ssim} | Processing: {result.processingTime} ms
          </p>
        </div>
      )}
    </motion.section>
  );
}

export default LiveDemo;
