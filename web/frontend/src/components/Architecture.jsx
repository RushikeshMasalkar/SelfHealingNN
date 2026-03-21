import { motion } from 'framer-motion';

function Architecture() {
  return (
    <section id="architecture" className="section container glass-card">
      <motion.h2 initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
        Architecture
      </motion.h2>
      <p>Animated dataflow from corruption injection through ConvVAE healing and ResNet-18 inference.</p>

      <svg viewBox="0 0 900 260" width="100%" height="260" style={{ marginTop: '1rem' }}>
        <defs>
          <linearGradient id="pipeGlow" x1="0" x2="1">
            <stop offset="0%" stopColor="#00d4ff" />
            <stop offset="100%" stopColor="#7c3aed" />
          </linearGradient>
        </defs>

        {[
          { x: 20, label: 'Input\n224x224 RGB' },
          { x: 200, label: 'Noise\nInjector' },
          { x: 380, label: 'ConvVAE\nLatent z=512' },
          { x: 580, label: 'ResNet-18\nClassifier' },
          { x: 760, label: 'Top-5\nPrediction' }
        ].map((box) => (
          <g key={box.x}>
            <rect x={box.x} y="90" rx="16" ry="16" width="130" height="90" fill="rgba(255,255,255,0.06)" stroke="url(#pipeGlow)" />
            <text x={box.x + 65} y="125" fill="#eaf5ff" textAnchor="middle" fontSize="14" fontFamily="Sora">
              {box.label.split('\n').map((line, idx) => (
                <tspan key={line} x={box.x + 65} dy={idx === 0 ? 0 : 18}>{line}</tspan>
              ))}
            </text>
          </g>
        ))}

        {[150, 330, 530, 710].map((start) => (
          <g key={start}>
            <line x1={start} y1="135" x2={start + 50} y2="135" stroke="url(#pipeGlow)" strokeWidth="3" />
            <motion.circle
              cx={start + 8}
              cy={135}
              r="6"
              fill="#00d4ff"
              animate={{ cx: [start + 8, start + 42, start + 8] }}
              transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
            />
          </g>
        ))}
      </svg>
    </section>
  );
}

export default Architecture;
