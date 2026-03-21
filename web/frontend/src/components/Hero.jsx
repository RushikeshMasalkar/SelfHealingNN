import { motion } from 'framer-motion';

const dots = Array.from({ length: 60 }, (_, i) => ({
  id: i,
  x: Math.random() * 100,
  y: Math.random() * 100,
  s: 2 + Math.random() * 4,
  d: 8 + Math.random() * 8
}));

function Hero() {
  return (
    <section id="hero" className="hero section">
      <div className="particles">
        {dots.map((dot) => (
          <motion.span
            key={dot.id}
            style={{
              position: 'absolute',
              left: `${dot.x}%`,
              top: `${dot.y}%`,
              width: `${dot.s}px`,
              height: `${dot.s}px`,
              borderRadius: '50%',
              background: 'rgba(0,212,255,0.8)',
              boxShadow: '0 0 14px rgba(0,212,255,0.7)'
            }}
            animate={{ y: [0, -24, 0], opacity: [0.2, 0.9, 0.2] }}
            transition={{ duration: dot.d, repeat: Infinity, ease: 'easeInOut' }}
          />
        ))}
      </div>
      <div className="container hero-content">
        <motion.h1
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          style={{ fontSize: 'clamp(2rem, 6vw, 4rem)', marginBottom: '0.8rem' }}
        >
          Self-Healing Neural Network for ImageNet-100
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.7 }}
          style={{ maxWidth: '720px', marginBottom: '1.2rem' }}
        >
          A two-stage AI system where a 512-dimensional Convolutional VAE repairs corrupted images before a ResNet-18
          expert performs final classification.
        </motion.p>
        <div>
          <span className="kpi">Clean 94.2%</span>
          <span className="kpi">Noisy 51.7%</span>
          <span className="kpi">Healed 87.6%</span>
        </div>
      </div>
    </section>
  );
}

export default Hero;
