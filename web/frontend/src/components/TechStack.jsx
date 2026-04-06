import { motion } from 'framer-motion';

const stack = [
  'PyTorch',
  'ConvVAE',
  'ResNet-18',
  'CIFAR-100',
  'Node.js + Express',
  'React + Vite',
  'Framer Motion',
  'Recharts'
];

function TechStack() {
  return (
    <section id="stack" className="section container glass-card">
      <h2>Tech Stack</h2>
      <div className="tech-grid">
        {stack.map((item, i) => (
          <motion.div
            key={item}
            className="glass-card"
            style={{ padding: '1rem', textAlign: 'center' }}
            initial={{ opacity: 0, y: 22 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ delay: i * 0.05 }}
          >
            {item}
          </motion.div>
        ))}
      </div>
    </section>
  );
}

export default TechStack;
