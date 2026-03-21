import { motion } from 'framer-motion';

const steps = [
  'Load a 224x224 RGB sample from ImageNet-100.',
  'Inject Gaussian, Salt & Pepper, or block occlusion noise.',
  'ConvVAE encoder maps corrupted image into 512-dimensional latent space.',
  'ConvVAE decoder reconstructs a denoised approximation.',
  'ResNet-18 predicts class probabilities from healed image.',
  'System reports top prediction, confidence, PSNR, and SSIM.'
];

function HowItWorks() {
  return (
    <section id="how" className="section container glass-card">
      <h2>How It Works</h2>
      <div className="step-grid">
        {steps.map((step, idx) => (
          <motion.div
            key={step}
            className="glass-card"
            initial={{ opacity: 0, x: idx % 2 === 0 ? -50 : 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, amount: 0.35 }}
            transition={{ duration: 0.45, delay: idx * 0.05 }}
            style={{ padding: '1rem 1.1rem' }}
          >
            <strong style={{ color: '#00d4ff' }}>Step {idx + 1}</strong>
            <p style={{ margin: '0.4rem 0 0' }}>{step}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

export default HowItWorks;
