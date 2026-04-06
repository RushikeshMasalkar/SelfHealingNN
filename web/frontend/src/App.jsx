import { motion } from 'framer-motion';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Architecture from './components/Architecture';
import LiveDemo from './components/LiveDemo';
import Metrics from './components/Metrics';
import HowItWorks from './components/HowItWorks';
import TechStack from './components/TechStack';
import Footer from './components/Footer';

function App() {
  return (
    <div className="app-shell">
      <Navbar />
      <Hero />

      <motion.section
        id="problem"
        className="section container glass-card"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.25 }}
        transition={{ duration: 0.6 }}
      >
        <h2>Problem Statement</h2>
        <p>
          Standard classifiers collapse when corruption rises. On CIFAR-100, direct inference drops from
          <strong> 94.2%</strong> on clean images to <strong>51.7%</strong> at heavy noise. This project inserts a
          Convolutional VAE healer before ResNet-18 so the system recovers useful structure before classification.
        </p>
      </motion.section>

      <Architecture />
      <LiveDemo />
      <Metrics />
      <HowItWorks />
      <TechStack />
      <Footer />
    </div>
  );
}

export default App;
