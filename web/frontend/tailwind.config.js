/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        navy: '#0a0f1e',
        electric: '#00d4ff',
        neon: '#7c3aed'
      }
    }
  },
  plugins: []
};
