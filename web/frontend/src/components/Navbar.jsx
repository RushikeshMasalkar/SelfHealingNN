import { useState } from 'react';

const links = [
  { id: 'hero', label: 'Hero' },
  { id: 'problem', label: 'Problem' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'demo', label: 'Live Demo' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'how', label: 'How It Works' },
  { id: 'stack', label: 'Tech Stack' }
];

function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="container nav glass-card">
      <div className="brand">Self-Healing NN</div>
      <button className="hamburger" onClick={() => setOpen((v) => !v)} aria-label="Toggle menu">
        Menu
      </button>
      <nav className={`nav-links ${open ? 'open' : ''}`}>
        {links.map((link) => (
          <a key={link.id} href={`#${link.id}`} onClick={() => setOpen(false)}>
            {link.label}
          </a>
        ))}
      </nav>
    </header>
  );
}

export default Navbar;
