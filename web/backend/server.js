import cors from 'cors';
import express from 'express';

import classifyRouter from './routes/classify.js';
import metricsRouter from './routes/metrics.js';

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', service: 'self-healing-nn-backend' });
});

app.use('/api', classifyRouter);
app.use('/api', metricsRouter);

app.get('/api/samples', (_req, res) => {
  res.json({
    samples: [
      '/data/samples/sample_1.jpg',
      '/data/samples/sample_2.jpg',
      '/data/samples/sample_3.jpg',
      '/data/samples/sample_4.jpg',
      '/data/samples/sample_5.jpg'
    ]
  });
});

app.listen(port, () => {
  console.log(`Backend listening on http://localhost:${port}`);
});
