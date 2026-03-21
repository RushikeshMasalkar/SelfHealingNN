import { Router } from 'express';

const router = Router();

const classNames = [
  'n01440764_tench',
  'n01443537_goldfish',
  'n01614925_bald_eagle',
  'n01820546_lorikeet',
  'n02085620_chihuahua',
  'n02123045_tabby_cat',
  'n02129165_lion',
  'n02165456_ladybug',
  'n02487347_macaque',
  'n02802426_basketball'
];

function randomFloat(min, max, digits = 4) {
  const val = min + Math.random() * (max - min);
  return Number(val.toFixed(digits));
}

router.post('/classify', async (_req, res) => {
  const started = Date.now();

  await new Promise((resolve) => setTimeout(resolve, 900));

  const classId = Math.floor(Math.random() * classNames.length);
  const top5 = classNames
    .map((name) => ({ class_name: name, confidence: randomFloat(0.05, 0.95) }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5)
    .map((item) => ({ ...item, confidence: Number(item.confidence.toFixed(4)) }));

  const response = {
    prediction: {
      class_id: classId,
      class_name: classNames[classId]
    },
    confidence: top5[0].confidence,
    top5,
    processingTime: Date.now() - started,
    psnr: randomFloat(26.8, 31.6, 2),
    ssim: randomFloat(0.79, 0.91, 3)
  };

  res.json(response);
});

export default router;
