import express, { Request, Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { VisualizationRequest, VisualizationResponse } from './types/transformer';
import { processTransformer } from './utils/transformer';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Serve static files in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static('client/build'));
}

app.post('/api/visualize', (req: Request<{}, {}, VisualizationRequest>, res: Response<VisualizationResponse>) => {
  const { text, numLayers = 6, numHeads = 8, embeddingDim = 512 } = req.body;
  
  if (!text?.trim()) {
    return res.status(400).json({ 
      error: 'Text input required' 
    } as any);
  }

  const state = processTransformer(text, numLayers, numHeads, embeddingDim);
  
  res.json({
    state,
    metadata: {
      numLayers,
      numHeads,
      embeddingDim,
      vocabSize: state.tokens.length
    }
  });
});

app.get('/api/health', (_req: Request, res: Response) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Serve React app for all other routes
if (process.env.NODE_ENV === 'production') {
  app.get('*', (_req: Request, res: Response) => {
    res.sendFile(path.join(__dirname, '../../client/build/index.html'));
  });
}

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
