import { TokenData, AttentionHead, LayerOutput, TransformerState } from '../types/transformer';

const createEmbedding = (token: string, dim: number, seed: number): number[] => {
  const hash = token.split('').reduce((acc, char) => acc + char.charCodeAt(0) * seed, 0);
  return Array.from({ length: dim }, (_, i) => 
    Math.sin(hash * (i + 1) * 0.01) * Math.cos(hash * (i + 1) * 0.02)
  );
};

const createPositionalEncoding = (position: number, dim: number): number[] => {
  return Array.from({ length: dim }, (_, i) => {
    const angle = position / Math.pow(10000, (2 * i) / dim);
    return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  });
};

const addVectors = (v1: number[], v2: number[]): number[] => 
  v1.map((val, i) => val + v2[i]);

const matrixMultiply = (a: number[], b: number[]): number => 
  a.reduce((sum, val, i) => sum + val * b[i], 0);

const softmax = (scores: number[]): number[] => {
  const max = Math.max(...scores);
  const exp = scores.map(s => Math.exp(s - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(e => e / sum);
};

const computeAttention = (
  queries: number[][],
  keys: number[][],
  values: number[][],
  dim: number
): { scores: number[][]; weights: number[][]; output: number[][] } => {
  const scale = Math.sqrt(dim);
  
  const scores = queries.map(q => 
    keys.map(k => matrixMultiply(q, k) / scale)
  );
  
  const weights = scores.map(softmax);
  
  const output = weights.map(w =>
    values[0].map((_, i) =>
      w.reduce((sum, weight, j) => sum + weight * values[j][i], 0)
    )
  );
  
  return { scores, weights, output };
};

export const processTransformer = (
  text: string,
  numLayers: number = 6,
  numHeads: number = 8,
  embeddingDim: number = 512
): TransformerState => {
  const tokens = text.toLowerCase().split(/\s+/).filter(Boolean);
  
  const tokenData: TokenData[] = tokens.map((token, pos) => {
    const embedding = createEmbedding(token, embeddingDim, 42);
    const posEncoding = createPositionalEncoding(pos, embeddingDim);
    return {
      token,
      embedding: addVectors(embedding, posEncoding),
      position: pos
    };
  });

  const processLayer = (input: TokenData[], layerIdx: number): LayerOutput => {
    const headDim = Math.floor(embeddingDim / numHeads);
    
    const attentionHeads: AttentionHead[] = Array.from({ length: numHeads }, (_, h) => {
      const start = h * headDim;
      const end = start + headDim;
      
      const queries = input.map(t => t.embedding.slice(start, end));
      const keys = input.map(t => t.embedding.slice(start, end));
      const values = input.map(t => t.embedding.slice(start, end));
      
      const { scores, weights } = computeAttention(queries, keys, values, headDim);
      
      return { query: queries[0], key: keys[0], value: values[0], scores, weights };
    });

    const ffOutput = input.map(t => {
      const hidden = t.embedding.map(x => Math.max(0, x * 1.2 + 0.1));
      return hidden.map(x => x * 0.9);
    });

    return { tokens: input, attention: attentionHeads, feedForward: ffOutput };
  };

  const encoderLayers: LayerOutput[] = [];
  let currentData = tokenData;
  
  for (let i = 0; i < numLayers; i++) {
    const layer = processLayer(currentData, i);
    encoderLayers.push(layer);
    currentData = layer.tokens.map((t, idx) => ({
      ...t,
      embedding: layer.feedForward[idx]
    }));
  }

  const decoderLayers: LayerOutput[] = [];
  for (let i = 0; i < numLayers; i++) {
    const layer = processLayer(currentData, i);
    decoderLayers.push(layer);
    currentData = layer.tokens.map((t, idx) => ({
      ...t,
      embedding: layer.feedForward[idx]
    }));
  }

  return {
    input: text,
    tokens,
    encoder: encoderLayers,
    decoder: decoderLayers,
    output: tokens.join(' ')
  };
};
