export interface TokenData {
  token: string;
  embedding: number[];
  position: number;
}

export interface AttentionHead {
  query: number[];
  key: number[];
  value: number[];
  scores: number[][];
  weights: number[][];
}

export interface LayerOutput {
  tokens: TokenData[];
  attention: AttentionHead[];
  feedForward: number[][];
}

export interface TransformerState {
  input: string;
  tokens: string[];
  encoder: LayerOutput[];
  decoder: LayerOutput[];
  output: string;
}

export interface VisualizationRequest {
  text: string;
  numLayers?: number;
  numHeads?: number;
  embeddingDim?: number;
}

export interface VisualizationResponse {
  state: TransformerState;
  metadata: {
    numLayers: number;
    numHeads: number;
    embeddingDim: number;
    vocabSize: number;
  };
}
