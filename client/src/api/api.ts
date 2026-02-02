import axios from 'axios';
import { VisualizationResponse } from '../types';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const visualizeTransformer = async (
  text: string,
  numLayers: number = 6,
  numHeads: number = 8,
  embeddingDim: number = 512
): Promise<VisualizationResponse> => {
  const { data } = await axios.post<VisualizationResponse>(`${API_URL}/visualize`, {
    text,
    numLayers,
    numHeads,
    embeddingDim
  });
  return data;
};
