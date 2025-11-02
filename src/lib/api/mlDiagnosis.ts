/**
 * ML Diagnosis API Client
 * Connects to backend ML endpoints for AI-powered disease diagnosis
 */

import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface MLPrediction {
  disease: string;
  confidence: number;
  severity: string;
  description: string;
  recommendations: string[];
  model_used: string;
  valid_symptoms: string[];
  invalid_symptoms: string[];
}

export interface MLDiagnosisResponse {
  success: boolean;
  predictions: MLPrediction[];
  total_predictions: number;
  ml_available: boolean;
  message: string;
}

export interface MLHealthResponse {
  ml_available: boolean;
  total_symptoms: number;
  total_diseases: number;
  models_loaded: string[];
}

/**
 * Get AI-powered disease diagnosis from symptoms
 */
export const getMlDiagnosis = async (
  symptoms: string[]
): Promise<MLDiagnosisResponse> => {
  const response = await axios.post<MLDiagnosisResponse>(
    `${API_URL}/api/ml/diagnose`,
    { symptoms }
  );
  return response.data;
};

/**
 * Check ML service health
 */
export const checkMlHealth = async (): Promise<MLHealthResponse> => {
  const response = await axios.get<MLHealthResponse>(
    `${API_URL}/api/ml/health`
  );
  return response.data;
};
