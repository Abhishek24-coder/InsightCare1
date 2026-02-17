import apiClient from './apiClient';
import { Symptom } from '@/store/slices/symptomSlice';

export interface DiagnosisResult {
  id: string;
  disease: string;
  probability: number;
  description: string;
  recommendations: string[];
  severity: 'low' | 'medium' | 'high';
}

export interface SymptomSubmission {
  symptoms: Symptom[];
}

export const symptomsApi = {
  submitSymptoms: async (data: SymptomSubmission): Promise<DiagnosisResult[]> => {
    const response = await apiClient.post<DiagnosisResult[]>('/diagnose', data);
    return response.data;
  },

  getHistory: async (): Promise<DiagnosisResult[]> => {
    const response = await apiClient.get<DiagnosisResult[]>('/history');
    return response.data;
  },
};
