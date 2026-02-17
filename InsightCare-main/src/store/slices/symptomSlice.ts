import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface Symptom {
  id: string;
  name: string;
  severity: string;
  duration: string;
}

interface SymptomState {
  symptoms: Symptom[];
  submittedData: Symptom[] | null;
}

const initialState: SymptomState = {
  symptoms: [],
  submittedData: null,
};

const symptomSlice = createSlice({
  name: 'symptom',
  initialState,
  reducers: {
    addSymptom: (state: SymptomState, action: PayloadAction<Symptom>) => {
      state.symptoms.push(action.payload);
    },
    removeSymptom: (state: SymptomState, action: PayloadAction<string>) => {
      state.symptoms = state.symptoms.filter((s: Symptom) => s.id !== action.payload);
    },
    updateSymptom: (state: SymptomState, action: PayloadAction<Symptom>) => {
      const index = state.symptoms.findIndex((s: Symptom) => s.id === action.payload.id);
      if (index !== -1) {
        state.symptoms[index] = action.payload;
      }
    },
    setSymptoms: (state: SymptomState, action: PayloadAction<Symptom[]>) => {
      state.symptoms = action.payload;
    },
    submitSymptoms: (state: SymptomState) => {
      state.submittedData = state.symptoms;
    },
    clearSymptoms: (state: SymptomState) => {
      state.symptoms = [];
      state.submittedData = null;
    },
  },
});

export const {
  addSymptom,
  removeSymptom,
  updateSymptom,
  setSymptoms,
  submitSymptoms,
  clearSymptoms,
} = symptomSlice.actions;
export default symptomSlice.reducer;
