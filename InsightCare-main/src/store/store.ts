import { configureStore } from '@reduxjs/toolkit';
import authReducer from './slices/authSlice';
import symptomReducer from './slices/symptomSlice';

export const store = configureStore({
  reducer: {
    auth: authReducer,
    symptom: symptomReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
