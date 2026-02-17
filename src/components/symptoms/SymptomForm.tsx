'use client';

import {
  Box,
  TextField,
  Button,
  IconButton,
  Grid,
  MenuItem,
  Typography,
  Paper,
  Tooltip,
} from '@mui/material';
import { useForm, useFieldArray, Controller } from 'react-hook-form';
import { motion, AnimatePresence } from 'framer-motion';
import DeleteIcon from '@mui/icons-material/Delete';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import SendIcon from '@mui/icons-material/Send';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

const MotionPaper = motion(Paper);

export interface SymptomFormData {
  symptoms: Array<{
    name: string;
    severity: string;
    duration: string;
  }>;
}

interface SymptomFormProps {
  onSubmit: (data: SymptomFormData) => void;
  isLoading?: boolean;
}

const severityOptions = [
  { value: 'mild', label: 'Mild' },
  { value: 'moderate', label: 'Moderate' },
  { value: 'severe', label: 'Severe' },
];

const durationOptions = [
  { value: 'less_than_24h', label: 'Less than 24 hours' },
  { value: '1_3_days', label: '1-3 days' },
  { value: '4_7_days', label: '4-7 days' },
  { value: '1_2_weeks', label: '1-2 weeks' },
  { value: 'more_than_2_weeks', label: 'More than 2 weeks' },
];

export default function SymptomForm({ onSubmit, isLoading = false }: SymptomFormProps) {
  const {
    control,
    handleSubmit,
    formState: { errors },
  } = useForm<SymptomFormData>({
    defaultValues: {
      symptoms: [{ name: '', severity: '', duration: '' }],
    },
  });

  const { fields, append, remove } = useFieldArray({
    control,
    name: 'symptoms',
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="h5" component="h2" fontWeight={600}>
            Enter Your Symptoms
          </Typography>
          <Tooltip title="Provide detailed information about each symptom for accurate diagnosis">
            <HelpOutlineIcon color="action" fontSize="small" />
          </Tooltip>
        </Box>

        <AnimatePresence>
          {fields.map((field, index) => (
            <MotionPaper
              key={field.id}
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -100 }}
              transition={{ duration: 0.3 }}
              sx={{
                p: 3,
                background: 'linear-gradient(135deg, rgba(25, 118, 210, 0.02) 0%, rgba(255,255,255,1) 100%)',
                border: '1px solid',
                borderColor: 'divider',
                '&:hover': {
                  boxShadow: 3,
                  borderColor: 'primary.main',
                },
              }}
            >
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Symptom #{index + 1}
                  </Typography>
                </Grid>
                
                <Grid item xs={12} sm={4}>
                  <Controller
                    name={`symptoms.${index}.name`}
                    control={control}
                    rules={{ 
                      required: 'Symptom name is required',
                      minLength: {
                        value: 2,
                        message: 'Symptom name must be at least 2 characters'
                      },
                      maxLength: {
                        value: 100,
                        message: 'Symptom name must not exceed 100 characters'
                      },
                      pattern: {
                        value: /^[a-zA-Z\s\-']+$/,
                        message: 'Symptom name can only contain letters, spaces, hyphens, and apostrophes'
                      }
                    }}
                    render={({ field }) => (
                      <TextField
                        {...field}
                        fullWidth
                        label="Symptom Name"
                        placeholder="e.g., Headache, Fever, Cough"
                        error={!!errors.symptoms?.[index]?.name}
                        helperText={errors.symptoms?.[index]?.name?.message}
                        variant="outlined"
                      />
                    )}
                  />
                </Grid>

                <Grid item xs={12} sm={3}>
                  <Controller
                    name={`symptoms.${index}.severity`}
                    control={control}
                    rules={{ required: 'Severity is required' }}
                    render={({ field }) => (
                      <TextField
                        {...field}
                        select
                        fullWidth
                        label="Severity"
                        error={!!errors.symptoms?.[index]?.severity}
                        helperText={errors.symptoms?.[index]?.severity?.message}
                        variant="outlined"
                      >
                        {severityOptions.map((option) => (
                          <MenuItem key={option.value} value={option.value}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </TextField>
                    )}
                  />
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Controller
                    name={`symptoms.${index}.duration`}
                    control={control}
                    rules={{ required: 'Duration is required' }}
                    render={({ field }) => (
                      <TextField
                        {...field}
                        select
                        fullWidth
                        label="Duration"
                        error={!!errors.symptoms?.[index]?.duration}
                        helperText={errors.symptoms?.[index]?.duration?.message}
                        variant="outlined"
                      >
                        {durationOptions.map((option) => (
                          <MenuItem key={option.value} value={option.value}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </TextField>
                    )}
                  />
                </Grid>

                <Grid item xs={12} sm={1} sx={{ display: 'flex', alignItems: 'center' }}>
                  <Tooltip title="Remove symptom">
                    <span>
                      <IconButton
                        color="error"
                        onClick={() => remove(index)}
                        disabled={fields.length === 1}
                        aria-label="Remove symptom"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Grid>
              </Grid>
            </MotionPaper>
          ))}
        </AnimatePresence>

        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Button
            startIcon={<AddCircleIcon />}
            onClick={() => append({ name: '', severity: '', duration: '' })}
            variant="outlined"
            size="large"
            sx={{ textTransform: 'none', fontWeight: 600 }}
          >
            Add Another Symptom
          </Button>

          <Button
            type="submit"
            variant="contained"
            size="large"
            endIcon={<SendIcon />}
            disabled={isLoading}
            sx={{
              ml: 'auto',
              px: 4,
              textTransform: 'none',
              fontWeight: 600,
              background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #1565c0 0%, #1976d2 100%)',
              },
            }}
          >
            {isLoading ? 'Analyzing...' : 'Analyze Symptoms'}
          </Button>
        </Box>

        <Paper sx={{ p: 2, bgcolor: 'info.lighter', border: '1px solid', borderColor: 'info.light' }}>
          <Typography variant="body2" color="text.secondary">
            <strong>Note:</strong> The more detailed your symptom description, the more accurate the diagnosis. Include all relevant symptoms you&apos;re experiencing.
          </Typography>
        </Paper>
      </Box>
    </form>
  );
}
