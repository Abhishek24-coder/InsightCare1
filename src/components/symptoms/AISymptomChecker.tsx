'use client';

import { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Button,
  Chip,
  Grid,
  Alert,
  CircularProgress,
  LinearProgress,
  Divider,
  Card,
  CardContent,
  TextField,
  InputAdornment,
  Checkbox,
  FormControlLabel,
  FormGroup,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PsychologyIcon from '@mui/icons-material/Psychology';
import { getMlDiagnosis, checkMlHealth, MLDiagnosisResponse } from '@/lib/api/mlDiagnosis';

// Common symptoms for quick selection
const COMMON_SYMPTOMS = [
  'fever', 'cough', 'fatigue', 'headache', 'body_ache', 'chills',
  'sore_throat', 'runny_nose', 'shortness_of_breath', 'nausea',
  'vomiting', 'diarrhea', 'abdominal_pain', 'chest_pain', 'dizziness',
  'weakness', 'loss_of_appetite', 'weight_loss', 'skin_rash', 'itching',
  'joint_pain', 'muscle_pain', 'back_pain', 'sweating', 'sneezing',
  'continuous_sneezing', 'shivering', 'stomach_pain', 'acidity',
  'high_fever', 'restlessness', 'lethargy', 'patches_in_throat',
  'irregular_sugar_level', 'cough', 'cold', 'breathlessness',
];

export default function AISymptomChecker() {
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [diagnosisResult, setDiagnosisResult] = useState<MLDiagnosisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mlAvailable, setMlAvailable] = useState(true);

  // Check ML health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await checkMlHealth();
        setMlAvailable(health.ml_available);
      } catch (err) {
        console.error('Failed to check ML health:', err);
        setMlAvailable(false);
      }
    };
    checkHealth();
  }, []);

  const handleToggleSymptom = (symptom: string) => {
    setSelectedSymptoms((prev) =>
      prev.includes(symptom)
        ? prev.filter((s) => s !== symptom)
        : [...prev, symptom]
    );
  };

  const handleAnalyze = async () => {
    if (selectedSymptoms.length === 0) {
      setError('Please select at least one symptom');
      return;
    }

    setLoading(true);
    setError(null);
    setDiagnosisResult(null);

    try {
      const result = await getMlDiagnosis(selectedSymptoms);
      setDiagnosisResult(result);
      setMlAvailable(result.ml_available);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze symptoms. Please try again.');
      console.error('Diagnosis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedSymptoms([]);
    setDiagnosisResult(null);
    setError(null);
    setSearchTerm('');
  };

  const filteredSymptoms = COMMON_SYMPTOMS.filter((symptom) =>
    symptom.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Moderate Confidence';
    return 'Low Confidence';
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high': return 'error';
      case 'moderate': return 'warning';
      case 'low': return 'success';
      default: return 'info';
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <PsychologyIcon sx={{ fontSize: 48, color: 'primary.main' }} />
          <Box>
            <Typography variant="h3" component="h1" fontWeight={700}>
              AI Symptom Checker
            </Typography>
          <Typography variant="body2" color="text.secondary">
            Powered by Hybrid Quantum-Classical AI: RF + XGBoost + QSVM
          </Typography>
          </Box>
        </Box>
        <Typography variant="body1" color="text.secondary">
          Select your symptoms below and get instant AI-powered disease predictions {mlAvailable ? '✅ Hybrid ensemble active (70% Classical + 30% Quantum)' : '⚠️ ML models unavailable'}
        </Typography>
      </Box>

      {!mlAvailable && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          ML models are currently unavailable. Using fallback diagnosis system.
        </Alert>
      )}

      <Grid container spacing={4}>
        {/* Left Panel - Symptom Selection */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
            <Typography variant="h5" gutterBottom fontWeight={600}>
              Select Your Symptoms
            </Typography>

            <TextField
              fullWidth
              placeholder="Search symptoms..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
              sx={{ mb: 3 }}
            />

            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Selected: {selectedSymptoms.length} symptoms
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1, minHeight: 40 }}>
                {selectedSymptoms.map((symptom) => (
                  <Chip
                    key={symptom}
                    label={symptom.replace(/_/g, ' ')}
                    onDelete={() => handleToggleSymptom(symptom)}
                    color="primary"
                    variant="filled"
                  />
                ))}
                {selectedSymptoms.length === 0 && (
                  <Typography variant="caption" color="text.secondary">
                    No symptoms selected yet
                  </Typography>
                )}
              </Box>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Box sx={{ maxHeight: 400, overflow: 'auto', pr: 1 }}>
              <FormGroup>
                {filteredSymptoms.map((symptom) => (
                  <FormControlLabel
                    key={symptom}
                    control={
                      <Checkbox
                        checked={selectedSymptoms.includes(symptom)}
                        onChange={() => handleToggleSymptom(symptom)}
                      />
                    }
                    label={symptom.replace(/_/g, ' ')}
                  />
                ))}
              </FormGroup>
            </Box>

            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                size="large"
                fullWidth
                onClick={handleAnalyze}
                disabled={loading || selectedSymptoms.length === 0}
                startIcon={loading ? <CircularProgress size={20} /> : <LocalHospitalIcon />}
              >
                {loading ? 'Analyzing...' : 'Analyze Symptoms'}
              </Button>
              <Button variant="outlined" onClick={handleReset}>
                Reset
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Right Panel - Results */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3, minHeight: 600 }}>
            <Typography variant="h5" gutterBottom fontWeight={600}>
              AI Diagnosis Results
            </Typography>

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            {!diagnosisResult && !loading && !error && (
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: 400,
                  color: 'text.secondary',
                }}
              >
                <LocalHospitalIcon sx={{ fontSize: 80, mb: 2, opacity: 0.3 }} />
                <Typography variant="body1" textAlign="center">
                  Select symptoms and click &quot;Analyze&quot; to see AI predictions
                </Typography>
              </Box>
            )}

            {loading && (
              <Box sx={{ textAlign: 'center', py: 8 }}>
                <CircularProgress size={60} />
                <Typography variant="body1" sx={{ mt: 2 }}>
                  AI models analyzing your symptoms...
                </Typography>
              </Box>
            )}

            {diagnosisResult && diagnosisResult.success && diagnosisResult.predictions.length > 0 && (
              <Box>
                {/* Primary Prediction */}
                {diagnosisResult.predictions.map((prediction, index) => (
                  <Card
                    key={index}
                    sx={{
                      mb: 3,
                      background: index === 0 
                        ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                        : 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                      color: 'white',
                    }}
                  >
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <CheckCircleIcon sx={{ mr: 1 }} />
                        <Typography variant="overline">
                          {index === 0 ? 'Primary' : 'Alternative'} Prediction ({prediction.model_used === 'hybrid_ensemble' ? 'Quantum-Classical Hybrid' : prediction.model_used})
                        </Typography>
                      </Box>
                      <Typography variant="h4" fontWeight={700} gutterBottom>
                        {prediction.disease}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                        <Chip
                          label={getConfidenceLabel(prediction.confidence)}
                          color={getConfidenceColor(prediction.confidence)}
                          size="small"
                        />
                        <Chip
                          label={`Severity: ${prediction.severity}`}
                          color={getSeverityColor(prediction.severity)}
                          size="small"
                        />
                        <Typography variant="h6">
                          {(prediction.confidence * 100).toFixed(1)}% Confidence
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={prediction.confidence * 100}
                        sx={{
                          mt: 2,
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(255,255,255,0.3)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: 'white',
                          },
                        }}
                      />

                      {/* Description */}
                      <Accordion sx={{ mt: 2, backgroundColor: 'rgba(255,255,255,0.1)' }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: 'white' }} />}>
                          <Typography sx={{ color: 'white' }}>About this condition</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Typography sx={{ color: 'white' }}>{prediction.description}</Typography>
                        </AccordionDetails>
                      </Accordion>

                      {/* Recommendations */}
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          💡 Recommendations:
                        </Typography>
                        <Box component="ul" sx={{ pl: 2, m: 0 }}>
                          {prediction.recommendations.map((rec, i) => (
                            <Typography component="li" key={i} variant="body2">
                              {rec}
                            </Typography>
                          ))}
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                ))}

                {/* Symptoms Summary */}
                <Box sx={{ mt: 3 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    ✅ Valid Symptoms Used:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                    {diagnosisResult.predictions[0].valid_symptoms.map((symptom) => (
                      <Chip
                        key={symptom}
                        label={symptom.replace(/_/g, ' ')}
                        size="small"
                        color="success"
                        variant="outlined"
                      />
                    ))}
                  </Box>

                  {diagnosisResult.predictions[0].invalid_symptoms.length > 0 && (
                    <>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        ⚠️ Invalid/Unknown Symptoms:
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {diagnosisResult.predictions[0].invalid_symptoms.map((symptom) => (
                          <Chip
                            key={symptom}
                            label={symptom.replace(/_/g, ' ')}
                            size="small"
                            color="error"
                            variant="outlined"
                          />
                        ))}
                      </Box>
                    </>
                  )}
                </Box>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Disclaimer */}
      <Alert severity="warning" icon={<WarningAmberIcon />} sx={{ mt: 4 }}>
        <Typography variant="body2" fontWeight={600} gutterBottom>
          ⚠️ Important Medical Disclaimer
        </Typography>
        <Typography variant="body2">
          This AI diagnosis tool is for informational purposes only and should not replace
          professional medical advice. The predictions are based on machine learning models trained
          on medical data but are not 100% accurate. Always consult with a qualified healthcare
          provider for proper diagnosis and treatment. If you have severe symptoms or a medical
          emergency, seek immediate medical attention.
        </Typography>
      </Alert>
    </Container>
  );
}
