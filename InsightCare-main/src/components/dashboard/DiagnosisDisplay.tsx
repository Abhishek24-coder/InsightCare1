'use client';

import { Grid, Typography, Box, Chip, Button, Stack, Divider, LinearProgress } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip as RechartsTooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { motion } from 'framer-motion';
import InfoCard from '../common/InfoCard';
import { DiagnosisResult } from '@/lib/api/symptomsApi';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import DownloadIcon from '@mui/icons-material/Download';
import PrintIcon from '@mui/icons-material/Print';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';

const MotionBox = motion(Box);
const MotionGrid = motion(Grid);

interface DiagnosisDisplayProps {
  results: DiagnosisResult[];
}

const COLORS = ['#1976d2', '#42a5f5', '#64b5f6', '#90caf9', '#bbdefb'];
const SEVERITY_COLORS = {
  low: '#4caf50',
  medium: '#ff9800',
  high: '#f44336',
};

export default function DiagnosisDisplay({ results }: DiagnosisDisplayProps) {
  const chartData = results.map((result, index) => ({
    name: result.disease,
    value: result.probability,
    fill: COLORS[index % COLORS.length],
  }));

  const barData = results.map((result) => ({
    disease: result.disease.length > 15 ? result.disease.substring(0, 15) + '...' : result.disease,
    probability: result.probability,
    fill: COLORS[results.indexOf(result) % COLORS.length],
  }));

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'low':
        return <CheckCircleIcon sx={{ color: SEVERITY_COLORS.low }} />;
      case 'medium':
        return <WarningIcon sx={{ color: SEVERITY_COLORS.medium }} />;
      case 'high':
        return <ErrorIcon sx={{ color: SEVERITY_COLORS.high }} />;
      default:
        return null;
    }
  };

  const getSeverityColor = (severity: string) => {
    return SEVERITY_COLORS[severity as keyof typeof SEVERITY_COLORS] || '#9e9e9e';
  };

  const handlePrint = () => {
    window.print();
  };

  const handleDownload = () => {
    // Placeholder for download functionality
    alert('Download functionality would export this report as PDF');
  };

  return (
    <Box>
      {/* Header with Actions */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4, flexWrap: 'wrap', gap: 2 }}>
        <Box>
          <Typography variant="h4" gutterBottom fontWeight={700}>
            Diagnosis Results
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Powered by Quantum AI Analysis
          </Typography>
        </Box>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            startIcon={<PrintIcon />}
            onClick={handlePrint}
            sx={{ textTransform: 'none' }}
          >
            Print Report
          </Button>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleDownload}
            sx={{ textTransform: 'none' }}
          >
            Download PDF
          </Button>
        </Stack>
      </Box>

      <Grid container spacing={3}>
        {/* Primary Diagnosis Card */}
        <Grid item xs={12}>
          <MotionBox
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <InfoCard
              sx={{
                background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
                color: 'white',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                <LocalHospitalIcon sx={{ fontSize: 60, opacity: 0.9 }} />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="overline" sx={{ opacity: 0.9 }}>
                    Primary Diagnosis
                  </Typography>
                  <Typography variant="h3" fontWeight={700} gutterBottom>
                    {results[0]?.disease}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
                    <Typography variant="h4" fontWeight={600}>
                      {results[0]?.probability.toFixed(1)}%
                    </Typography>
                    <Chip
                      label={`${results[0]?.severity.toUpperCase()} SEVERITY`}
                      sx={{
                        bgcolor: 'rgba(255,255,255,0.2)',
                        color: 'white',
                        fontWeight: 600,
                      }}
                    />
                  </Box>
                  <Typography variant="body1" sx={{ opacity: 0.95, mb: 2 }}>
                    {results[0]?.description}
                  </Typography>
                  <Divider sx={{ bgcolor: 'rgba(255,255,255,0.2)', my: 2 }} />
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    RECOMMENDED ACTIONS:
                  </Typography>
                  <Stack spacing={1}>
                    {results[0]?.recommendations.map((rec, idx) => (
                      <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
                        <CheckCircleIcon sx={{ fontSize: 20, mt: 0.3 }} />
                        <Typography variant="body2">{rec}</Typography>
                      </Box>
                    ))}
                  </Stack>
                </Box>
              </Box>
            </InfoCard>
          </MotionBox>
        </Grid>

        {/* Probability Distribution Chart */}
        <MotionGrid
          item
          xs={12}
          md={6}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <InfoCard title="Probability Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <RechartsTooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </InfoCard>
        </MotionGrid>

        {/* Confidence Levels Bar Chart */}
        <MotionGrid
          item
          xs={12}
          md={6}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <InfoCard title="Confidence Levels">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={barData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="disease" type="category" width={100} />
                <RechartsTooltip />
                <Bar dataKey="probability" radius={[0, 8, 8, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </InfoCard>
        </MotionGrid>

        {/* All Diagnoses */}
        <Grid item xs={12}>
          <Typography variant="h5" gutterBottom fontWeight={600} sx={{ mt: 2 }}>
            Differential Diagnoses
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            All possible diagnoses ranked by probability
          </Typography>
        </Grid>

        {results.map((result, index) => (
          <MotionGrid
            item
            xs={12}
            md={6}
            key={result.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 + index * 0.1 }}
          >
            <InfoCard
              sx={{
                height: '100%',
                border: '2px solid',
                borderColor: index === 0 ? 'primary.main' : 'divider',
                position: 'relative',
                overflow: 'visible',
              }}
            >
              {index === 0 && (
                <Chip
                  label="Most Likely"
                  color="primary"
                  size="small"
                  sx={{
                    position: 'absolute',
                    top: -12,
                    right: 16,
                    fontWeight: 600,
                  }}
                />
              )}
              <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2, mb: 2 }}>
                {getSeverityIcon(result.severity)}
                <Box sx={{ flex: 1 }}>
                  <Typography variant="h6" fontWeight={600} gutterBottom>
                    {result.disease}
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2" color="text.secondary">
                        Confidence Score
                      </Typography>
                      <Typography variant="body2" fontWeight={600}>
                        {result.probability.toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={result.probability}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        bgcolor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 4,
                          bgcolor: getSeverityColor(result.severity),
                        },
                      }}
                    />
                  </Box>
                  <Chip
                    label={`Severity: ${result.severity.toUpperCase()}`}
                    size="small"
                    sx={{
                      bgcolor: getSeverityColor(result.severity),
                      color: 'white',
                      fontWeight: 600,
                      mb: 2,
                    }}
                  />
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                {result.description}
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                Recommendations:
              </Typography>
              <Stack spacing={0.5}>
                {result.recommendations.map((rec, idx) => (
                  <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
                    <CheckCircleIcon sx={{ fontSize: 16, color: 'success.main', mt: 0.3 }} />
                    <Typography variant="body2">{rec}</Typography>
                  </Box>
                ))}
              </Stack>
            </InfoCard>
          </MotionGrid>
        ))}

        {/* Disclaimer */}
        <Grid item xs={12}>
          <Box
            sx={{
              p: 3,
              bgcolor: 'warning.lighter',
              border: '1px solid',
              borderColor: 'warning.light',
              borderRadius: 1,
            }}
          >
            <Stack direction="row" spacing={2} alignItems="flex-start">
              <WarningIcon color="warning" />
              <Box>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  Important Medical Disclaimer
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  This AI-generated diagnosis is for informational purposes only and should not replace professional medical advice. 
                  Please consult with a qualified healthcare provider for proper diagnosis and treatment. In case of emergency, 
                  contact your local emergency services immediately.
                </Typography>
              </Box>
            </Stack>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}
