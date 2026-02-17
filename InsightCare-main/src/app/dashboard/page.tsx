'use client';

import { useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Container, Typography, Box, Button, Alert, Paper, Grid, Card, CardContent } from '@mui/material';
import { motion } from 'framer-motion';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import DiagnosisDisplay from '@/components/dashboard/DiagnosisDisplay';
import { DiagnosisResult } from '@/lib/api/symptomsApi';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import TimelineIcon from '@mui/icons-material/Timeline';
import AssessmentIcon from '@mui/icons-material/Assessment';

const MotionBox = motion(Box);
const MotionCard = motion(Card);

export default function DashboardPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [showMockData, setShowMockData] = useState(true); // Set to true to show example data
  const [diagnosisResults, setDiagnosisResults] = useState<DiagnosisResult[]>([]);

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login');
    }
  }, [status, router]);

  // Mock data for demonstration (replace with actual API call)
  useEffect(() => {
    if (showMockData) {
      const mockResults: DiagnosisResult[] = [
        {
          id: '1',
          disease: 'Common Cold',
          probability: 65,
          description: 'A viral infection of your nose and throat (upper respiratory tract). Usually harmless and symptoms typically resolve within 7-10 days.',
          recommendations: [
            'Get plenty of rest',
            'Stay hydrated with water and warm fluids',
            'Use over-the-counter cold medications if needed',
            'Consider seeing a doctor if symptoms persist beyond 10 days',
          ],
          severity: 'low',
        },
        {
          id: '2',
          disease: 'Seasonal Allergies',
          probability: 25,
          description: 'An immune system response to foreign substances that are typically not harmful to your body.',
          recommendations: [
            'Avoid known allergens when possible',
            'Take antihistamine medications',
            'Consider allergy testing',
            'Use air purifiers indoors',
          ],
          severity: 'low',
        },
        {
          id: '3',
          disease: 'Flu (Influenza)',
          probability: 10,
          description: 'A contagious respiratory illness caused by influenza viruses that infect the nose, throat, and sometimes the lungs.',
          recommendations: [
            'Rest and stay home',
            'Drink plenty of fluids',
            'Take antiviral medications if prescribed',
            'Seek medical attention if symptoms worsen',
          ],
          severity: 'medium',
        },
      ];

      setDiagnosisResults(mockResults);
    }
  }, [showMockData]);

  if (status === 'loading') {
    return (
      <Container>
        <LoadingSpinner message="Loading..." />
      </Container>
    );
  }

  if (status === 'unauthenticated' || !session) {
    return (
      <Container>
        <Box sx={{ py: 8, textAlign: 'center' }}>
          <Typography variant="h5" gutterBottom>
            Redirecting to login...
          </Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        {/* Welcome Section */}
        <MotionBox
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Typography variant="h3" component="h1" gutterBottom fontWeight={700}>
            Welcome back, {session.user?.name || 'User'}! 👋
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Your AI-powered health dashboard
          </Typography>
        </MotionBox>

        {/* Quick Actions */}
        <Grid container spacing={3} sx={{ mb: 4, mt: 2 }}>
          <Grid item xs={12} md={4}>
            <MotionCard
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              sx={{ height: '100%', cursor: 'pointer', '&:hover': { boxShadow: 6 } }}
              onClick={() => router.push('/symptoms')}
            >
              <CardContent sx={{ textAlign: 'center', py: 4 }}>
                <AddCircleOutlineIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom fontWeight={600}>
                  New Diagnosis
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Start a new symptom analysis
                </Typography>
              </CardContent>
            </MotionCard>
          </Grid>

          <Grid item xs={12} md={4}>
            <MotionCard
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              sx={{ height: '100%' }}
            >
              <CardContent sx={{ textAlign: 'center', py: 4 }}>
                <TimelineIcon sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom fontWeight={600}>
                  Health History
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  View past diagnoses
                </Typography>
              </CardContent>
            </MotionCard>
          </Grid>

          <Grid item xs={12} md={4}>
            <MotionCard
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              sx={{ height: '100%' }}
            >
              <CardContent sx={{ textAlign: 'center', py: 4 }}>
                <AssessmentIcon sx={{ fontSize: 60, color: 'info.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom fontWeight={600}>
                  Health Insights
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  AI-powered recommendations
                </Typography>
              </CardContent>
            </MotionCard>
          </Grid>
        </Grid>

        {/* Sample Diagnosis Section */}
        {diagnosisResults.length > 0 && (
          <Box sx={{ mt: 4 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h4" fontWeight={600}>
                Sample Diagnosis Report
              </Typography>
              <Button
                variant="contained"
                startIcon={<AddCircleOutlineIcon />}
                onClick={() => router.push('/symptoms')}
              >
                Start New Diagnosis
              </Button>
            </Box>

            <Alert severity="info" sx={{ mb: 3 }}>
              <strong>Demo Mode:</strong> This is a sample diagnosis report to showcase the dashboard features.
              Click &quot;Start New Diagnosis&quot; to analyze your own symptoms.
            </Alert>

            <DiagnosisDisplay results={diagnosisResults} />
          </Box>
        )}
      </Box>
    </Container>
  );
}
