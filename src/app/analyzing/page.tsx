'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Container, Box, Typography, LinearProgress, Stack, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import PsychologyIcon from '@mui/icons-material/Psychology';
import BiotechIcon from '@mui/icons-material/Biotech';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const MotionBox = motion(Box);

export default function AnalyzingPage() {
  const router = useRouter();
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { icon: <BiotechIcon />, label: 'Processing Symptoms', delay: 1000 },
    { icon: <PsychologyIcon />, label: 'Quantum AI Analysis', delay: 2000 },
    { icon: <AnalyticsIcon />, label: 'Generating Diagnosis', delay: 3000 },
  ];

  useEffect(() => {
    // Simulate progress
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer);
          setTimeout(() => router.push('/dashboard'), 500);
          return 100;
        }
        return prev + 1;
      });
    }, 40);

    // Update steps
    const stepTimer1 = setTimeout(() => setCurrentStep(1), 1000);
    const stepTimer2 = setTimeout(() => setCurrentStep(2), 2500);
    const stepTimer3 = setTimeout(() => setCurrentStep(3), 3500);

    return () => {
      clearInterval(timer);
      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      clearTimeout(stepTimer3);
    };
  }, [router]);

  return (
    <Container maxWidth="md">
      <Box
        sx={{
          minHeight: '70vh',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        {/* Animated Icon */}
        <MotionBox
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 360],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          sx={{ mb: 4 }}
        >
          <PsychologyIcon
            sx={{
              fontSize: 120,
              color: 'primary.main',
              filter: 'drop-shadow(0 0 20px rgba(25, 118, 210, 0.5))',
            }}
          />
        </MotionBox>

        {/* Title */}
        <Typography variant="h3" gutterBottom fontWeight={700}>
          Analyzing Your Symptoms
        </Typography>
        <Typography variant="h6" color="text.secondary" gutterBottom sx={{ mb: 4 }}>
          Our quantum-enhanced AI is processing your medical data
        </Typography>

        {/* Progress Bar */}
        <Box sx={{ width: '100%', mb: 4 }}>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{
              height: 8,
              borderRadius: 4,
              bgcolor: 'grey.200',
              '& .MuiLinearProgress-bar': {
                borderRadius: 4,
                background: 'linear-gradient(90deg, #1976d2 0%, #42a5f5 100%)',
              },
            }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {progress}% Complete
          </Typography>
        </Box>

        {/* Steps */}
        <Stack spacing={2} sx={{ width: '100%', maxWidth: 500 }}>
          {steps.map((step, index) => (
            <MotionBox
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: currentStep >= index ? 1 : 0.3, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Chip
                icon={currentStep > index ? <CheckCircleIcon /> : step.icon}
                label={step.label}
                color={currentStep > index ? 'success' : currentStep === index ? 'primary' : 'default'}
                sx={{
                  width: '100%',
                  justifyContent: 'flex-start',
                  py: 3,
                  fontSize: '1rem',
                  fontWeight: currentStep === index ? 600 : 400,
                }}
              />
            </MotionBox>
          ))}
        </Stack>

        {/* Info Text */}
        <Typography variant="body2" color="text.secondary" sx={{ mt: 4, maxWidth: 600 }}>
          Please wait while our advanced algorithms analyze your symptoms using quantum computing technology. This usually takes a few seconds.
        </Typography>
      </Box>
    </Container>
  );
}
