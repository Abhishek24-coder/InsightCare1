'use client';

import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import {
  Container,
  Typography,
  Box,
  Button,
  Grid,
  Card,
  CardContent,
  Chip,
  Stack,
  Avatar,
  useTheme,
  alpha,
} from '@mui/material';
import { motion, useScroll, useTransform } from 'framer-motion';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import SecurityIcon from '@mui/icons-material/Security';
import SpeedIcon from '@mui/icons-material/Speed';
import PsychologyIcon from '@mui/icons-material/Psychology';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import BoltIcon from '@mui/icons-material/Bolt';
import LockIcon from '@mui/icons-material/Lock';
import VerifiedIcon from '@mui/icons-material/Verified';

const MotionBox = motion(Box);
const MotionCard = motion(Card);

export default function HomePage() {
  const router = useRouter();
  const { data: session } = useSession();

  const features = [
    {
      icon: <PsychologyIcon sx={{ fontSize: 50, color: 'primary.main' }} />,
      title: 'Quantum-Enhanced AI',
      description: 'Advanced quantum computing algorithms detect complex medical patterns that traditional systems miss.',
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 50, color: 'secondary.main' }} />,
      title: 'Rapid Diagnosis',
      description: 'Get accurate disease predictions in seconds with our cutting-edge hybrid AI technology.',
    },
    {
      icon: <TrendingUpIcon sx={{ fontSize: 50, color: 'success.main' }} />,
      title: 'Personalized Insights',
      description: 'Receive tailored health recommendations based on your unique symptom profile and medical history.',
    },
    {
      icon: <SecurityIcon sx={{ fontSize: 50, color: 'info.main' }} />,
      title: 'HIPAA & GDPR Compliant',
      description: 'Your health data is encrypted and secured with industry-leading privacy standards.',
    },
  ];

  const steps = [
    { number: '01', title: 'Describe Symptoms', description: 'Enter your health concerns through our intuitive interface' },
    { number: '02', title: 'AI Analysis', description: 'Quantum AI processes your data with unprecedented accuracy' },
    { number: '03', title: 'Get Results', description: 'Receive detailed diagnosis with confidence scores and recommendations' },
  ];

  const theme = useTheme();

  return (
    <Box sx={{ overflow: 'hidden' }}>
      {/* Modern Hero Section with Glassmorphism */}
      <Box
        sx={{
          position: 'relative',
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          background: `linear-gradient(135deg, 
            ${alpha(theme.palette.primary.main, 0.05)} 0%, 
            ${alpha(theme.palette.secondary.main, 0.05)} 100%)`,
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `radial-gradient(circle at 20% 30%, ${alpha(theme.palette.primary.main, 0.15)} 0%, transparent 50%),
                        radial-gradient(circle at 80% 70%, ${alpha(theme.palette.secondary.main, 0.15)} 0%, transparent 50%)`,
            pointerEvents: 'none',
          },
        }}
      >
        {/* Floating Background Elements */}
        <MotionBox
          animate={{
            y: [0, -30, 0],
            rotate: [0, 5, 0],
          }}
          transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
          sx={{
            position: 'absolute',
            top: '10%',
            right: '10%',
            width: 300,
            height: 300,
            borderRadius: '50%',
            background: `radial-gradient(circle, ${alpha(theme.palette.primary.main, 0.2)} 0%, transparent 70%)`,
            filter: 'blur(60px)',
            zIndex: 0,
          }}
        />
        <MotionBox
          animate={{
            y: [0, 40, 0],
            rotate: [0, -5, 0],
          }}
          transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
          sx={{
            position: 'absolute',
            bottom: '10%',
            left: '5%',
            width: 400,
            height: 400,
            borderRadius: '50%',
            background: `radial-gradient(circle, ${alpha(theme.palette.secondary.main, 0.2)} 0%, transparent 70%)`,
            filter: 'blur(80px)',
            zIndex: 0,
          }}
        />

        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1, py: 8 }}>
          <Grid container spacing={6} alignItems="center">
            <Grid item xs={12} md={6}>
              <MotionBox
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: [0.6, 0.05, 0.01, 0.9] }}
              >
                {/* Premium Badge */}
                <MotionBox
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2, duration: 0.6 }}
                  sx={{ display: 'inline-flex', mb: 3 }}
                >
                  <Chip
                    icon={<AutoAwesomeIcon sx={{ fontSize: 16 }} />}
                    label="Quantum AI Powered"
                    sx={{
                      background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                      color: 'white',
                      fontWeight: 700,
                      fontSize: '0.875rem',
                      px: 1,
                      py: 2.5,
                      borderRadius: '12px',
                      boxShadow: `0 8px 24px ${alpha(theme.palette.primary.main, 0.3)}`,
                      '& .MuiChip-icon': { color: 'white' },
                    }}
                  />
                </MotionBox>

                {/* Main Heading */}
                <Typography
                  variant="h1"
                  component="h1"
                  sx={{
                    mb: 3,
                    fontWeight: 800,
                    fontSize: { xs: '2.75rem', md: '4rem' },
                    background: `linear-gradient(135deg, ${theme.palette.text.primary} 0%, ${theme.palette.primary.main} 100%)`,
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    lineHeight: 1.1,
                  }}
                >
                  Healthcare
                  <br />
                  Reimagined
                </Typography>

                {/* Subheading */}
                <Typography
                  variant="h5"
                  sx={{
                    mb: 4,
                    color: 'text.secondary',
                    fontWeight: 500,
                    lineHeight: 1.6,
                    fontSize: { xs: '1.125rem', md: '1.375rem' },
                  }}
                >
                  Experience next-generation AI diagnosis with unprecedented accuracy.
                  Your health insights, delivered instantly.
                </Typography>

                {/* Feature Pills - Professional & Symmetric */}
                <Stack 
                  direction="row" 
                  spacing={2} 
                  flexWrap="wrap" 
                  sx={{ 
                    gap: 2,
                    justifyContent: { xs: 'center', sm: 'flex-start' }
                  }}
                >
                  {[
                    { 
                      icon: <BoltIcon sx={{ fontSize: 24, color: 'primary.main' }} />, 
                      text: 'Instant Results' 
                    },
                    { 
                      icon: <LockIcon sx={{ fontSize: 24, color: 'primary.main' }} />, 
                      text: '100% Secure' 
                    },
                    { 
                      icon: <VerifiedIcon sx={{ fontSize: 24, color: 'primary.main' }} />, 
                      text: '99.9% Accurate' 
                    }
                  ].map((item, index) => (
                    <MotionBox
                      key={item.text}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 + index * 0.1, duration: 0.5 }}
                    >
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: 1.5,
                          minWidth: 180,
                          px: 3,
                          py: 1.5,
                          borderRadius: '14px',
                          background: `linear-gradient(135deg, 
                            ${alpha('#ffffff', 0.95)} 0%, 
                            ${alpha('#ffffff', 0.75)} 100%)`,
                          backdropFilter: 'blur(20px)',
                          border: `1px solid ${alpha('#ffffff', 0.6)}`,
                          boxShadow: `0 8px 24px ${alpha(theme.palette.primary.main, 0.1)}, 
                                      0 2px 8px ${alpha(theme.palette.primary.main, 0.06)}`,
                          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                            boxShadow: `0 12px 32px ${alpha(theme.palette.primary.main, 0.18)}, 
                                        0 4px 12px ${alpha(theme.palette.primary.main, 0.12)}`,
                            background: `linear-gradient(135deg, 
                              ${alpha('#ffffff', 1)} 0%, 
                              ${alpha('#ffffff', 0.85)} 100%)`,
                          },
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {item.icon}
                        </Box>
                        <Typography
                          sx={{
                            fontWeight: 600,
                            fontSize: '0.95rem',
                            color: 'text.primary',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {item.text}
                        </Typography>
                      </Box>
                    </MotionBox>
                  ))}
                </Stack>
              </MotionBox>
            </Grid>

            {/* Right Side - Floating Card Animation */}
            <Grid item xs={12} md={6}>
              <MotionBox
                initial={{ opacity: 0, scale: 0.9, y: 40 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ duration: 1, delay: 0.3, ease: [0.6, 0.05, 0.01, 0.9] }}
                sx={{ position: 'relative', display: 'flex', justifyContent: 'center' }}
              >
                {/* Main Glassmorphism Card */}
                <MotionBox
                  animate={{
                    y: [0, -20, 0],
                  }}
                  transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
                  sx={{
                    position: 'relative',
                    width: '100%',
                    maxWidth: 480,
                  }}
                >
                  <Box
                    sx={{
                      background: `linear-gradient(135deg, 
                        ${alpha('#ffffff', 0.9)} 0%, 
                        ${alpha('#ffffff', 0.7)} 100%)`,
                      backdropFilter: 'blur(20px)',
                      borderRadius: '32px',
                      p: 5,
                      border: `1px solid ${alpha('#ffffff', 0.8)}`,
                      boxShadow: `0 24px 80px ${alpha(theme.palette.primary.main, 0.2)}`,
                    }}
                  >
                    {/* Icon */}
                    <Box
                      sx={{
                        width: 100,
                        height: 100,
                        borderRadius: '24px',
                        background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mb: 3,
                        boxShadow: `0 12px 32px ${alpha(theme.palette.primary.main, 0.3)}`,
                      }}
                    >
                      <LocalHospitalIcon sx={{ fontSize: 56, color: 'white' }} />
                    </Box>

                    <Typography variant="h4" fontWeight={700} sx={{ mb: 2 }}>
                      Your Health, Decoded
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mb: 3, lineHeight: 1.8 }}>
                      Advanced AI analysis provides comprehensive health insights in real-time
                    </Typography>

                    {/* Stats */}
                    <Stack spacing={2}>
                      {[
                        { label: 'Diagnostic Accuracy', value: '99.9%' },
                        { label: 'Analysis Speed', value: '<3 sec' },
                        { label: 'Happy Patients', value: '50K+' },
                      ].map((stat) => (
                        <Box
                          key={stat.label}
                          sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            p: 2,
                            borderRadius: '12px',
                            bgcolor: alpha(theme.palette.primary.main, 0.06),
                          }}
                        >
                          <Typography variant="body2" fontWeight={600} color="text.secondary">
                            {stat.label}
                          </Typography>
                          <Typography variant="h6" fontWeight={700} color="primary">
                            {stat.value}
                          </Typography>
                        </Box>
                      ))}
                    </Stack>
                  </Box>
                </MotionBox>
              </MotionBox>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Modern Features Section with Glassmorphism */}
      <Box id="features">
        <Container maxWidth="lg" sx={{ py: { xs: 10, md: 16 }, position: 'relative' }}>
        {/* Section Header */}
        <MotionBox
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          sx={{ textAlign: 'center', mb: 8 }}
        >
          <Chip
            label="WHY CHOOSE US"
            sx={{
              mb: 2,
              fontWeight: 700,
              fontSize: '0.75rem',
              bgcolor: alpha(theme.palette.primary.main, 0.1),
              color: 'primary.main',
              borderRadius: '8px',
              px: 2,
            }}
          />
          <Typography
            variant="h2"
            component="h2"
            sx={{
              mb: 2,
              fontWeight: 800,
              fontSize: { xs: '2rem', md: '3rem' },
              background: `linear-gradient(135deg, ${theme.palette.text.primary} 0%, ${theme.palette.primary.main} 100%)`,
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Powered by Quantum AI
          </Typography>
          <Typography
            variant="h6"
            color="text.secondary"
            sx={{
              maxWidth: 700,
              mx: 'auto',
              fontWeight: 500,
              lineHeight: 1.8,
            }}
          >
            Advanced technology delivering unprecedented accuracy and insights for your healthcare needs
          </Typography>
        </MotionBox>

        {/* Features Grid */}
        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <MotionCard
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{
                  duration: 0.6,
                  delay: index * 0.1,
                  ease: [0.6, 0.05, 0.01, 0.9],
                }}
                whileHover={{
                  y: -12,
                  transition: { duration: 0.3, ease: 'easeOut' },
                }}
                sx={{
                  height: '100%',
                  background: `linear-gradient(135deg, 
                    ${alpha('#ffffff', 0.95)} 0%, 
                    ${alpha('#ffffff', 0.85)} 100%)`,
                  backdropFilter: 'blur(20px)',
                  border: `1px solid ${alpha('#ffffff', 0.6)}`,
                  borderRadius: '24px',
                  overflow: 'hidden',
                  position: 'relative',
                  cursor: 'pointer',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '4px',
                    background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                    opacity: 0,
                    transition: 'opacity 0.3s',
                  },
                  '&:hover::before': {
                    opacity: 1,
                  },
                  '&:hover': {
                    boxShadow: `0 20px 60px ${alpha(theme.palette.primary.main, 0.2)}`,
                  },
                }}
              >
                <CardContent sx={{ p: 4, textAlign: 'center' }}>
                  {/* Icon Container */}
                  <MotionBox
                    whileHover={{ rotate: 360, scale: 1.1 }}
                    transition={{ duration: 0.6 }}
                    sx={{
                      width: 80,
                      height: 80,
                      borderRadius: '20px',
                      background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, ${alpha(theme.palette.secondary.main, 0.1)})`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mx: 'auto',
                      mb: 3,
                      border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                    }}
                  >
                    {feature.icon}
                  </MotionBox>

                  <Typography
                    variant="h6"
                    sx={{
                      mb: 2,
                      fontWeight: 700,
                      fontSize: '1.125rem',
                    }}
                  >
                    {feature.title}
                  </Typography>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{
                      lineHeight: 1.7,
                      fontSize: '0.9375rem',
                    }}
                  >
                    {feature.description}
                  </Typography>
                </CardContent>
              </MotionCard>
            </Grid>
          ))}
        </Grid>
        </Container>
      </Box>

      {/* Modern How It Works Section */}
      <Box
        id="how-it-works"
        sx={{
          position: 'relative',
          py: { xs: 10, md: 16 },
          background: `linear-gradient(180deg, 
            ${alpha(theme.palette.primary.main, 0.02)} 0%, 
            ${alpha(theme.palette.secondary.main, 0.03)} 100%)`,
        }}
      >
        <Container maxWidth="lg">
          {/* Section Header */}
          <MotionBox
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            sx={{ textAlign: 'center', mb: 10 }}
          >
            <Chip
              label="SIMPLE PROCESS"
              sx={{
                mb: 2,
                fontWeight: 700,
                fontSize: '0.75rem',
                bgcolor: alpha(theme.palette.secondary.main, 0.1),
                color: 'secondary.main',
                borderRadius: '8px',
                px: 2,
              }}
            />
            <Typography
              variant="h2"
              component="h2"
              sx={{
                mb: 2,
                fontWeight: 800,
                fontSize: { xs: '2rem', md: '3rem' },
              }}
            >
              How It Works
            </Typography>
            <Typography
              variant="h6"
              color="text.secondary"
              sx={{
                maxWidth: 600,
                mx: 'auto',
                fontWeight: 500,
                lineHeight: 1.8,
              }}
            >
              Get accurate health insights in three simple steps
            </Typography>
          </MotionBox>

          {/* Steps with Connecting Lines */}
          <Box sx={{ position: 'relative' }}>
            {/* Connection Line */}
            <Box
              sx={{
                display: { xs: 'none', md: 'block' },
                position: 'absolute',
                top: '100px',
                left: '20%',
                right: '20%',
                height: '4px',
                background: `linear-gradient(90deg, ${alpha(theme.palette.primary.main, 0.3)}, ${alpha(theme.palette.secondary.main, 0.3)})`,
                borderRadius: '2px',
                zIndex: 0,
              }}
            />

            <Grid container spacing={6}>
              {steps.map((step, index) => (
                <Grid item xs={12} md={4} key={index}>
                  <MotionBox
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{
                      duration: 0.6,
                      delay: index * 0.2,
                      ease: [0.6, 0.05, 0.01, 0.9],
                    }}
                    sx={{ position: 'relative', zIndex: 1 }}
                  >
                    {/* Step Card */}
                    <Box
                      sx={{
                        position: 'relative',
                        p: 4,
                        borderRadius: '28px',
                        background: `linear-gradient(135deg, 
                          ${alpha('#ffffff', 0.95)} 0%, 
                          ${alpha('#ffffff', 0.9)} 100%)`,
                        backdropFilter: 'blur(20px)',
                        border: `2px solid ${alpha(theme.palette.primary.main, 0.15)}`,
                        textAlign: 'center',
                        transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                        '&:hover': {
                          transform: 'translateY(-8px)',
                          boxShadow: `0 24px 64px ${alpha(theme.palette.primary.main, 0.2)}`,
                          border: `2px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                        },
                      }}
                    >
                      {/* Number Badge */}
                      <Box
                        sx={{
                          width: 80,
                          height: 80,
                          borderRadius: '20px',
                          background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          mx: 'auto',
                          mb: 3,
                          boxShadow: `0 12px 32px ${alpha(theme.palette.primary.main, 0.35)}`,
                          position: 'relative',
                          '&::before': {
                            content: '""',
                            position: 'absolute',
                            inset: -4,
                            borderRadius: '22px',
                            background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.3)}, ${alpha(theme.palette.secondary.main, 0.3)})`,
                            zIndex: -1,
                          },
                        }}
                      >
                        <Typography
                          variant="h3"
                          sx={{
                            fontWeight: 800,
                            color: 'white',
                            fontSize: '2rem',
                          }}
                        >
                          {step.number}
                        </Typography>
                      </Box>

                      <Typography
                        variant="h5"
                        sx={{
                          mb: 2,
                          fontWeight: 700,
                          fontSize: '1.375rem',
                        }}
                      >
                        {step.title}
                      </Typography>
                      <Typography
                        variant="body1"
                        color="text.secondary"
                        sx={{
                          lineHeight: 1.8,
                          fontSize: '1rem',
                        }}
                      >
                        {step.description}
                      </Typography>
                    </Box>
                  </MotionBox>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Container>
      </Box>

      {/* Modern Trust/Security Section */}
      <Container maxWidth="lg" sx={{ py: { xs: 10, md: 16 } }}>
        <MotionBox
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <Box
            sx={{
              position: 'relative',
              borderRadius: '32px',
              overflow: 'hidden',
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
              p: { xs: 5, md: 8 },
              color: 'white',
              boxShadow: `0 24px 80px ${alpha(theme.palette.primary.main, 0.35)}`,
            }}
          >
            {/* Decorative Elements */}
            <Box
              sx={{
                position: 'absolute',
                top: -50,
                right: -50,
                width: 300,
                height: 300,
                borderRadius: '50%',
                background: alpha('#ffffff', 0.1),
                filter: 'blur(60px)',
              }}
            />
            <Box
              sx={{
                position: 'absolute',
                bottom: -80,
                left: -80,
                width: 350,
                height: 350,
                borderRadius: '50%',
                background: alpha('#ffffff', 0.1),
                filter: 'blur(80px)',
              }}
            />

            <Box sx={{ position: 'relative', zIndex: 1, textAlign: 'center' }}>
              {/* Icon */}
              <MotionBox
                animate={{
                  scale: [1, 1.1, 1],
                }}
                transition={{ duration: 3, repeat: Infinity }}
                sx={{ display: 'inline-block', mb: 3 }}
              >
                <Box
                  sx={{
                    width: 100,
                    height: 100,
                    borderRadius: '24px',
                    background: alpha('#ffffff', 0.2),
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mx: 'auto',
                    backdropFilter: 'blur(10px)',
                  }}
                >
                  <VerifiedUserIcon sx={{ fontSize: 56 }} />
                </Box>
              </MotionBox>

              <Typography
                variant="h3"
                sx={{
                  mb: 2,
                  fontWeight: 800,
                  fontSize: { xs: '2rem', md: '2.75rem' },
                }}
              >
                Your Privacy is Our Priority
              </Typography>
              <Typography
                variant="h6"
                sx={{
                  mb: 5,
                  opacity: 0.95,
                  maxWidth: 800,
                  mx: 'auto',
                  fontWeight: 500,
                  lineHeight: 1.8,
                  fontSize: { xs: '1rem', md: '1.125rem' },
                }}
              >
                InsightCare adheres to the highest standards of data security and privacy compliance.
                Your medical information is encrypted end-to-end and never shared without your explicit consent.
              </Typography>

              {/* Compliance Badges */}
              <Stack
                direction={{ xs: 'column', sm: 'row' }}
                spacing={2}
                justifyContent="center"
                flexWrap="wrap"
                sx={{ gap: 2 }}
              >
                {['HIPAA Compliant', 'GDPR Certified', 'End-to-End Encrypted'].map((label, index) => (
                  <MotionBox
                    key={label}
                    initial={{ opacity: 0, scale: 0.8 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1, duration: 0.5 }}
                  >
                    <Chip
                      icon={<CheckCircleIcon />}
                      label={label}
                      sx={{
                        bgcolor: 'white',
                        color: 'primary.main',
                        fontWeight: 700,
                        fontSize: '1rem',
                        px: 2,
                        py: 3.5,
                        '& .MuiChip-icon': {
                          color: 'success.main',
                        },
                      }}
                    />
                  </MotionBox>
                ))}
              </Stack>
            </Box>
          </Box>
        </MotionBox>
      </Container>

      {/* FAQ Section */}
      <Box
        id="faq"
        sx={{
          py: { xs: 10, md: 16 },
          background: `linear-gradient(180deg, 
            ${alpha(theme.palette.background.default, 0.5)} 0%, 
            ${alpha(theme.palette.primary.main, 0.02)} 100%)`,
        }}
      >
        <Container maxWidth="md">
          <MotionBox
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            sx={{ textAlign: 'center', mb: 8 }}
          >
            <Chip
              label="FAQ"
              sx={{
                mb: 2,
                fontWeight: 700,
                fontSize: '0.75rem',
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                color: 'primary.main',
                borderRadius: '8px',
                px: 2,
              }}
            />
            <Typography
              variant="h2"
              component="h2"
              sx={{
                mb: 2,
                fontWeight: 800,
                fontSize: { xs: '2rem', md: '3rem' },
                background: `linear-gradient(135deg, ${theme.palette.text.primary} 0%, ${theme.palette.primary.main} 100%)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Frequently Asked Questions
            </Typography>
            <Typography
              variant="h6"
              color="text.secondary"
              sx={{
                maxWidth: 600,
                mx: 'auto',
                fontWeight: 500,
                lineHeight: 1.8,
              }}
            >
              Everything you need to know about InsightCare
            </Typography>
          </MotionBox>

          {/* FAQ Items */}
          <Stack spacing={3}>
            {[
              {
                q: 'How accurate is InsightCare\'s AI diagnosis?',
                a: 'Our quantum-enhanced AI achieves 99.9% accuracy by analyzing complex medical patterns. However, it\'s designed to complement, not replace, professional medical advice.'
              },
              {
                q: 'Is my health data secure?',
                a: 'Absolutely. We use end-to-end encryption and comply with HIPAA and GDPR standards. Your data is never shared without your explicit consent.'
              },
              {
                q: 'How long does the analysis take?',
                a: 'Our AI processes your symptoms and delivers comprehensive results in under 3 seconds, providing you with instant health insights.'
              },
              {
                q: 'Do I need medical knowledge to use InsightCare?',
                a: 'Not at all! Our intuitive interface guides you through the process. Simply describe your symptoms in plain language.'
              },
              {
                q: 'Can I use InsightCare for emergency situations?',
                a: 'InsightCare is designed for non-emergency health assessments. For emergencies, always call emergency services or visit your nearest hospital.'
              },
              {
                q: 'How much does InsightCare cost?',
                a: 'We offer a free tier for basic analysis. Premium plans start at $9.99/month for advanced features and unlimited diagnoses.'
              },
            ].map((faq, index) => (
              <MotionBox
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <Box
                  sx={{
                    p: 4,
                    borderRadius: '20px',
                    background: `linear-gradient(135deg, 
                      ${alpha('#ffffff', 0.95)} 0%, 
                      ${alpha('#ffffff', 0.9)} 100%)`,
                    backdropFilter: 'blur(20px)',
                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    transition: 'all 0.3s',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: `0 12px 32px ${alpha(theme.palette.primary.main, 0.15)}`,
                    },
                  }}
                >
                  <Typography
                    variant="h6"
                    sx={{
                      mb: 1.5,
                      fontWeight: 700,
                      color: 'primary.main',
                    }}
                  >
                    {faq.q}
                  </Typography>
                  <Typography
                    variant="body1"
                    color="text.secondary"
                    sx={{ lineHeight: 1.8 }}
                  >
                    {faq.a}
                  </Typography>
                </Box>
              </MotionBox>
            ))}
          </Stack>
        </Container>
      </Box>

      {/* Modern CTA Section */}
      <Box
        sx={{
          position: 'relative',
          py: { xs: 10, md: 16 },
          background: `linear-gradient(180deg, 
            ${alpha(theme.palette.primary.main, 0.03)} 0%, 
            transparent 100%)`,
        }}
      >
        <Container maxWidth="md">
          <MotionBox
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            sx={{ textAlign: 'center' }}
          >
            <Typography
              variant="h2"
              sx={{
                mb: 2,
                fontWeight: 800,
                fontSize: { xs: '2rem', md: '3rem' },
                background: `linear-gradient(135deg, ${theme.palette.text.primary} 0%, ${theme.palette.primary.main} 100%)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Ready to Transform Your Healthcare?
            </Typography>
            <Typography
              variant="h6"
              color="text.secondary"
              sx={{
                mb: 5,
                fontWeight: 500,
                lineHeight: 1.8,
                maxWidth: 600,
                mx: 'auto',
              }}
            >
              Join thousands of users experiencing the future of AI-powered health diagnosis
            </Typography>

            <Stack
              direction={{ xs: 'column', sm: 'row' }}
              spacing={2}
              justifyContent="center"
            >
              <Button
                variant="contained"
                size="large"
                onClick={() => router.push(session ? '/symptoms' : '/login')}
                endIcon={<RocketLaunchIcon />}
                sx={{
                  px: 5,
                  py: 2,
                  fontSize: '1.125rem',
                  fontWeight: 700,
                  borderRadius: '14px',
                  background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  boxShadow: `0 12px 32px ${alpha(theme.palette.primary.main, 0.35)}`,
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: `0 20px 48px ${alpha(theme.palette.primary.main, 0.45)}`,
                  },
                }}
              >
                {session ? 'Start Analysis Now' : 'Get Started Free'}
              </Button>
            </Stack>

            {/* Social Proof */}
            <Box sx={{ mt: 6 }}>
              <Stack
                direction="row"
                spacing={4}
                justifyContent="center"
                flexWrap="wrap"
                sx={{ gap: 3 }}
              >
                <Box>
                  <Typography variant="h4" fontWeight={800} color="primary.main">
                    50K+
                  </Typography>
                  <Typography variant="body2" color="text.secondary" fontWeight={600}>
                    Active Users
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="h4" fontWeight={800} color="primary.main">
                    99.9%
                  </Typography>
                  <Typography variant="body2" color="text.secondary" fontWeight={600}>
                    Accuracy Rate
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="h4" fontWeight={800} color="primary.main">
                    4.9/5
                  </Typography>
                  <Typography variant="body2" color="text.secondary" fontWeight={600}>
                    User Rating
                  </Typography>
                </Box>
              </Stack>
            </Box>
          </MotionBox>
        </Container>
      </Box>
    </Box>
  );
}
