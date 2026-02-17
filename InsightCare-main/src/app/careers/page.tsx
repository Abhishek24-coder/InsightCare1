'use client';

import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Button,
  Stack,
  Chip,
  useTheme,
  alpha,
} from '@mui/material';
import { motion } from 'framer-motion';
import WorkIcon from '@mui/icons-material/Work';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import AttachMoneyIcon from '@mui/icons-material/AttachMoney';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import SchoolIcon from '@mui/icons-material/School';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import HomeWorkIcon from '@mui/icons-material/HomeWork';

const MotionBox = motion(Box);
const MotionCard = motion(Card);

export default function CareersPage() {
  const theme = useTheme();

  const benefits = [
    { icon: <AttachMoneyIcon />, title: 'Competitive Salary', description: 'Top-tier compensation packages' },
    { icon: <HealthAndSafetyIcon />, title: 'Health Insurance', description: 'Comprehensive medical coverage' },
    { icon: <HomeWorkIcon />, title: 'Remote Work', description: 'Flexible work-from-home options' },
    { icon: <SchoolIcon />, title: 'Learning Budget', description: '$2000 annual learning allowance' },
  ];

  const openings = [
    {
      title: 'Senior AI/ML Engineer',
      department: 'Engineering',
      location: 'Remote / San Francisco',
      type: 'Full-time',
      salary: '$150K - $200K',
      description: 'Build next-generation AI models for medical diagnosis. Experience with PyTorch and TensorFlow required.'
    },
    {
      title: 'Product Designer',
      department: 'Design',
      location: 'Remote / New York',
      type: 'Full-time',
      salary: '$120K - $160K',
      description: 'Design beautiful, intuitive healthcare experiences. 5+ years of product design experience required.'
    },
    {
      title: 'Full Stack Developer',
      department: 'Engineering',
      location: 'Remote',
      type: 'Full-time',
      salary: '$130K - $170K',
      description: 'Work on our Next.js platform. Strong TypeScript and React skills required.'
    },
    {
      title: 'Medical Content Writer',
      department: 'Content',
      location: 'Remote',
      type: 'Contract',
      salary: '$80K - $100K',
      description: 'Create accurate, accessible medical content. Medical degree or nursing background required.'
    },
    {
      title: 'DevOps Engineer',
      department: 'Engineering',
      location: 'Remote / London',
      type: 'Full-time',
      salary: '$140K - $180K',
      description: 'Manage our cloud infrastructure and CI/CD pipelines. AWS and Kubernetes expertise required.'
    },
    {
      title: 'Customer Success Manager',
      department: 'Customer Success',
      location: 'Remote',
      type: 'Full-time',
      salary: '$90K - $120K',
      description: 'Help our users get the most out of InsightCare. Healthcare industry experience preferred.'
    },
  ];

  return (
    <Box>
      {/* Hero Section */}
      <Box
        sx={{
          position: 'relative',
          pt: { xs: 12, md: 16 },
          pb: { xs: 8, md: 12 },
          background: `linear-gradient(135deg, 
            ${alpha(theme.palette.primary.main, 0.05)} 0%, 
            ${alpha(theme.palette.secondary.main, 0.05)} 100%)`,
        }}
      >
        <Container maxWidth="lg">
          <MotionBox
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            sx={{ textAlign: 'center', mb: 6 }}
          >
            <Typography
              variant="h1"
              sx={{
                mb: 3,
                fontWeight: 800,
                fontSize: { xs: '2.5rem', md: '4rem' },
                background: `linear-gradient(135deg, ${theme.palette.text.primary} 0%, ${theme.palette.primary.main} 100%)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Join Our Mission
            </Typography>
            <Typography
              variant="h5"
              color="text.secondary"
              sx={{
                maxWidth: 700,
                mx: 'auto',
              fontWeight: 500,
              lineHeight: 1.8,
            }}
          >
            Help us revolutionize healthcare and make a real difference in people&apos;s lives.
            We&apos;re looking for passionate individuals to join our team.
          </Typography>
        </MotionBox>
      </Container>
    </Box>      {/* Benefits Section */}
      <Container maxWidth="lg" sx={{ py: { xs: 8, md: 12 } }}>
        <Typography
          variant="h2"
          textAlign="center"
          sx={{
            mb: 6,
            fontWeight: 800,
            fontSize: { xs: '2rem', md: '3rem' },
          }}
        >
          Why Work With Us?
        </Typography>

        <Grid container spacing={4}>
          {benefits.map((benefit, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <MotionCard
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.6 }}
                whileHover={{ y: -8 }}
                sx={{
                  p: 3,
                  textAlign: 'center',
                  height: '100%',
                  borderRadius: '20px',
                  background: 'white',
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
              >
                <Box
                  sx={{
                    width: 64,
                    height: 64,
                    borderRadius: '16px',
                    background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, ${alpha(theme.palette.secondary.main, 0.1)})`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mx: 'auto',
                    mb: 2,
                    color: 'primary.main',
                  }}
                >
                  {benefit.icon}
                </Box>
                <Typography variant="h6" fontWeight={700} sx={{ mb: 1 }}>
                  {benefit.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {benefit.description}
                </Typography>
              </MotionCard>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Job Openings */}
      <Box sx={{ bgcolor: alpha(theme.palette.background.default, 0.5), py: { xs: 8, md: 12 } }}>
        <Container maxWidth="lg">
          <Typography
            variant="h2"
            textAlign="center"
            sx={{
              mb: 2,
              fontWeight: 800,
              fontSize: { xs: '2rem', md: '3rem' },
            }}
          >
            Open Positions
          </Typography>
          <Typography
            variant="h6"
            textAlign="center"
            color="text.secondary"
            sx={{ mb: 6 }}
          >
            {openings.length} positions available across multiple departments
          </Typography>

          <Stack spacing={3}>
            {openings.map((job, index) => (
              <MotionCard
                key={index}
                initial={{ opacity: 0, x: -40 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.05, duration: 0.5 }}
                sx={{
                  p: 4,
                  borderRadius: '20px',
                  background: 'white',
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  transition: 'all 0.3s',
                  '&:hover': {
                    transform: 'translateX(8px)',
                    boxShadow: `0 12px 32px ${alpha(theme.palette.primary.main, 0.15)}`,
                  },
                }}
              >
                <Grid container spacing={3} alignItems="center">
                  <Grid item xs={12} md={8}>
                    <Stack spacing={2}>
                      <Box>
                        <Typography variant="h5" fontWeight={700} sx={{ mb: 1 }}>
                          {job.title}
                        </Typography>
                        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                          {job.description}
                        </Typography>
                      </Box>
                      <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1 }}>
                        <Chip
                          icon={<WorkIcon sx={{ fontSize: 16 }} />}
                          label={job.department}
                          size="small"
                          sx={{ bgcolor: alpha(theme.palette.primary.main, 0.1) }}
                        />
                        <Chip
                          icon={<LocationOnIcon sx={{ fontSize: 16 }} />}
                          label={job.location}
                          size="small"
                          sx={{ bgcolor: alpha(theme.palette.secondary.main, 0.1) }}
                        />
                        <Chip
                          icon={<AccessTimeIcon sx={{ fontSize: 16 }} />}
                          label={job.type}
                          size="small"
                          sx={{ bgcolor: alpha(theme.palette.info.main, 0.1) }}
                        />
                        <Chip
                          icon={<AttachMoneyIcon sx={{ fontSize: 16 }} />}
                          label={job.salary}
                          size="small"
                          sx={{ bgcolor: alpha(theme.palette.success.main, 0.1) }}
                        />
                      </Stack>
                    </Stack>
                  </Grid>
                  <Grid item xs={12} md={4} sx={{ textAlign: { xs: 'left', md: 'right' } }}>
                    <Button
                      variant="contained"
                      size="large"
                      sx={{
                        borderRadius: '12px',
                        px: 4,
                        background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                        '&:hover': {
                          transform: 'translateY(-2px)',
                        },
                      }}
                    >
                      Apply Now
                    </Button>
                  </Grid>
                </Grid>
              </MotionCard>
            ))}
          </Stack>
        </Container>
      </Box>

      {/* CTA Section */}
      <Container maxWidth="md" sx={{ py: { xs: 8, md: 12 }, textAlign: 'center' }}>
        <MotionBox
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <Typography variant="h3" fontWeight={800} sx={{ mb: 2 }}>
            Don&apos;t See Your Role?
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
            We&apos;re always looking for talented people. Send us your resume and we&apos;ll keep you in mind
            for future opportunities.
          </Typography>
          <Button
            variant="outlined"
            size="large"
            sx={{
              borderRadius: '12px',
              px: 4,
              py: 1.5,
              borderWidth: 2,
              '&:hover': {
                borderWidth: 2,
              },
            }}
          >
            Submit General Application
          </Button>
        </MotionBox>
      </Container>
    </Box>
  );
}
