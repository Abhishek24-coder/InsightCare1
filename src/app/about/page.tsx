'use client';

import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Avatar,
  Stack,
  useTheme,
  alpha,
} from '@mui/material';
import { motion } from 'framer-motion';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import VisibilityIcon from '@mui/icons-material/Visibility';
import FavoriteIcon from '@mui/icons-material/Favorite';
import SecurityIcon from '@mui/icons-material/Security';
import PeopleIcon from '@mui/icons-material/People';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

const MotionBox = motion(Box);
const MotionCard = motion(Card);

export default function AboutPage() {
  const theme = useTheme();

  const values = [
    {
      icon: <FavoriteIcon sx={{ fontSize: 40 }} />,
      title: 'Patient-Centric',
      description: 'We put patients first in everything we do, ensuring accessible and personalized healthcare for all.'
    },
    {
      icon: <SecurityIcon sx={{ fontSize: 40 }} />,
      title: 'Privacy & Security',
      description: 'Your health data is protected with military-grade encryption and strict compliance standards.'
    },
    {
      icon: <RocketLaunchIcon sx={{ fontSize: 40 }} />,
      title: 'Innovation',
      description: 'Leveraging cutting-edge AI and quantum computing to push the boundaries of medical diagnosis.'
    },
  ];

  const stats = [
    { value: '50K+', label: 'Active Users' },
    { value: '99.9%', label: 'Accuracy Rate' },
    { value: '24/7', label: 'Availability' },
    { value: '150+', label: 'Countries' },
  ];

  const team = [
    { name: 'Yash Sharma', role: 'CEO & Founder', avatar: 'Y' },
    { name: 'Harsh Parmar', role: 'Chief Technology Officer', avatar: 'H' },
    { name: 'Abhishek Shrivastava', role: 'Head of AI Research', avatar: 'A' },
    { name: 'Lavlesh Yadav', role: 'VP of Engineering', avatar: 'L' },
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
            sx={{ textAlign: 'center', mb: 8 }}
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
              About InsightCare
            </Typography>
            <Typography
              variant="h5"
              color="text.secondary"
              sx={{
                maxWidth: 800,
                mx: 'auto',
                fontWeight: 500,
                lineHeight: 1.8,
              }}
            >
              Revolutionizing healthcare through AI-powered diagnosis and personalized medicine.
              Our mission is to make world-class healthcare accessible to everyone, everywhere.
            </Typography>
          </MotionBox>

          {/* Stats */}
          <Grid container spacing={3} sx={{ mb: 8 }}>
            {stats.map((stat, index) => (
              <Grid item xs={6} md={3} key={index}>
                <MotionBox
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1, duration: 0.5 }}
                  sx={{
                    textAlign: 'center',
                    p: 3,
                    borderRadius: '20px',
                    background: `linear-gradient(135deg, 
                      ${alpha('#ffffff', 0.95)} 0%, 
                      ${alpha('#ffffff', 0.9)} 100%)`,
                    backdropFilter: 'blur(20px)',
                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  }}
                >
                  <Typography
                    variant="h3"
                    sx={{
                      fontWeight: 800,
                      background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      mb: 1,
                    }}
                  >
                    {stat.value}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" fontWeight={600}>
                    {stat.label}
                  </Typography>
                </MotionBox>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Mission & Vision */}
      <Container maxWidth="lg" sx={{ py: { xs: 8, md: 12 } }}>
        <Grid container spacing={6}>
          <Grid item xs={12} md={6}>
            <MotionCard
              initial={{ opacity: 0, x: -40 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              sx={{
                height: '100%',
                p: 4,
                borderRadius: '24px',
                background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)}, ${alpha(theme.palette.primary.main, 0.02)})`,
                border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
              }}
            >
              <Box
                sx={{
                  width: 60,
                  height: 60,
                  borderRadius: '16px',
                  background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mb: 3,
                }}
              >
                <RocketLaunchIcon sx={{ fontSize: 32, color: 'white' }} />
              </Box>
              <Typography variant="h4" fontWeight={700} sx={{ mb: 2 }}>
                Our Mission
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                To democratize access to world-class healthcare by leveraging artificial intelligence
                and quantum computing. We believe everyone deserves accurate, fast, and affordable
                medical diagnosis, regardless of their location or economic status.
              </Typography>
            </MotionCard>
          </Grid>

          <Grid item xs={12} md={6}>
            <MotionCard
              initial={{ opacity: 0, x: 40 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              sx={{
                height: '100%',
                p: 4,
                borderRadius: '24px',
                background: `linear-gradient(135deg, ${alpha(theme.palette.secondary.main, 0.05)}, ${alpha(theme.palette.secondary.main, 0.02)})`,
                border: `1px solid ${alpha(theme.palette.secondary.main, 0.1)}`,
              }}
            >
              <Box
                sx={{
                  width: 60,
                  height: 60,
                  borderRadius: '16px',
                  background: `linear-gradient(135deg, ${theme.palette.secondary.main}, ${theme.palette.primary.main})`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mb: 3,
                }}
              >
                <VisibilityIcon sx={{ fontSize: 32, color: 'white' }} />
              </Box>
              <Typography variant="h4" fontWeight={700} sx={{ mb: 2 }}>
                Our Vision
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                To become the global standard for AI-powered healthcare diagnosis, helping billions
                of people make informed health decisions. We envision a future where advanced medical
                insights are just a click away for everyone on Earth.
              </Typography>
            </MotionCard>
          </Grid>
        </Grid>
      </Container>

      {/* Core Values */}
      <Box sx={{ bgcolor: alpha(theme.palette.background.default, 0.5), py: { xs: 8, md: 12 } }}>
        <Container maxWidth="lg">
          <Typography
            variant="h2"
            textAlign="center"
            sx={{
              mb: 6,
              fontWeight: 800,
              fontSize: { xs: '2rem', md: '3rem' },
            }}
          >
            Our Core Values
          </Typography>

          <Grid container spacing={4}>
            {values.map((value, index) => (
              <Grid item xs={12} md={4} key={index}>
                <MotionCard
                  initial={{ opacity: 0, y: 40 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.2, duration: 0.6 }}
                  whileHover={{ y: -8 }}
                  sx={{
                    p: 4,
                    textAlign: 'center',
                    height: '100%',
                    borderRadius: '24px',
                    background: 'white',
                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  }}
                >
                  <Box
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
                      color: 'primary.main',
                    }}
                  >
                    {value.icon}
                  </Box>
                  <Typography variant="h5" fontWeight={700} sx={{ mb: 2 }}>
                    {value.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                    {value.description}
                  </Typography>
                </MotionCard>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Team Section */}
      <Container maxWidth="lg" sx={{ py: { xs: 8, md: 12 } }}>
        <Typography
          variant="h2"
          textAlign="center"
          sx={{
            mb: 2,
            fontWeight: 800,
            fontSize: { xs: '2rem', md: '3rem' },
          }}
        >
          Meet Our Team
        </Typography>
        <Typography
          variant="h6"
          textAlign="center"
          color="text.secondary"
          sx={{ mb: 6, maxWidth: 600, mx: 'auto' }}
        >
          Passionate experts dedicated to transforming healthcare through innovation
        </Typography>

        <Grid container spacing={4}>
          {team.map((member, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <MotionBox
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
                sx={{ textAlign: 'center' }}
              >
                <Avatar
                  sx={{
                    width: 120,
                    height: 120,
                    mx: 'auto',
                    mb: 2,
                    fontSize: '3rem',
                    fontWeight: 700,
                    background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                    boxShadow: `0 8px 24px ${alpha(theme.palette.primary.main, 0.3)}`,
                  }}
                >
                  {member.avatar}
                </Avatar>
                <Typography variant="h6" fontWeight={700} sx={{ mb: 0.5 }}>
                  {member.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {member.role}
                </Typography>
              </MotionBox>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* CTA Section */}
      <Box
        sx={{
          bgcolor: alpha(theme.palette.primary.main, 0.03),
          py: { xs: 8, md: 12 },
          textAlign: 'center',
        }}
      >
        <Container maxWidth="md">
          <Typography variant="h3" fontWeight={800} sx={{ mb: 2 }}>
            Join Us in Transforming Healthcare
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
            Interested in being part of our mission? Check out our open positions.
          </Typography>
          <Stack direction="row" spacing={2} justifyContent="center">
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Box
                component="a"
                href="/careers"
                sx={{
                  display: 'inline-block',
                  px: 4,
                  py: 2,
                  borderRadius: '12px',
                  background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  color: 'white',
                  textDecoration: 'none',
                  fontWeight: 700,
                  fontSize: '1.125rem',
                  boxShadow: `0 8px 24px ${alpha(theme.palette.primary.main, 0.3)}`,
                }}
              >
                View Open Positions
              </Box>
            </motion.div>
          </Stack>
        </Container>
      </Box>
    </Box>
  );
}
