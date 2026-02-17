'use client';

import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  TextField,
  Button,
  Stack,
  useTheme,
  alpha,
} from '@mui/material';
import { motion } from 'framer-motion';
import { useState } from 'react';
import EmailIcon from '@mui/icons-material/Email';
import PhoneIcon from '@mui/icons-material/Phone';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import SendIcon from '@mui/icons-material/Send';

const MotionBox = motion(Box);
const MotionCard = motion(Card);

export default function ContactPage() {
  const theme = useTheme();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement contact form submission
    alert('Thank you for contacting us! We\'ll get back to you soon.');
    setFormData({ name: '', email: '', subject: '', message: '' });
  };

  const contactInfo = [
    {
      icon: <EmailIcon sx={{ fontSize: 32 }} />,
      title: 'Email Us',
      value: 'support@insightcare.com',
      link: 'mailto:support@insightcare.com'
    },
    {
      icon: <PhoneIcon sx={{ fontSize: 32 }} />,
      title: 'Call Us',
      value: '+1 (555) 123-4567',
      link: 'tel:+15551234567'
    },
    {
      icon: <LocationOnIcon sx={{ fontSize: 32 }} />,
      title: 'Visit Us',
      value: '123 Healthcare Ave, San Francisco, CA 94102',
      link: 'https://maps.google.com'
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
              Get In Touch
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
            Have questions? We&apos;d love to hear from you. Send us a message and we&apos;ll respond as soon as possible.
          </Typography>
        </MotionBox>
      </Container>
    </Box>      {/* Contact Info Cards */}
      <Container maxWidth="lg" sx={{ py: { xs: 8, md: 12 } }}>
        <Grid container spacing={4} sx={{ mb: 8 }}>
          {contactInfo.map((info, index) => (
            <Grid item xs={12} md={4} key={index}>
              <MotionCard
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.6 }}
                sx={{
                  p: 4,
                  textAlign: 'center',
                  height: '100%',
                  borderRadius: '20px',
                  background: 'white',
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  textDecoration: 'none',
                  transition: 'all 0.3s',
                  cursor: 'pointer',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: `0 12px 32px ${alpha(theme.palette.primary.main, 0.15)}`,
                  },
                }}
                onClick={() => window.open(info.link, '_blank')}
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
                  {info.icon}
                </Box>
                <Typography variant="h6" fontWeight={700} sx={{ mb: 1 }}>
                  {info.title}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  {info.value}
                </Typography>
              </MotionCard>
            </Grid>
          ))}
        </Grid>

        {/* Contact Form */}
        <Grid container spacing={6}>
          <Grid item xs={12} md={6}>
            <MotionBox
              initial={{ opacity: 0, x: -40 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <Typography variant="h3" fontWeight={800} sx={{ mb: 2 }}>
                Send Us a Message
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 4, lineHeight: 1.8 }}>
                Fill out the form below and our team will get back to you within 24 hours.
                For urgent matters, please call us directly.
              </Typography>

              <Stack spacing={2}>
                <Box>
                  <Typography variant="body2" fontWeight={600} sx={{ mb: 1 }}>
                    📧 General Inquiries
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    support@insightcare.com
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" fontWeight={600} sx={{ mb: 1 }}>
                    💼 Business & Partnerships
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    partnerships@insightcare.com
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" fontWeight={600} sx={{ mb: 1 }}>
                    🔒 Security & Privacy
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    security@insightcare.com
                  </Typography>
                </Box>
              </Stack>
            </MotionBox>
          </Grid>

          <Grid item xs={12} md={6}>
            <MotionCard
              initial={{ opacity: 0, x: 40 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              sx={{
                p: 4,
                borderRadius: '24px',
                background: 'white',
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                boxShadow: `0 8px 32px ${alpha(theme.palette.primary.main, 0.1)}`,
              }}
            >
              <form onSubmit={handleSubmit}>
                <Stack spacing={3}>
                  <TextField
                    fullWidth
                    label="Your Name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    required
                  />
                  <TextField
                    fullWidth
                    label="Email Address"
                    name="email"
                    type="email"
                    value={formData.email}
                    onChange={handleChange}
                    required
                  />
                  <TextField
                    fullWidth
                    label="Subject"
                    name="subject"
                    value={formData.subject}
                    onChange={handleChange}
                    required
                  />
                  <TextField
                    fullWidth
                    label="Message"
                    name="message"
                    multiline
                    rows={4}
                    value={formData.message}
                    onChange={handleChange}
                    required
                  />
                  <Button
                    type="submit"
                    variant="contained"
                    size="large"
                    endIcon={<SendIcon />}
                    sx={{
                      borderRadius: '12px',
                      py: 1.5,
                      background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                      boxShadow: `0 8px 24px ${alpha(theme.palette.primary.main, 0.3)}`,
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: `0 12px 32px ${alpha(theme.palette.primary.main, 0.4)}`,
                      },
                    }}
                  >
                    Send Message
                  </Button>
                </Stack>
              </form>
            </MotionCard>
          </Grid>
        </Grid>
      </Container>

      {/* Map Section (Placeholder) */}
      <Box
        sx={{
          bgcolor: alpha(theme.palette.background.default, 0.5),
          py: { xs: 8, md: 12 },
        }}
      >
        <Container maxWidth="lg">
          <Box
            sx={{
              height: 400,
              borderRadius: '24px',
              overflow: 'hidden',
              background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, ${alpha(theme.palette.secondary.main, 0.1)})`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Typography variant="h5" color="text.secondary">
              🗺️ Map Placeholder - San Francisco Office
            </Typography>
          </Box>
        </Container>
      </Box>
    </Box>
  );
}
