'use client';

import {
  Box,
  Container,
  Typography,
  Link as MuiLink,
  Grid,
  Stack,
  IconButton,
  Divider,
  useTheme,
  alpha,
} from '@mui/material';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import TwitterIcon from '@mui/icons-material/Twitter';
import FacebookIcon from '@mui/icons-material/Facebook';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import InstagramIcon from '@mui/icons-material/Instagram';
import EmailIcon from '@mui/icons-material/Email';
import PhoneIcon from '@mui/icons-material/Phone';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import FavoriteIcon from '@mui/icons-material/Favorite';

export default function Footer() {
  const theme = useTheme();
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    product: [
      { label: 'Features', href: '/#features' },
      { label: 'How It Works', href: '/#how-it-works' },
      { label: 'Pricing', href: '/#pricing' },
      { label: 'FAQ', href: '/#faq' },
    ],
    company: [
      { label: 'About Us', href: '/about' },
      { label: 'Careers', href: '/careers' },
      { label: 'Blog', href: '/blog' },
      { label: 'Press Kit', href: '/press' },
    ],
    legal: [
      { label: 'Privacy Policy', href: '/privacy' },
      { label: 'Terms of Service', href: '/terms' },
      { label: 'Cookie Policy', href: '/cookies' },
      { label: 'HIPAA Compliance', href: '/hipaa' },
    ],
    support: [
      { label: 'Help Center', href: '/help' },
      { label: 'Contact Us', href: '/contact' },
      { label: 'Status', href: '/status' },
      { label: 'API Documentation', href: '/docs' },
    ],
  };

  return (
    <Box
      component="footer"
      sx={{
        mt: 'auto',
        background: `linear-gradient(180deg, 
          ${alpha(theme.palette.background.default, 0.5)} 0%, 
          ${theme.palette.grey[100]} 100%)`,
        borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      }}
    >
      <Container maxWidth="lg">
        {/* Main Footer Content */}
        <Box sx={{ py: 8 }}>
          <Grid container spacing={4}>
            {/* Brand Section */}
            <Grid item xs={12} md={4}>
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <Box
                    sx={{
                      width: 40,
                      height: 40,
                      borderRadius: '10px',
                      background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.3)}`,
                    }}
                  >
                    <LocalHospitalIcon sx={{ color: 'white', fontSize: 24 }} />
                  </Box>
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: 800,
                      background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                    }}
                  >
                    InsightCare
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3, lineHeight: 1.8 }}>
                  Next-generation AI-powered healthcare diagnosis platform. Experience the future
                  of personalized medicine.
                </Typography>
                {/* Social Links */}
                <Stack direction="row" spacing={1}>
                  {[
                    { icon: <TwitterIcon />, href: '#' },
                    { icon: <FacebookIcon />, href: '#' },
                    { icon: <LinkedInIcon />, href: '#' },
                    { icon: <InstagramIcon />, href: '#' },
                  ].map((social, index) => (
                    <IconButton
                      key={index}
                      href={social.href}
                      sx={{
                        bgcolor: alpha(theme.palette.primary.main, 0.08),
                        color: 'primary.main',
                        '&:hover': {
                          bgcolor: 'primary.main',
                          color: 'white',
                          transform: 'translateY(-2px)',
                        },
                        transition: 'all 0.3s',
                      }}
                    >
                      {social.icon}
                    </IconButton>
                  ))}
                </Stack>
              </Box>
            </Grid>

            {/* Links Sections */}
            <Grid item xs={12} md={8}>
              <Grid container spacing={3}>
                {Object.entries(footerLinks).map(([category, links]) => (
                  <Grid item xs={6} sm={3} key={category}>
                    <Typography
                      variant="subtitle2"
                      sx={{
                        fontWeight: 700,
                        mb: 2,
                        textTransform: 'uppercase',
                        fontSize: '0.8125rem',
                        letterSpacing: '0.05em',
                      }}
                    >
                      {category}
                    </Typography>
                    <Stack spacing={1.5}>
                      {links.map((link) => (
                        <MuiLink
                          key={link.label}
                          href={link.href}
                          sx={{
                            color: 'text.secondary',
                            textDecoration: 'none',
                            fontSize: '0.875rem',
                            fontWeight: 500,
                            transition: 'all 0.2s',
                            '&:hover': {
                              color: 'primary.main',
                              pl: 0.5,
                            },
                          }}
                        >
                          {link.label}
                        </MuiLink>
                      ))}
                    </Stack>
                  </Grid>
                ))}
              </Grid>
            </Grid>
          </Grid>
        </Box>

        <Divider sx={{ opacity: 0.5 }} />

        {/* Bottom Footer */}
        <Box
          sx={{
            py: 3,
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              fontSize: '0.875rem',
            }}
          >
            © {currentYear} InsightCare. Made with{' '}
            <FavoriteIcon sx={{ fontSize: 16, color: 'error.main' }} /> for better healthcare.
          </Typography>
          <Stack direction="row" spacing={2} alignItems="center">
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <EmailIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
              <Typography variant="body2" color="text.secondary" fontSize="0.875rem">
                support@insightcare.com
              </Typography>
            </Box>
          </Stack>
        </Box>
      </Container>
    </Box>
  );
}
