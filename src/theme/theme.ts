'use client';

import { createTheme } from '@mui/material/styles';

// Modern Professional Healthcare Theme with Advanced Styling
const theme = createTheme({
  palette: {
    primary: {
      main: '#0066FF', // Modern vibrant blue
      light: '#4D94FF',
      dark: '#0052CC',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#00D4AA', // Modern teal/cyan
      light: '#33DDB8',
      dark: '#00A888',
      contrastText: '#FFFFFF',
    },
    error: {
      main: '#FF4757',
      light: '#FF6B7A',
      dark: '#EE5A6F',
    },
    warning: {
      main: '#FFA502',
      light: '#FFB829',
      dark: '#FF8C00',
    },
    info: {
      main: '#3742FA',
      light: '#5F68FB',
      dark: '#2C35C8',
    },
    success: {
      main: '#1DD1A1',
      light: '#48DBAE',
      dark: '#10AC84',
    },
    background: {
      default: '#F8F9FE', // Soft blue-white
      paper: '#FFFFFF',
    },
    text: {
      primary: '#1A1A2E',
      secondary: '#6B7280',
    },
    // Custom colors for modern UI
    grey: {
      50: '#F9FAFB',
      100: '#F3F4F6',
      200: '#E5E7EB',
      300: '#D1D5DB',
      400: '#9CA3AF',
      500: '#6B7280',
      600: '#4B5563',
      700: '#374151',
      800: '#1F2937',
      900: '#111827',
    },
  },
  typography: {
    fontFamily: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontSize: '3.5rem',
      fontWeight: 700,
      letterSpacing: '-0.02em',
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2.75rem',
      fontWeight: 700,
      letterSpacing: '-0.01em',
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '2.25rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
      lineHeight: 1.4,
    },
    h4: {
      fontSize: '1.875rem',
      fontWeight: 600,
      letterSpacing: '-0.005em',
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.5,
    },
    h6: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.5,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.7,
      letterSpacing: '0.00938em',
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
      letterSpacing: '0.01071em',
    },
    button: {
      fontWeight: 600,
      letterSpacing: '0.02em',
    },
  },
  shape: {
    borderRadius: 16, // More modern rounded corners
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0, 0, 0, 0.04)',
    '0px 4px 8px rgba(0, 0, 0, 0.06)',
    '0px 8px 16px rgba(0, 0, 0, 0.08)',
    '0px 12px 24px rgba(0, 0, 0, 0.10)',
    '0px 16px 32px rgba(0, 0, 0, 0.12)',
    '0px 20px 40px rgba(0, 0, 0, 0.14)',
    '0px 24px 48px rgba(0, 0, 0, 0.16)',
    '0px 28px 56px rgba(0, 0, 0, 0.18)',
    '0px 32px 64px rgba(0, 0, 0, 0.20)',
    // Glassmorphism shadows
    '0px 8px 32px rgba(0, 102, 255, 0.15)',
    '0px 12px 40px rgba(0, 212, 170, 0.15)',
    '0px 16px 48px rgba(58, 66, 250, 0.20)',
    '0px 20px 60px rgba(0, 0, 0, 0.15)',
    '0px 24px 80px rgba(0, 102, 255, 0.25)',
    '0px 28px 100px rgba(0, 0, 0, 0.20)',
    '0px 32px 120px rgba(0, 102, 255, 0.30)',
    '0px 36px 140px rgba(0, 0, 0, 0.25)',
    '0px 40px 160px rgba(0, 0, 0, 0.30)',
    '0px 44px 180px rgba(0, 0, 0, 0.35)',
    '0px 48px 200px rgba(0, 0, 0, 0.40)',
    '0px 52px 220px rgba(0, 0, 0, 0.45)',
    '0px 56px 240px rgba(0, 0, 0, 0.50)',
    '0px 60px 260px rgba(0, 0, 0, 0.55)',
    '0px 64px 280px rgba(0, 0, 0, 0.60)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          fontSize: '1rem',
          padding: '12px 24px',
          borderRadius: '12px',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          boxShadow: 'none',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0px 12px 24px rgba(0, 102, 255, 0.25)',
          },
          '&:active': {
            transform: 'translateY(0px)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #0066FF 0%, #3742FA 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #0052CC 0%, #2C35C8 100%)',
          },
        },
        outlined: {
          borderWidth: '2px',
          '&:hover': {
            borderWidth: '2px',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '20px',
          boxShadow: '0px 8px 32px rgba(0, 0, 0, 0.08)',
          border: '1px solid rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(10px)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0px 16px 48px rgba(0, 102, 255, 0.15)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '16px',
          backgroundImage: 'none',
        },
        elevation1: {
          boxShadow: '0px 4px 16px rgba(0, 0, 0, 0.06)',
        },
        elevation2: {
          boxShadow: '0px 8px 24px rgba(0, 0, 0, 0.08)',
        },
        elevation3: {
          boxShadow: '0px 12px 32px rgba(0, 0, 0, 0.10)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: '12px',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0px 4px 12px rgba(0, 102, 255, 0.1)',
            },
            '&.Mui-focused': {
              transform: 'translateY(-2px)',
              boxShadow: '0px 8px 20px rgba(0, 102, 255, 0.15)',
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: '10px',
          fontWeight: 600,
          fontSize: '0.875rem',
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          fontWeight: 500,
        },
      },
    },
  },
});

export default theme;
