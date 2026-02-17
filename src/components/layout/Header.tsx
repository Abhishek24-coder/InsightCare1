'use client';

import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  useTheme,
  alpha,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  useMediaQuery,
} from '@mui/material';
import { useSession, signOut } from 'next-auth/react';
import { useRouter, usePathname } from 'next/navigation';
import { useState } from 'react';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import PersonIcon from '@mui/icons-material/Person';
import DashboardIcon from '@mui/icons-material/Dashboard';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import LogoutIcon from '@mui/icons-material/Logout';
import HomeIcon from '@mui/icons-material/Home';
import InfoIcon from '@mui/icons-material/Info';
import ContactSupportIcon from '@mui/icons-material/ContactSupport';
import FeaturedPlayListIcon from '@mui/icons-material/FeaturedPlayList';

export default function Header() {
  const { data: session } = useSession();
  const router = useRouter();
  const pathname = usePathname();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    handleProfileMenuClose();
    await signOut({ redirect: false });
    router.push('/');
  };

  const handleNavigation = (path: string) => {
    router.push(path);
    setMobileOpen(false);
  };

  // Navigation items for non-authenticated users
  const publicNavItems = [
    { label: 'Home', path: '/', icon: <HomeIcon /> },
    { label: 'Features', path: '/#features', icon: <FeaturedPlayListIcon /> },
    { label: 'How It Works', path: '/#how-it-works', icon: <InfoIcon /> },
    { label: 'FAQ', path: '/#faq', icon: <ContactSupportIcon /> },
  ];

  // Navigation items for authenticated users
  const authNavItems = [
    { label: 'Home', path: '/', icon: <HomeIcon /> },
    { label: 'Symptoms', path: '/symptoms', icon: <MedicalServicesIcon /> },
    { label: 'Dashboard', path: '/dashboard', icon: <DashboardIcon /> },
  ];

  const navItems = session ? authNavItems : publicNavItems;

  const isActivePath = (path: string) => {
    if (path === '/') return pathname === '/';
    return pathname?.startsWith(path);
  };

  // Mobile Drawer
  const drawer = (
    <Box sx={{ width: 280, height: '100%', bgcolor: 'background.paper' }}>
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <LocalHospitalIcon sx={{ fontSize: 32, color: 'primary.main' }} />
          <Typography variant="h6" fontWeight={700}>
            InsightCare
          </Typography>
        </Box>
        <IconButton onClick={handleDrawerToggle}>
          <CloseIcon />
        </IconButton>
      </Box>
      <Divider />
      <List>
        {navItems.map((item) => (
          <ListItem key={item.label} disablePadding>
            <ListItemButton
              onClick={() => handleNavigation(item.path)}
              selected={isActivePath(item.path)}
              sx={{
                py: 1.5,
                '&.Mui-selected': {
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  borderRight: `3px solid ${theme.palette.primary.main}`,
                },
              }}
            >
              <Box sx={{ mr: 2, display: 'flex', color: 'primary.main' }}>
                {item.icon}
              </Box>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider sx={{ my: 2 }} />
      {session ? (
        <Box sx={{ px: 2 }}>
          <Button
            fullWidth
            variant="outlined"
            color="error"
            startIcon={<LogoutIcon />}
            onClick={handleLogout}
            sx={{ mb: 2 }}
          >
            Logout
          </Button>
        </Box>
      ) : (
        <Box sx={{ px: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Button
            fullWidth
            variant="outlined"
            onClick={() => handleNavigation('/login')}
          >
            Sign In
          </Button>
          <Button
            fullWidth
            variant="contained"
            onClick={() => handleNavigation('/signup')}
          >
            Sign Up
          </Button>
        </Box>
      )}
    </Box>
  );

  return (
    <>
      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          bgcolor: alpha(theme.palette.background.paper, 0.95),
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          color: 'text.primary',
        }}
      >
        <Container maxWidth="lg">
          <Toolbar disableGutters sx={{ height: 72 }}>
            {/* Logo */}
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                cursor: 'pointer',
                mr: 4,
              }}
              onClick={() => handleNavigation('/')}
            >
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
                  fontSize: '1.25rem',
                  background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  display: { xs: 'none', sm: 'block' },
                }}
              >
                InsightCare
              </Typography>
            </Box>

            {/* Desktop Navigation */}
            {!isMobile && (
              <Box sx={{ display: 'flex', gap: 1, flexGrow: 1 }}>
                {navItems.map((item) => (
                  <Button
                    key={item.label}
                    onClick={() => handleNavigation(item.path)}
                    sx={{
                      color: isActivePath(item.path) ? 'primary.main' : 'text.primary',
                      fontWeight: isActivePath(item.path) ? 700 : 600,
                      fontSize: '0.9375rem',
                      px: 2,
                      position: 'relative',
                      '&::after': {
                        content: '""',
                        position: 'absolute',
                        bottom: 8,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        width: isActivePath(item.path) ? '70%' : '0%',
                        height: '3px',
                        background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                        borderRadius: '2px',
                        transition: 'width 0.3s',
                      },
                      '&:hover::after': {
                        width: '70%',
                      },
                    }}
                  >
                    {item.label}
                  </Button>
                ))}
              </Box>
            )}

            <Box sx={{ flexGrow: 1, display: { xs: 'block', md: 'none' } }} />

            {/* Auth Buttons / Profile Menu */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {session ? (
                <>
                  {!isMobile && (
                    <Typography
                      variant="body2"
                      sx={{
                        color: 'text.secondary',
                        fontWeight: 600,
                      }}
                    >
                      {session.user?.name}
                    </Typography>
                  )}
                  <IconButton
                    onClick={handleProfileMenuOpen}
                    sx={{
                      border: `2px solid ${theme.palette.primary.main}`,
                      padding: '4px',
                    }}
                  >
                    <Avatar
                      sx={{
                        width: 36,
                        height: 36,
                        background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                        fontSize: '1rem',
                        fontWeight: 700,
                      }}
                    >
                      {session.user?.name?.charAt(0) || 'U'}
                    </Avatar>
                  </IconButton>
                </>
              ) : (
                <>
                  {!isMobile && (
                    <>
                      <Button
                        variant="outlined"
                        onClick={() => handleNavigation('/login')}
                        sx={{
                          borderRadius: '10px',
                          px: 3,
                          fontWeight: 600,
                        }}
                      >
                        Sign In
                      </Button>
                      <Button
                        variant="contained"
                        onClick={() => handleNavigation('/signup')}
                        sx={{
                          borderRadius: '10px',
                          px: 3,
                          fontWeight: 600,
                          background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                          boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.3)}`,
                        }}
                      >
                        Sign Up
                      </Button>
                    </>
                  )}
                </>
              )}

              {/* Mobile Menu Icon */}
              {isMobile && (
                <IconButton
                  color="inherit"
                  edge="end"
                  onClick={handleDrawerToggle}
                >
                  <MenuIcon />
                </IconButton>
              )}
            </Box>
          </Toolbar>
        </Container>
      </AppBar>

      {/* Profile Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleProfileMenuClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        PaperProps={{
          sx: {
            mt: 1.5,
            minWidth: 200,
            borderRadius: '12px',
            boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.15)}`,
          },
        }}
      >
        <Box sx={{ px: 2, py: 1.5 }}>
          <Typography variant="subtitle2" fontWeight={700}>
            {session?.user?.name}
          </Typography>
          <Typography variant="body2" color="text.secondary" fontSize="0.8125rem">
            {session?.user?.email}
          </Typography>
        </Box>
        <Divider />
        <MenuItem onClick={() => { handleProfileMenuClose(); handleNavigation('/dashboard'); }}>
          <DashboardIcon sx={{ mr: 1.5, fontSize: 20 }} />
          Dashboard
        </MenuItem>
        <MenuItem onClick={() => { handleProfileMenuClose(); handleNavigation('/symptoms'); }}>
          <MedicalServicesIcon sx={{ mr: 1.5, fontSize: 20 }} />
          Symptoms
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleLogout} sx={{ color: 'error.main' }}>
          <LogoutIcon sx={{ mr: 1.5, fontSize: 20 }} />
          Logout
        </MenuItem>
      </Menu>

      {/* Mobile Drawer */}
      <Drawer
        anchor="right"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        ModalProps={{ keepMounted: true }}
        PaperProps={{
          sx: {
            borderTopLeftRadius: '20px',
            borderBottomLeftRadius: '20px',
          },
        }}
      >
        {drawer}
      </Drawer>
    </>
  );
}
