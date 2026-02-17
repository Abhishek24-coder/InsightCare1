'use client';

import { Alert, AlertTitle, Box } from '@mui/material';

interface ErrorMessageProps {
  message: string;
  title?: string;
}

export default function ErrorMessage({ message, title = 'Error' }: ErrorMessageProps) {
  return (
    <Box sx={{ my: 2 }}>
      <Alert severity="error">
        <AlertTitle>{title}</AlertTitle>
        {message}
      </Alert>
    </Box>
  );
}
