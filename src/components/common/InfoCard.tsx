'use client';

import { Card, CardContent, CardHeader, SxProps, Theme } from '@mui/material';
import { ReactNode } from 'react';

interface InfoCardProps {
  title?: string;
  children: ReactNode;
  sx?: SxProps<Theme>;
  actions?: ReactNode;
}

export default function InfoCard({ title, children, sx, actions }: InfoCardProps) {
  return (
    <Card sx={{ ...sx }}>
      {title && <CardHeader title={title} action={actions} />}
      <CardContent>{children}</CardContent>
    </Card>
  );
}
