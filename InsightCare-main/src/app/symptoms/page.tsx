'use client';

import { useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Container, Typography, Box } from '@mui/material';
import { useMutation } from '@tanstack/react-query';
import SymptomForm, { SymptomFormData } from '@/components/symptoms/SymptomForm';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import ErrorMessage from '@/components/common/ErrorMessage';
import { symptomsApi } from '@/lib/api/symptomsApi';
import { useAppDispatch } from '@/store/hooks';
import { setSymptoms } from '@/store/slices/symptomSlice';

export default function SymptomsPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const dispatch = useAppDispatch();

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login');
    }
  }, [status, router]);

  const { mutate, isPending, isError, error } = useMutation({
    mutationFn: symptomsApi.submitSymptoms,
    onSuccess: () => {
      // Navigate to analyzing page
      router.push('/analyzing');
    },
  });

  const handleSubmit = (data: SymptomFormData) => {
    const symptomsWithIds = data.symptoms.map((symptom, index) => ({
      id: `symptom-${Date.now()}-${index}`,
      ...symptom,
    }));

    // Store in Redux
    dispatch(setSymptoms(symptomsWithIds));

    // Submit to API
    mutate({ symptoms: symptomsWithIds });
  };

  if (status === 'loading') {
    return (
      <Container>
        <LoadingSpinner message="Loading..." />
      </Container>
    );
  }

  if (status === 'unauthenticated' || !session) {
    return (
      <Container>
        <Box sx={{ py: 8, textAlign: 'center' }}>
          <Typography variant="h5" gutterBottom>
            Redirecting to login...
          </Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Symptom Analysis
        </Typography>
        
        <Typography variant="body1" color="text.secondary" paragraph>
          Please describe your symptoms in detail. The more accurate information you provide,
          the better our analysis will be.
        </Typography>

        {isError && (
          <ErrorMessage
            title="Submission Failed"
            message={error instanceof Error ? error.message : 'Failed to submit symptoms. Please try again.'}
          />
        )}

        <Box sx={{ mt: 4 }}>
          <SymptomForm onSubmit={handleSubmit} isLoading={isPending} />
        </Box>
      </Box>
    </Container>
  );
}
