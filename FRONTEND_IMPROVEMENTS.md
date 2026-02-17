# InsightCare Frontend - Improvement Roadmap

## 🔴 Critical Issues (Fix ASAP)

### 1. Authentication Security
**Current Status**: Mock authentication with no real validation
**Problem**: 
- Any password ≥6 characters allows login
- No database integration
- User data not persisted
- Signup doesn't actually create accounts

**Action Plan**:
- [ ] Wait for backend `/auth/register` and `/auth/login` endpoints
- [ ] Replace mock auth with real API calls
- [ ] Store JWT tokens securely
- [ ] Implement token refresh logic
- [ ] Add "Remember Me" functionality

**Files to Update**:
- `src/app/login/page.tsx`
- `src/app/signup/page.tsx`
- `src/app/api/auth/[...nextauth]/route.ts`

---

### 2. API Error Handling
**Current Status**: Basic error handling only
**Problem**:
- Generic error messages
- No retry mechanism
- Poor user feedback
- API calls will fail (no backend yet)

**Action Plan**:
- [ ] Add React Error Boundary
- [ ] Implement graceful degradation
- [ ] Add toast notifications
- [ ] Create error state components
- [ ] Add retry logic for failed requests

**Create New Files**:
- `src/components/common/ErrorBoundary.tsx`
- `src/components/common/Toast.tsx`
- `src/hooks/useApiError.ts`

---

### 3. Environment Configuration
**Current Status**: Missing `.env.local` file
**Problem**:
- Google OAuth won't work
- API endpoint undefined
- NextAuth secret missing

**Action Plan**:
- [x] Create `.env.local` file
- [ ] Configure Google OAuth credentials
- [ ] Add backend API URL
- [ ] Generate NextAuth secret

**Command to Generate Secret**:
```bash
openssl rand -base64 32
```

**Required `.env.local` Content**:
```env
NEXTAUTH_SECRET=<generated-secret>
NEXTAUTH_URL=http://localhost:3000
GOOGLE_CLIENT_ID=<your-google-client-id>
GOOGLE_CLIENT_SECRET=<your-google-secret>
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

---

## 🟡 Medium Priority Improvements

### 4. Enhanced Error Handling

**Install Dependencies**:
```bash
npm install react-hot-toast
```

**Create Error Hook** (`src/hooks/useApiError.ts`):
```typescript
import { toast } from 'react-hot-toast';
import { AxiosError } from 'axios';

export function useApiError() {
  const handleError = (error: unknown) => {
    if (error instanceof AxiosError) {
      const message = error.response?.data?.message || 'An error occurred';
      toast.error(message);
    } else if (error instanceof Error) {
      toast.error(error.message);
    } else {
      toast.error('An unexpected error occurred');
    }
  };

  return { handleError };
}
```

**Update API Client** (`src/lib/api/apiClient.ts`):
- Add better error categorization
- Add request retry logic
- Add timeout handling

---

### 5. Form Validation Improvements

**Current Gaps**:
- No email format validation
- No password strength indicator
- No real-time validation feedback

**Recommended Libraries**:
```bash
npm install zod @hookform/resolvers
```

**Create Validation Schema** (`src/lib/validations/auth.ts`):
```typescript
import { z } from 'zod';

export const loginSchema = z.object({
  email: z.string().email('Invalid email format'),
  password: z.string().min(6, 'Password must be at least 6 characters'),
});

export const signupSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters'),
  email: z.string().email('Invalid email format'),
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});
```

---

### 6. Loading States - Skeleton Loaders

**Install Dependency**:
```bash
npm install react-loading-skeleton
```

**Create Skeleton Components** (`src/components/common/SkeletonCard.tsx`):
```typescript
import Skeleton from 'react-loading-skeleton';
import 'react-loading-skeleton/dist/skeleton.css';

export function DiagnosisCardSkeleton() {
  return (
    <Card>
      <CardContent>
        <Skeleton height={30} width="80%" />
        <Skeleton height={20} count={3} />
      </CardContent>
    </Card>
  );
}
```

**Update Dashboard** (`src/app/dashboard/page.tsx`):
```typescript
// Replace LoadingSpinner with skeleton loaders
if (status === 'loading') {
  return (
    <Container>
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <DiagnosisCardSkeleton />
        </Grid>
        {/* Repeat for other cards */}
      </Grid>
    </Container>
  );
}
```

---

### 7. Accessibility (a11y) Improvements

**Current Gaps**:
- Missing ARIA labels
- Incomplete keyboard navigation
- No screen reader optimization

**Action Items**:
- [ ] Add `aria-label` to all icon buttons
- [ ] Add `role` attributes where needed
- [ ] Test with keyboard-only navigation
- [ ] Test with screen reader (NVDA/JAWS)
- [ ] Add focus indicators
- [ ] Ensure color contrast meets WCAG AA standards

**Example Fix** (`src/components/symptoms/SymptomForm.tsx`):
```typescript
<IconButton
  color="error"
  onClick={() => remove(index)}
  disabled={fields.length === 1}
  aria-label={`Remove symptom ${index + 1}`}  // ADD THIS
  aria-disabled={fields.length === 1}          // ADD THIS
>
  <DeleteIcon />
</IconButton>
```

---

## 🟢 Nice-to-Have Enhancements

### 8. Performance Optimization

**Lazy Loading**:
```typescript
// src/app/dashboard/page.tsx
import dynamic from 'next/dynamic';

const DiagnosisDisplay = dynamic(
  () => import('@/components/dashboard/DiagnosisDisplay'),
  { loading: () => <DiagnosisCardSkeleton /> }
);
```

**Memoization**:
```typescript
import { useMemo } from 'react';

// In DiagnosisDisplay component
const chartData = useMemo(() => 
  results.map((result, index) => ({
    name: result.disease,
    value: result.probability,
    fill: COLORS[index % COLORS.length],
  })),
  [results]
);
```

---

### 9. Testing Infrastructure

**Install Testing Libraries**:
```bash
npm install --save-dev @testing-library/react @testing-library/jest-dom @testing-library/user-event jest jest-environment-jsdom
```

**Create Jest Config** (`jest.config.js`):
```javascript
const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: './',
});

const customJestConfig = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  testEnvironment: 'jest-environment-jsdom',
};

module.exports = createJestConfig(customJestConfig);
```

**Example Test** (`src/components/symptoms/__tests__/SymptomForm.test.tsx`):
```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import SymptomForm from '../SymptomForm';

describe('SymptomForm', () => {
  it('should add new symptom field', () => {
    const mockSubmit = jest.fn();
    render(<SymptomForm onSubmit={mockSubmit} />);
    
    const addButton = screen.getByText('Add Another Symptom');
    fireEvent.click(addButton);
    
    expect(screen.getAllByText(/Symptom #/)).toHaveLength(2);
  });
});
```

---

### 10. Analytics & Monitoring

**Error Tracking with Sentry**:
```bash
npm install @sentry/nextjs
```

**Initialize Sentry** (`sentry.client.config.ts`):
```typescript
import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: 1.0,
});
```

**Google Analytics** (`src/lib/analytics.ts`):
```typescript
export const pageview = (url: string) => {
  window.gtag('config', process.env.NEXT_PUBLIC_GA_ID, {
    page_path: url,
  });
};

export const event = ({ action, category, label, value }: {
  action: string;
  category: string;
  label: string;
  value?: number;
}) => {
  window.gtag('event', action, {
    event_category: category,
    event_label: label,
    value: value,
  });
};
```

---

## 📅 Implementation Timeline

### Week 1: Critical Fixes
- Day 1-2: Environment setup + Error boundaries
- Day 3-4: Toast notifications + Error handling
- Day 5: Form validation improvements

### Week 2: UX Enhancements
- Day 1-2: Loading skeletons
- Day 3-4: Accessibility fixes
- Day 5: Code cleanup & documentation

### Week 3: Quality & Testing
- Day 1-2: Write unit tests
- Day 3-4: Integration tests
- Day 5: Performance optimization

### Week 4: Advanced Features
- Day 1-2: Analytics integration
- Day 3-4: Error monitoring (Sentry)
- Day 5: Final polish & review

---

## 🔗 Integration Points with Backend

**Once backend is ready, update these files:**

1. **Authentication**:
   - `src/app/api/auth/[...nextauth]/route.ts`
   - Replace mock authorize() with real API call
   
2. **Symptom Submission**:
   - `src/lib/api/symptomsApi.ts`
   - Update endpoint URLs
   - Handle real API responses

3. **User Profile**:
   - Create new API endpoint handler
   - Add user settings page

4. **Health History**:
   - Implement history retrieval
   - Add pagination support

---

## 📚 Additional Resources

- [Next.js Error Handling](https://nextjs.org/docs/advanced-features/error-handling)
- [React Hook Form Validation](https://react-hook-form.com/get-started#SchemaValidation)
- [Material-UI Accessibility](https://mui.com/material-ui/guides/accessibility/)
- [Testing Library Best Practices](https://testing-library.com/docs/react-testing-library/intro/)
- [Sentry Next.js Setup](https://docs.sentry.io/platforms/javascript/guides/nextjs/)

---

## ✅ Success Criteria

**Frontend is "production-ready" when:**
- [ ] All critical security issues resolved
- [ ] Comprehensive error handling implemented
- [ ] Accessibility score ≥95 (Lighthouse)
- [ ] Test coverage ≥70%
- [ ] Performance score ≥90 (Lighthouse)
- [ ] Zero console errors/warnings
- [ ] Environment variables documented
- [ ] Integration with backend API tested
- [ ] User documentation complete
- [ ] Code reviewed and approved

---

**Current Status**: ✅ Phase 1 Complete (Frontend Structure)
**Next Phase**: 🔄 Phase 2 Improvements (This Document)
**Target**: 🎯 Production-Ready Frontend by End of Month
