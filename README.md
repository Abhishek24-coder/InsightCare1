# InsightCare - Quantum AI-Powered Healthcare Platform

InsightCare is a cutting-edge healthcare application leveraging quantum AI technology to provide highly accurate, personalized disease diagnosis from symptom analysis with exceptional precision.

## Features

- 🔐 **Secure Authentication** - OAuth2 authentication via Google using NextAuth.js
- 📝 **Symptom Input Form** - Dynamic form with React Hook Form for adding multiple symptoms
- 📊 **Interactive Dashboard** - Visual diagnosis results with charts and detailed recommendations
- 🎨 **Modern UI** - Material-UI components with custom healthcare-themed design
- 📱 **Responsive Design** - Fully responsive on mobile and desktop devices
- 🔒 **Protected Routes** - Secure pages that require authentication
- ⚡ **State Management** - Redux Toolkit for global state management
- 🔄 **API Integration** - React Query for efficient data fetching and caching

## Tech Stack

- **Framework:** Next.js 14 with App Router
- **Language:** TypeScript
- **UI Library:** Material-UI (MUI)
- **Form Handling:** React Hook Form
- **State Management:** Redux Toolkit
- **Authentication:** NextAuth.js
- **API Client:** Axios with React Query
- **Charts:** MUI X-Charts
- **Styling:** Material-UI theming with healthcare color palette

## Getting Started

### Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Google OAuth credentials (for authentication)

### Installation

1. Clone the repository:
\`\`\`bash
git clone <repository-url>
cd InsightCare
\`\`\`

2. Install dependencies:
\`\`\`bash
npm install
\`\`\`

3. Set up environment variables:
   - Copy `.env.local.example` to `.env.local`
   - Add your Google OAuth credentials:
     - `GOOGLE_CLIENT_ID`
     - `GOOGLE_CLIENT_SECRET`
   - Generate a secret for `NEXTAUTH_SECRET`

4. Run the development server:
\`\`\`bash
npm run dev
\`\`\`

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

\`\`\`
InsightCare/
├── src/
│   ├── app/                    # Next.js app router pages
│   │   ├── api/               # API routes
│   │   ├── login/             # Login page
│   │   ├── symptoms/          # Symptom input page
│   │   ├── dashboard/         # Results dashboard
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Home page
│   ├── components/            # Reusable components
│   │   ├── common/           # Common UI components
│   │   ├── dashboard/        # Dashboard-specific components
│   │   ├── layout/           # Layout components
│   │   ├── providers/        # Context providers
│   │   └── symptoms/         # Symptom form components
│   ├── lib/                   # Utility libraries
│   │   └── api/              # API client and endpoints
│   ├── store/                 # Redux store
│   │   ├── slices/           # Redux slices
│   │   ├── hooks.ts          # Typed Redux hooks
│   │   └── store.ts          # Store configuration
│   └── theme/                 # MUI theme configuration
├── public/                    # Static assets
├── .env.local.example        # Environment variables template
├── next.config.mjs           # Next.js configuration
├── tsconfig.json             # TypeScript configuration
└── package.json              # Project dependencies
\`\`\`

## Key Features

### Authentication
- Google OAuth integration via NextAuth.js
- Secure session management
- Protected routes that redirect to login if not authenticated

### Symptom Input
- Dynamic form with ability to add/remove symptom entries
- Field validation with error messages
- Severity and duration selection for each symptom

### Dashboard
- Visual probability distribution with pie charts
- Detailed diagnosis cards with recommendations
- Severity indicators (low, medium, high)
- Responsive grid layout

### UI/UX
- Healthcare-themed color palette (blues, whites, greens)
- Accessible navigation and semantic HTML
- Loading states and error handling
- Mobile-responsive design

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## Configuration

### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URIs:
   - `http://localhost:3000/api/auth/callback/google` (development)
   - Your production URL + `/api/auth/callback/google`

### Environment Variables

\`\`\`env
NEXTAUTH_SECRET=your-secret-key-here
NEXTAUTH_URL=http://localhost:3000
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
NEXT_PUBLIC_API_URL=http://localhost:8000/api
\`\`\`

## Backend Integration

The frontend is designed to integrate with a backend API. Update the `NEXT_PUBLIC_API_URL` environment variable to point to your backend server.

Expected API endpoints:
- `POST /api/diagnose` - Submit symptoms and receive diagnosis
- `GET /api/history` - Retrieve user's diagnosis history

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with questions you may have regarding medical conditions.

## Support

For issues and questions, please open an issue in the GitHub repository.
