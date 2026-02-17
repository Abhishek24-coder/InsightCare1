# InsightCare Backend API

AI-powered disease diagnosis system backend built with FastAPI and PostgreSQL.

## Features

- ✅ User authentication with JWT tokens
- ✅ Password hashing with bcrypt
- ✅ AI-powered symptom analysis (mock implementation for Phase 2)
- ✅ Diagnosis history tracking
- ✅ RESTful API with auto-generated docs
- ✅ PostgreSQL database with SQLAlchemy ORM
- ✅ CORS configured for frontend integration

## Tech Stack

- **FastAPI 0.109.0** - Modern Python web framework
- **PostgreSQL** - Primary database
- **SQLAlchemy** - ORM
- **JWT** - Token-based authentication
- **bcrypt** - Password hashing
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- PostgreSQL 12+ installed and running
- Git

### 2. Install PostgreSQL

**Windows:**
Download from https://www.postgresql.org/download/windows/

**Mac:**
```bash
brew install postgresql
brew services start postgresql
```

**Linux:**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### 3. Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database and user
CREATE DATABASE insightcare;
CREATE USER insightcare_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE insightcare TO insightcare_user;
\q
```

### 4. Clone and Setup Project

```bash
# Navigate to project directory
cd insightcare-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your credentials
# Update DATABASE_URL with your PostgreSQL credentials
# Update SECRET_KEY with a secure random string
```

Example `.env`:
```env
DATABASE_URL=postgresql://insightcare_user:your_password@localhost:5432/insightcare
SECRET_KEY=your-super-secret-jwt-key-min-32-chars-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=1440
ALGORITHM=HS256
ENVIRONMENT=development
FRONTEND_URL=http://localhost:3000
```

### 6. Run the Application

```bash
# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## API Endpoints

### Authentication

- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get JWT token
- `GET /api/auth/me` - Get current user profile (requires auth)

### Diagnosis

- `POST /api/diagnosis/analyze` - Analyze symptoms and get predictions (requires auth)
- `GET /api/diagnosis/history` - Get diagnosis history (requires auth)
- `GET /api/diagnosis/{diagnosis_id}` - Get specific diagnosis details (requires auth)

### Health

- `GET /api/health` - Check API and database status

## Testing the API

### Using Swagger UI

1. Open http://localhost:8000/docs
2. Test endpoints interactively
3. For protected endpoints:
   - First call `/api/auth/register` or `/api/auth/login`
   - Copy the `access_token` from response
   - Click "Authorize" button at top
   - Enter: `Bearer <your_token>`
   - Now you can test protected endpoints

### Using curl

```bash
# Register user
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "password": "password123"
  }'

# Login
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "password123"
  }'

# Analyze symptoms (use token from login)
curl -X POST "http://localhost:8000/api/diagnosis/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_token>" \
  -d '{
    "symptoms": ["fever", "cough", "headache"],
    "severity": "moderate",
    "duration": "2 days"
  }'
```

## Project Structure

```
insightcare-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Configuration
│   ├── database.py          # Database setup
│   ├── models/              # SQLAlchemy models
│   │   ├── user.py
│   │   └── diagnosis.py
│   ├── schemas/             # Pydantic schemas
│   │   ├── user_schema.py
│   │   └── diagnosis_schema.py
│   ├── api/                 # API endpoints
│   │   ├── auth.py
│   │   ├── diagnosis.py
│   │   └── health.py
│   ├── services/            # Business logic
│   │   ├── auth_service.py
│   │   └── diagnosis_service.py
│   └── utils/               # Utilities
│       ├── security.py
│       └── dependencies.py
├── requirements.txt
├── .env.example
├── .env                     # Your local config (not in git)
└── README.md
```

## Frontend Integration

The backend is configured to work with the Next.js frontend:

1. Frontend should run on `http://localhost:3000`
2. API base URL: `http://localhost:8000/api`
3. Authentication: Include JWT token in `Authorization: Bearer <token>` header
4. CORS is pre-configured for the frontend

Update frontend's `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Database Schema

### Users Table
- `id` (UUID, primary key)
- `name` (string, required)
- `email` (string, unique, required)
- `password_hash` (string, required)
- `created_at` (timestamp)
- `updated_at` (timestamp)
- `last_login` (timestamp, nullable)
- `is_active` (boolean, default true)

### Diagnoses Table
- `id` (UUID, primary key)
- `user_id` (UUID, foreign key to users)
- `symptoms` (text array)
- `severity` (string, nullable)
- `duration` (string, nullable)
- `predictions` (JSONB)
- `created_at` (timestamp)

## Development Notes

### Current Implementation (Phase 2)

- Mock AI diagnosis using rule-based matching
- Returns top 5 predictions based on symptom keywords
- Full CRUD operations for users and diagnoses

### Future Enhancements (Phase 3)

- Real ML/AI model integration
- Advanced error handling and logging
- Rate limiting
- Database migrations with Alembic
- Unit and integration tests
- Performance optimization
- Caching layer

## Security Features

✅ Password hashing with bcrypt (salt rounds >= 10)  
✅ JWT token authentication (24-hour expiry)  
✅ Input validation with Pydantic  
✅ SQL injection protection with SQLAlchemy ORM  
✅ CORS configuration  
✅ Environment variable management  

## Troubleshooting

### Database Connection Error

If you see "could not connect to server":
1. Check PostgreSQL is running: `pg_ctl status` or `brew services list`
2. Verify credentials in `.env`
3. Ensure database exists: `psql -U postgres -l`

### Import Errors

If you get import errors after installing packages:
1. Ensure virtual environment is activated
2. Reinstall requirements: `pip install -r requirements.txt --upgrade`
3. Check Python version: `python --version` (should be 3.10+)

### Port Already in Use

If port 8000 is busy:
```bash
# Use a different port
uvicorn app.main:app --reload --port 8001
```

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test thoroughly
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Create Pull Request

## License

Part of the InsightCare project by Team InsightCare.

## Contact

- **Frontend Lead:** Yash Sharma
- **Backend Lead:** Lavlesh Yadav
- **CTO:** Harsh Parmar
- **AI Research:** Abhishek Shrivastava

---

**Happy coding! 🚀**
