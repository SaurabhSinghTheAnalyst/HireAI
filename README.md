# Hiring Wizard Bot

An AI-powered hiring assistant that helps recruiters find and reach out to candidates.

## Features

- Candidate search and matching
- Resume analysis
- Skills extraction
- AI-powered personalized outreach messages

## Deployment on Hugging Face Spaces

### Backend Deployment

1. Create a new Space on Hugging Face:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Docker" as the SDK
   - Name your space (e.g., "hiring-wizard-bot")

2. Configure the Space:
   - Set the following environment variables in your Space settings:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

3. Push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://huggingface.co/spaces/your-username/your-space-name
   git push -u origin main
   ```

### Frontend Deployment

1. Build the frontend:
   ```bash
   cd hiring-wizard-bot
   npm run build
   ```

2. Deploy to a static hosting service (e.g., Vercel, Netlify) or include it in your Hugging Face Space.

## Local Development

### Backend

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   uvicorn api:app --reload
   ```

### Frontend

1. Install dependencies:
   ```bash
   cd hiring-wizard-bot
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
```

## License

MIT 