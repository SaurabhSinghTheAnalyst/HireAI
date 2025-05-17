from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
import pandas as pd
from typing import Optional, List, Dict, Any
from collections import Counter
import warnings
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add parent directory to path to import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hiring Wizard API",
    description="API for AI-powered hiring copilot",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Pydantic models for request/response
class JobQuery(BaseModel):
    query: str
    candidate_profile: Optional[str] = None

class CandidateProfile(BaseModel):
    name: str
    resume: str
    location: Optional[str] = None
    work_options: Optional[str] = None

class MatchResponse(BaseModel):
    score: int
    skills: str
    explanation: str

class CandidateResponse(BaseModel):
    name: str
    phone: str
    country: str
    open_to: str
    email: str
    resume: str
    score: int
    skills: str
    explanation: str

class SkillsResponse(BaseModel):
    skills: str

class LocationResponse(BaseModel):
    location: Optional[str]

class SearchQuery(BaseModel):
    query: str

class OutreachRequest(BaseModel):
    candidateEmail: str
    subject: str
    message: str
    candidateName: str
    candidateResume: str

class OutreachResponse(BaseModel):
    generatedMessage: str

# API Routes
@app.post("/api/match", response_model=MatchResponse)
async def get_candidate_match(query: JobQuery):
    """Get match score between job requirements and candidate profile"""
    try:
        score, skills, explanation = main.get_candidate_match_score(
            query.query,
            query.candidate_profile or ""
        )
        return MatchResponse(
            score=score,
            skills=skills,
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/skills", response_model=SkillsResponse)
async def extract_skills(resume: str = Query(..., description="Resume text to extract skills from")):
    """Extract skills from resume text"""
    try:
        skills = main.extract_skills_from_resume(resume)
        return SkillsResponse(skills=skills)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/candidates", response_model=List[CandidateResponse])
async def get_candidates():
    """Get all candidates from the database"""
    try:
        logger.info("Loading resumes from database...")
        df = main.load_resumes()
        logger.info(f"Loaded {len(df)} candidates from database")
        
        if df.empty:
            logger.info("No candidates found in database")
            return []
        
        # Process each candidate
        candidates = []
        for _, row in df.iterrows():
            # Extract skills from resume
            resume = row.get('Resume', '')
            skills = main.extract_skills_from_resume(resume)
            
            candidate = CandidateResponse(
                name=row.get('Name', ''),
                phone=row.get('Phone', ''),
                country=row.get('Country', ''),
                open_to=row.get('Open To', ''),
                email=row.get('Email', ''),
                resume=resume,
                score=0,  # Will be calculated when matched
                skills=skills,  # Use extracted skills
                explanation=''  # Will be generated when matched
            )
            candidates.append(candidate)
            logger.info(f"Processed candidate: {candidate.name}")
        
        logger.info(f"Returning {len(candidates)} candidates")
        return candidates
    except Exception as e:
        logger.error(f"Error in get_candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/location", response_model=LocationResponse)
async def extract_location(query: str = Query(..., description="Query to extract location from")):
    """Extract location from query"""
    try:
        df = main.load_resumes()
        country_list = df['Country'].dropna().unique().tolist()
        location = main.extract_location_from_query(query, country_list)
        return LocationResponse(location=location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/experience")
async def get_experience_years(resume: str = Query(..., description="Resume text to analyze")):
    """Get estimated years of experience from resume"""
    try:
        # This function is defined in main.py
        years = main.get_experience_years(resume)
        return {"experience": years}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_candidates(query: SearchQuery):
    try:
        logger.info(f"Received search query: {query.query}")
        candidates = main.search_candidates(query.query)
        logger.info(f"Found {len(candidates)} candidates")
        return candidates
    except Exception as e:
        logger.error(f"Error in search_candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/outreach", response_model=OutreachResponse)
async def generate_outreach(request: OutreachRequest):
    try:
        logger.info(f"Generating outreach message for {request.candidateName}")
        logger.info(f"Request data: {request.dict()}")
        
        # Extract key information from the resume
        try:
            skills = main.extract_skills_from_resume(request.candidateResume)
            logger.info(f"Extracted skills: {skills}")
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            skills = "Skills extraction failed"

        try:
            experience_years = main.get_experience_years(request.candidateResume)
            logger.info(f"Extracted experience years: {experience_years}")
        except Exception as e:
            logger.error(f"Error extracting experience years: {str(e)}")
            experience_years = "Experience extraction failed"
        
        # Create a more detailed prompt for the LLM
        prompt = f"""As a professional recruiter, write a personalized outreach email to {request.candidateName}.

Candidate Information:
- Name: {request.candidateName}
- Skills: {skills}
- Years of Experience: {experience_years}
- Resume: {request.candidateResume}

Recruiter's Initial Message:
{request.message}

Write a professional, personalized email that:
1. Opens with a personalized greeting
2. References specific skills and experience from their resume that match the opportunity
3. Highlights their relevant achievements or experience
4. Maintains a professional yet conversational tone
5. Incorporates the recruiter's message naturally
6. Includes a clear call to action
7. Ends with a professional sign-off

Format the response as a complete email with proper greeting and sign-off. Make it sound natural and personalized, not template-like."""

        logger.info("Sending request to OpenAI")
        # Generate message using OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional recruiter writing personalized outreach emails. Your goal is to write engaging, personalized emails that show you've reviewed the candidate's background and are genuinely interested in their profile."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        generated_message = response.choices[0].message.content
        logger.info(f"Successfully generated outreach message for {request.candidateName}")
        return OutreachResponse(generatedMessage=generated_message)

    except Exception as e:
        logger.error(f"Error generating outreach: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 