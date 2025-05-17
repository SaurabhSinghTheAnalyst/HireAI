import pandas as pd
import os
from openai import OpenAI
from typing import Tuple, List
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Sample candidate data
SAMPLE_CANDIDATES = [
    {
        "Name": "John Smith",
        "Phone": "+1-555-0123",
        "Country": "United States",
        "Open To": "Full-time, Remote",
        "Email": "john.smith@email.com",
        "Resume": "Experienced software engineer with 5 years of experience in Python and React. Strong background in full-stack development and cloud technologies."
    },
    {
        "Name": "Maria Garcia",
        "Phone": "+44-20-1234-5678",
        "Country": "United Kingdom",
        "Open To": "Full-time, Hybrid",
        "Email": "maria.garcia@email.com",
        "Resume": "Senior developer with expertise in Java and Spring Boot. 8 years of experience in enterprise applications and microservices architecture."
    },
    {
        "Name": "Alex Chen",
        "Phone": "+86-10-1234-5678",
        "Country": "China",
        "Open To": "Full-time, On-site",
        "Email": "alex.chen@email.com",
        "Resume": "Full-stack developer with 3 years of experience in JavaScript, Node.js, and React. Passionate about building scalable web applications."
    }
]

def load_resumes() -> pd.DataFrame:
    """Load candidate resumes from the database"""
    try:
        # Get the absolute path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        csv_path = os.path.join(parent_dir, 'Data', 'small_CV_DB.csv')
        
        # Load the CSV file
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading resumes: {str(e)}")
        # Fallback to sample data if CSV loading fails
        return pd.DataFrame(SAMPLE_CANDIDATES)

def get_candidate_match_score(query: str, candidate_profile: str) -> Tuple[int, str, str]:
    """Get match score between job requirements and candidate profile"""
    try:
        # Create the prompt for the LLM
        prompt = f"""
        Analyze the match between the job requirements and candidate profile.
        
        Job Requirements:
        {query}
        
        Candidate Profile:
        {candidate_profile}
        
        Provide:
        1. A match score (0-100)
        2. Key skills that match
        3. A brief explanation of the match
        
        Format the response as JSON:
        {{
            "score": <score>,
            "skills": "<comma-separated skills>",
            "explanation": "<explanation>"
        }}
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a hiring assistant that analyzes job-candidate matches."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return result["score"], result["skills"], result["explanation"]
        
    except Exception as e:
        print(f"Error in get_candidate_match_score: {str(e)}")
        return 0, "", "Error processing match"

def extract_skills_from_resume(resume: str) -> str:
    """Extract skills from resume text"""
    try:
        prompt = f"""
        Extract technical skills from this resume:
        {resume}
        
        Return only a comma-separated list of skills.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a hiring assistant that extracts skills from resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in extract_skills_from_resume: {str(e)}")
        return ""

def extract_location_from_query(query: str, country_list: List[str]) -> str:
    """Extract location from query"""
    try:
        prompt = f"""
        Extract the location from this query:
        {query}
        
        Available locations: {', '.join(country_list)}
        
        Return only the country name if found, or 'null' if no location is specified.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a hiring assistant that extracts locations from queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        location = response.choices[0].message.content.strip()
        return location if location.lower() != 'null' else None
        
    except Exception as e:
        print(f"Error in extract_location_from_query: {str(e)}")
        return None

def get_experience_years(resume: str) -> str:
    """Get estimated years of experience from resume"""
    try:
        prompt = f"""
        Estimate the years of experience from this resume:
        {resume}
        
        Return only the number of years as a string.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a hiring assistant that estimates experience from resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in get_experience_years: {str(e)}")
        return "0"