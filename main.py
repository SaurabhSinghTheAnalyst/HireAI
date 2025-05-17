import pandas as pd
import os
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_resumes():
    """Load the resume data"""
    try:
        df = pd.read_csv('Data/small_CV_DB.csv')
        return df
    except Exception as e:
        print(f"Error loading resume data: {str(e)}")
        return pd.DataFrame()

def get_candidate_match_score(job_query, candidate_profile):
    """
    Query the LLM to get a match score between job requirements and candidate profile
    """
    prompt = f"""
You are PeopleGPT, an advanced AI hiring copilot for recruiters.

# Recruiter Query
{job_query}

# Candidate Profile
{candidate_profile}

# Instructions
1. Parse the recruiter query for: role, seniority, required skills, location/country/region/Continent (if specified), work arrangement, and any other filters.
2. If a specific continent, country or region is mentioned in the query, only include candidates whose current location matches that country or region. If no location is specified, consider all candidates.
3. Analyze the candidate profile for matching skills, experience, location, and preferences.
4. Score the candidate's fit for the query on a scale of 0-100.
5. Extract and list the candidate's key technical and professional skills (up to 8 most relevant).
6. Provide a concise explanation for the score, referencing specific matches or gaps.

# Response Format
Score: [0-100]
Skills: [comma-separated list]
Explanation: [1-2 sentences]
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are PeopleGPT, an advanced AI hiring copilot for recruiters."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        ).choices[0].message.content
        
        # Robust extraction with fallbacks
        score = 0
        skills = ""
        explanation = "Unable to extract information"
        for line in response.splitlines():
            if line.startswith("Score:"):
                try:
                    score_text = line.split("Score:")[1].strip()
                    # Handle score ranges like "80-85" by taking the average
                    if "-" in score_text:
                        parts = score_text.split("-")
                        score = (int(parts[0]) + int(parts[1])) // 2
                    else:
                        score = int(score_text)
                except ValueError:
                    # If we can't convert to int, set a default
                    score = 0
            elif line.startswith("Skills:"):
                skills = line.split("Skills:")[1].strip()
            elif line.startswith("Explanation:"):
                explanation = line.split("Explanation:")[1].strip()
        return score, skills, explanation
    except Exception as e:
        print(f"Error in LLM processing: {str(e)}")
        return 0, "", f"Error in LLM processing: {str(e)}"

def extract_skills_from_resume(resume_text):
    """Extract skills from resume text using LLM"""
    prompt = f"""
Extract only the technical and professional skills from this resume. 
Return them as a comma-separated list.

Resume:
{resume_text}

Skills:
"""
    try:
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are a precise skills extraction system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        ).choices[0].message.content.strip()
        return response
    except Exception as e:
        return f"Error extracting skills: {str(e)}"

def extract_location_from_query(query, country_list):
    """Extract location from query using LLM"""
    prompt = f"""
Extract the location (country, region, or continent) from this query if specified.
If no location is specified, return null.

Query: {query}

Available locations: {', '.join(country_list)}

Return only the location name or null if none specified.
"""
    try:
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are a location extraction system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        ).choices[0].message.content.strip()
        
        # Clean up response
        response = response.lower().strip()
        if response == "null" or response == "none" or response == "no location specified":
            return None
            
        # Check if the extracted location is in our country list
        for country in country_list:
            if country.lower() in response:
                return country
                
        return None
    except Exception as e:
        print(f"Error extracting location: {str(e)}")
        return None

def get_experience_years(resume_text):
    """Estimate years of experience from resume text"""
    prompt = f"""
Estimate the total years of professional experience from this resume.
Return only a number.

Resume:
{resume_text}

Years of experience:
"""
    try:
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are an experience estimation system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        ).choices[0].message.content.strip()
        
        # Try to extract a number from the response
        try:
            return int(response)
        except ValueError:
            return 0
    except Exception as e:
        print(f"Error estimating experience: {str(e)}")
        return 0