from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
from typing import Optional, List, Dict
from rounding_check import calculations_check
from openai import OpenAI

app = FastAPI(title="AP Answer Scoring API",version="2.0.0",openapi_url="/openapi.json",docs_url="/docs")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# API Configuration
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_KEY = ""
CLAUDE_HEADERS = {
    "x-api-key": CLAUDE_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

# API Configuration for OpenAI
OPENAI_API_KEY = ""
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Define expected JSON response schema
SCORING_RESPONSE_SCHEMA = {
    "total_points": "<maximum points possible based on rubric>",
    "awarded_points": "<points the student earned on this response>",
    "score": "<student's percentage score (awarded_points / total_points) * 100>",
    "rationale_for_the_score": "<personalized explanation of the student's score highlighting strengths and areas for growth>",
    "feedback_to_the_student": [
        "<first specific suggestion to help the student improve>",
        "<second specific suggestion to help the student improve>",
        "<third specific suggestion to help the student improve>"
    ]
}

class ScoringRequest(BaseModel):
    question: str
    student_answer: str
    scoring_rubric: str

class ScoringResponse(BaseModel):
    total_points: float
    awarded_points: float
    score: float
    rationale_for_the_score: str
    feedback_to_the_student: list[str]

def create_scoring_prompt(question: str, student_answer: str, scoring_rubric: str) -> str:
    """Create a detailed prompt for Claude to score AP answers"""
    prompt = f"""You are an encouraging AP teacher providing personalized feedback to help students improve. Your role is to evaluate their work and offer constructive guidance. When writing feedback, imagine having a supportive one-on-one conversation with the student.

Task Description:
1. Review the student's work with an encouraging mindset
2. Calculate points earned and total possible points
3. Determine the percentage score
4. Explain the score in a way that recognizes strengths while gently pointing out areas for growth
5. Provide specific, actionable suggestions for improvement

Components to Evaluate:
Question Answered: {question}
Student's Response: {student_answer}
Scoring Criteria: {scoring_rubric}

If no scoring rubric is provided, evaluate the response based on:
1. The question's complexity (setting appropriate total points: 3-5 for basic questions, 5-10 for complex ones)
2. These key aspects of the student's answer:
   - How thoroughly the question was addressed
   - The depth of understanding demonstrated
   - Effective use of course concepts and terminology
   - Clarity and strength of reasoning
   - Organization of thoughts

CRITICAL INSTRUCTIONS:
1. Respond with a JSON object containing these specific fields:
   - total_points: The maximum points possible for this question
   - awarded_points: The points earned by the student's response
   - score: The percentage score (awarded_points / total_points) * 100
   - rationale_for_the_score: A personalized 2-3 line explanation of the score, highlighting what the student did well and where they can grow
   - feedback_to_the_student: Three specific suggestions written encouragingly (e.g., "Consider including more...")

IMPORTANT ADDITIONAL INSTRUCTION:
If the student's answer is similar to the reference text or is a direct copy-paste of the reference text, you must award zero points in awarded_points and score and explain the reason in the rationale. The total points would be according to the total number of parts and information required by the question, etc.

2. Format Requirements:
   - Use this exact JSON format:
{json.dumps(SCORING_RESPONSE_SCHEMA, indent=2)}
   - Keep all feedback within the JSON structure
   - Avoid any additional formatting

Note: All feedback should be:
- Personalized to the student's response
- Specific and actionable
- Encouraging and constructive
- Focused on improvement
- Relevant to AP course standards

PENALTY WARNING: Any response not in the exact JSON format specified above will result in a penalty. Make sure all feedback is specific, actionable, and directly related to the AP course standards.
"""
    return prompt

async def call_claude_api(messages: list) -> Optional[dict]:
    """Make API call to Claude with error handling and validation"""
    try:
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 8192,
            "temperature": 0,
            "messages": messages
        }
        
        response = requests.post(CLAUDE_API_URL, headers=CLAUDE_HEADERS, json=payload)
        
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Claude API call failed: {response.text}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Claude API: {str(e)}"
        )


def call_openai_api(messages: List[Dict[str, str]]) -> str:
    """
    Call the OpenAI API using the provided messages.
    Uses model "o1" with a high reasoning effort.
    Adds a retry mechanism: if a request fails, wait 4 seconds and retry, up to 3 times.
    """
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model="o1",
                messages=messages,
                reasoning_effort="medium"
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(4)
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error calling OpenAI API: {str(e)}"
                )

@app.post("/score/", response_model=ScoringResponse)
async def score_answer(request: ScoringRequest):
    """
    Score an AP answer based on the provided question, student answer, and scoring rubric.
    
    Returns:
        ScoringResponse: JSON object containing total points, awarded points, score, rationale, and feedback
    """
    try:
        # Input validation
        if not all([request.question, request.student_answer, request.scoring_rubric]):
            raise HTTPException(
                status_code=400,
                detail="All fields (question, student_answer, scoring_rubric) are required"
            )

        calculationResult = calculations_check(request.model_dump_json())

        if calculationResult.requires_calculation:
            request.scoring_rubric += f'''\nNote: {calculationResult.reasoning}'''

        # Create prompt
        prompt = create_scoring_prompt(
            request.question,
            request.student_answer,
            request.scoring_rubric
        )

        # Prepare messages for Claude
        messages = [{"role": "user", "content": prompt}]
        # Call openai API
        response = call_openai_api(messages)

        if not response:
            raise HTTPException(
                status_code=500,
                detail="Failed to get response from Claude API"
            )

        # Parse JSON response
        try:
            scoring_data = json.loads(response)
            return ScoringResponse(**scoring_data)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response from Claude: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AP Scoring API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
