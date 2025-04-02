# AP Answer Grader API Documentation

## Overview
The AP Answer Grader API provides automated scoring and feedback for AP-style answers using AI. The API evaluates student responses based on provided questions and rubrics, returning detailed scoring information and improvement suggestions.

Live API: `https://api-ap-grader.onrender.com`  
Streamlit App: `https://st-ap-grader.onrender.com`  
HTML Demo Interface: Download the `html-ap-grader-demo.html` file to try out the API locally

## Quick Start

### Using the HTML Demo
1. Download `html-ap-grader-demo.html`
2. Open in a web browser
3. Enter the question, student answer, and rubric
4. Click "Evaluate Answer" to see results

### API Endpoints

#### 1. Score Answer
```
POST /score/
```
Evaluates a student's AP answer and provides scoring feedback.

##### Request Body
```json
{
    "question": "Using your knowledge of United States history...",
    "student_answer": "The Montgomery Bus Boycott...",
    "scoring_rubric": "Part A (2 points): 1 point for..."
}
```

##### Response
```json
{
    "total_points": 5.0,
    "awarded_points": 4.5,
    "score": 90.0,
    "rationale_for_the_score": "Strong analysis of the Civil Rights Movement...",
    "feedback_to_the_student": [
        "Include more specific examples from the time period",
        "Strengthen the connection between events",
        "Expand on the long-term impact"
    ]
}
```

Response fields:
- `total_points`: Maximum points possible based on the rubric
- `awarded_points`: Points awarded to the student's answer
- `score`: Percentage score calculated as (awarded_points / total_points) * 100
- `rationale_for_the_score`: Detailed explanation of the scoring
- `feedback_to_the_student`: Three specific improvement suggestions

#### 2. Health Check
```
GET /health
```
Returns API status information.

##### Response
```json
{
    "status": "healthy",
    "message": "AP Scoring API is running"
}
```

## Error Responses
- `400`: Missing or invalid input
- `500`: Internal server error or AI service error

## Example Usage

### Python
```python
import requests

api_url = "https://api-ap-grader.onrender.com/score"
data = {
    "question": "Analyze the causes of the Civil War...",
    "student_answer": "The Civil War was caused by...",
    "scoring_rubric": "1 point for identifying economic factors..."
}

response = requests.post(api_url, json=data)
result = response.json()

print(f"Total Points: {result['total_points']}")
print(f"Awarded Points: {result['awarded_points']}")
print(f"Score: {result['score']}%")
print(f"Rationale: {result['rationale_for_the_score']}")
print("Feedback:", *result['feedback_to_the_student'], sep="\n- ")
```

### cURL
```bash
curl -X POST https://ap-grader-api.example.com/score \
     -H "Content-Type: application/json" \
     -d '{
           "question": "Analyze...",
           "student_answer": "The Civil War...",
           "scoring_rubric": "1 point for..."
         }'
```
