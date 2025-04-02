from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import base64
from typing import Optional, List, Dict
import time
import asyncio
from openai import OpenAI
import re

app = FastAPI(
    title="AP FRQ Grading API",
    version="2.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# API Configuration for Claude
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

# API endpoints for Language and Literature
LANG_ENDPOINT = "https://api-ap-lang-essay-grader.onrender.com/grade"
LIT_ENDPOINT = "https://api-ap-lit-essay-grader.onrender.com/grade"

# Define the final response model that includes both API responses.
class GradingResponse(BaseModel):
    scoring_response: str
    feedback_response: str
    html_for_app: str
    total_possible_points: float
    total_earned_points: float

def create_classification_prompt(course_title: str, question: str) -> str:
    """
    Create a prompt to classify the course of an FRQ based on the course title and question.
    """
    prompt = f"""####
You are an expert classifier for AP (Advanced Placement) Free Response Questions (FRQs). Your task is to determine the specific AP course that this FRQ belongs to based on the provided course title and the FRQ question content.

The provided course title is:
<course_title>
{course_title}
</course_title>

The FRQ question is:
<question>
{question}
</question>

Instructions:
1. First, examine the provided course title. If it clearly matches one of the acceptable AP course names listed below, use that as your classification.
2. If the course title is unclear, vague, or missing, analyze the FRQ question content to determine which AP course it most likely belongs to.
3. Consider the vocabulary, concepts, task verbs, and subject matter in the FRQ question.
4. Output your classification in the specified JSON format.

Acceptable AP course names:
- AP United States History
- AP World History
- AP Human Geography
- AP United States Government
- AP Microeconomics
- AP Environmental Science
- AP Biology
- AP Chemistry
- AP Physics
- AP Literature
- AP Language

Your response must be in the following JSON format and nothing else:
{{
  "course_of_the_frq": "[The exact AP course name from the list above]"
}}
"""
    return prompt

def create_claude_prompt(question: str, student_answer: str, scoring_rubric: str, image_data: str) -> str:
    """
    Create the first prompt (Prompt 1) for Claude using the provided question,
    student's answer, rubric, and image data (as base64 strings if provided).
    """
    prompt = f"""####
You are an expert evaluator for AP Environmental Science Free Response Questions (FRQs). Your task is to evaluate a student's response to an AP Environmental Science FRQ based on a question-specific rubric. You must be very strict in your scoring, awarding points only when the student's response closely matches the rubric criteria.
Here is the AP Environmental Science FRQ question:
<question>
{question}
</question>
Associated image or diagram (if provided):
<image>
{image_data}
</image>
Here is the question-specific scoring rubric:
<rubric>
{scoring_rubric}
</rubric>
Here is the student's response:
<response>
{student_answer}
</response>
Instructions:
1. Carefully read the question, rubric, and student response.
2. Conduct a detailed analysis of the response. Wrap your detailed analysis inside <response_breakdown> tags. In your analysis:
a. Identify and list all task verbs from the question, numbering them.
b. Break down the rubric into individual criteria.
c. For each rubric criterion:
- Quote the exact wording from the rubric.
- Paraphrase the criterion in simpler terms.
- Consider and list alternative phrasings or synonyms that could be accepted.
- Quote relevant parts of the student's response.
- Analyze how well the student's response meets the criterion.
- Explicitly state whether the criterion is met or not met, and why.
-No partial credit. Only the first response is considered if multiple answers are given.
- Provide a preliminary point estimate for the criterion.
- Note points earned or missed.
d. Note strengths and weaknesses of the response.
e. Perform any necessary calculations or data interpretations.
3. After your analysis, provide your evaluation using the following structure:
<evaluation>
<task_verb_analysis>
[Explain how well the response aligns with each identified task verb.]
</task_verb_analysis>
<earned_points_breakdown>
[Provide a detailed justification for each point earned, ensuring strict alignment with the rubric.]
</earned_points_breakdown>
<missed_points_explanation>
[Identify which parts did not earn points and provide specific reasons, referencing the rubric.]
</missed_points_explanation>
<improvement_suggestions>
[Provide clear, constructive feedback on how to improve the response to better match the rubric criteria.]
</improvement_suggestions>
<score_estimate>
[Provide an AP-style score estimate based on the rubric.]
</score_estimate>
</evaluation>
Remember to be extremely thorough and critical in your evaluation. Only award points when the student's response very closely matches the rubric criteria or uses appropriate synonyms that convey the same meaning. Ensure that your assessment is fair, consistent, and aligns strictly with AP's expectations for depth, accuracy, and scientific reasoning.
"""
    print("#### prompt ####")
    print(prompt)
    return prompt

def create_openai_prompt(evaluation_data: str) -> str:
    """
    Create the second prompt (Prompt 2) for OpenAI using the evaluation data from Claude.
    """
    print("#### prompt ####")
    prompt = f"""####
Task: Generate structured FRQ feedback for an AP Environmental Science response based on the evaluation data provided.

{{Evaluation data}}
{evaluation_data}

Instructions:
Maintain the exact structure below, including headings, sections, and formatting.
Use the earned points breakdown and missed points explanation to populate the "What You Did Well" and "How to Improve" sections.
The final score should match the calculated earned points.
Ensure the feedback does not give away the correct answer.
Ensure feedback is detailed, actionable, and aligned with AP Science reasoning standards.
Use a motivational and constructive tone while pointing out errors.
You have to use the provided template and not a word outside of the template.
<template>
Final Score: [Earned Points]/[Total Points]
AP Score Estimate: [Estimated AP Score]
(Your score suggests an overall AP score in the [1-5] range.)

⭐ To pass (score 3+), aim for 50% or higher on FRQs.
⭐ For a 4-5 (college credit level), target 70%+ per FRQ.

✅ What You Did Well ([Earned Points]/[Total Points])
📝 Correct Answers: [List of correct parts]
✔ [Key Strength #1]: [Brief positive feedback based on earned points]
✔ [Key Strength #2]: [Another strong point highlighting a well-done aspect]
✔ [Key Strength #3]: [Additional positive element]

⚠️ How to Improve ([Missed Points]/[Total Points])
🔹 Missed Parts:
[a-ii], b-i]...: ❌ 

🔹 (a-ii) [Missed Concept]
You answered "[Student Response]", but it should reflect "[Correct Concept]".
💡 Think Like a Scientist: [Guiding question for reconsideration].
📝 Tip: [Strategy to avoid this mistake in the future].

🔹 (b-i) [Missed Concept]
Your explanation of [Concept] was [issue: vague/incomplete/inaccurate].
✅ Next Steps: Focus on [specific improvement area] using [scientific reasoning/framework].

🔹 Next Steps to Improve Your FRQ Score

📌 Double-check numerical data before submitting answers to ensure accuracy.
📌 Use specific scientific terminology—avoid vague descriptions and provide precise characteristics.
📌 Practice structuring explanations using reasoning prompts: "Because ____, this leads to ____."
📌 Review key science concepts related to [list of key concepts based on errors] before attempting similar FRQs.
📌 Try rewriting sections (A), (B), and (C) using the strategies above and compare your answers—where do you see improvement

🚀 Keep practicing! Your reasoning skills are improving, and your score will too! 🎉
</template>
You have to customize the template based on the question, rubric and the answer. You have to follow these steps to customize the template.
📌 How to Customize This Template:
Adjust [Earned Points], [Missed Points], [Total Points] based on the evaluation.
Scale 6-point FRQs to a 10-point AP Score Estimate (e.g., 4/6 → 7/10).
Ensure each part is tagged (e.g., a-i, b-ii) to clarify correct/missed responses.
Tailor "Next Steps" and " Next Steps to Improve Your FRQ Score"  based on actual mistakes, not generic advice.
You have to use the provided template and not a word outside of the template.
"""
    print("#### prompt 2 ####")
    print(prompt)
    return prompt

async def call_claude_api(messages: List[Dict[str, str]]) -> str:
    """
    Call the Claude API using the specified messages.
    Uses model "claude-3-5-sonnet-20241022" with set parameters.
    Adds a retry mechanism: if a request fails, wait 4 seconds and retry, up to 3 times.
    """
    for attempt in range(3):
        try:
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 8192,
                "temperature": 0.4,
                "messages": messages
            }
            response = requests.post(CLAUDE_API_URL, headers=CLAUDE_HEADERS, json=payload)
            if response.status_code == 200:
                data = response.json()
                if "content" in data and isinstance(data["content"], list) and len(data["content"]) > 0:
                    return data["content"][0]["text"]
                else:
                    return data.get("text", "")
            else:
                if attempt < 2:
                    await asyncio.sleep(4)
                    continue
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Claude API call failed: {response.text}"
                    )
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(4)
            else:
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

async def classify_frq_course(course_title: str, question: str) -> str:
    """
    Call the Claude API to classify the course of an FRQ.
    Returns the classified course name.
    """
    classification_prompt = create_classification_prompt(course_title, question)
    classification_messages = [{"role": "user", "content": classification_prompt}]
    
    response_text = await call_claude_api(classification_messages)
    
    try:
        # Try to parse the entire response as JSON
        json_response = json.loads(response_text)
        return json_response.get("course_of_the_frq", "")
    except json.JSONDecodeError:
        # If that fails, try to extract just the JSON object from the response text
        json_pattern = r'({[\s\S]*})'
        match = re.search(json_pattern, response_text)
        if match:
            try:
                json_str = match.group(1)
                json_obj = json.loads(json_str)
                return json_obj.get("course_of_the_frq", "")
            except json.JSONDecodeError:
                pass
        
        # If JSON extraction fails, try to directly extract the course name
        course_pattern = r'"course_of_the_frq"\s*:\s*"([^"]+)"'
        match = re.search(course_pattern, response_text)
        if match:
            return match.group(1)
        
        # Fallback to the original course title if everything fails
        return course_title

def normalize_course_name(course_name: str) -> str:
    """
    Normalize the course name for comparison.
    Removes 'AP' prefix if present and converts to lowercase.
    """
    normalized = course_name.strip().lower()
    if normalized.startswith("ap "):
        normalized = normalized[3:]
    return normalized

def extract_score_from_env_science(scoring_response: str, feedback_response: str) -> tuple:
    """
    Extract earned points and total points from Environmental Science response.
    
    Returns:
        tuple: (total_earned_points, total_possible_points)
    """
    # First try to extract from feedback_response which has a clearer format
    score_match = re.search(r'Final Score: (\d+)/(\d+)', feedback_response)
    if score_match:
        earned_points = float(score_match.group(1))
        total_points = float(score_match.group(2))
        return (earned_points, total_points)
    
    # Fallback to extracting from scoring_response if feedback doesn't contain the score
    score_estimate_match = re.search(r'<score_estimate>[\s\S]*?Total Score: (\d+)/(\d+)', scoring_response)
    if score_estimate_match:
        earned_points = float(score_estimate_match.group(1))
        total_points = float(score_estimate_match.group(2))
        return (earned_points, total_points)
    
    # If all else fails, return default values
    return (0.0, 0.0)

def create_environmental_science_html(scoring_response: str, feedback_response: str) -> str:
    """
    Create HTML for Environmental Science FRQ responses.
    
    Args:
        scoring_response: The detailed evaluation from Claude
        feedback_response: The structured feedback from OpenAI
    
    Returns:
        HTML string for displaying the results in a visually appealing way
    """
    # Extract score from the feedback response
    score_match = re.search(r'Final Score: (\d+)/(\d+)', feedback_response)
    earned_points = score_match.group(1) if score_match else "?"
    total_points = score_match.group(2) if score_match else "?"
    
    # Extract AP score estimate
    ap_score_match = re.search(r'AP Score Estimate: (\d+)', feedback_response)
    ap_score = ap_score_match.group(1) if ap_score_match else "?"
    
    # Extract "What You Did Well" section
    did_well_match = re.search(r'✅ What You Did Well.*?(?=⚠️|$)', feedback_response, re.DOTALL)
    did_well_section = did_well_match.group(0) if did_well_match else ""
    
    # Extract "How to Improve" section
    improve_match = re.search(r'⚠️ How to Improve.*?(?=🚀|$)', feedback_response, re.DOTALL)
    improve_section = improve_match.group(0) if improve_match else ""
    
    # Extract "Next Steps" section
    next_steps_match = re.search(r'🔹 Next Steps to Improve Your FRQ Score.*?(?=🚀|$)', feedback_response, re.DOTALL)
    next_steps_section = next_steps_match.group(0) if next_steps_match else ""
    
    html = f"""
    <style>
    /* Global styles */
    .ap-grader-container {{
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background: linear-gradient(135deg, #faf8f5 0%, #fff9f5 100%);
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Header style */
    .ap-header {{
        background: linear-gradient(135deg, #FFE5E5 0%, #FFF0F0 100%);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    
    .ap-title {{
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        color: #2D3748;
        background: linear-gradient(120deg, #FF9A9E 0%, #FAD0C4 50%, #FFD1FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .ap-score {{
        display: flex;
        justify-content: space-around;
        margin-top: 25px;
    }}
    
    .ap-score-item {{
        text-align: center;
    }}
    
    .ap-score-label {{
        font-size: 16px;
        color: #4A5568;
        margin-bottom: 8px;
    }}
    
    .ap-score-value {{
        font-size: 24px;
        font-weight: 700;
        color: #2D3748;
    }}
    
    /* Card style */
    .ap-card {{
        background: #fff;
        border-radius: 8px;
        margin-bottom: 20px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }}
    
    .ap-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    
    .ap-card-title {{
        font-size: 20px;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 15px;
        color: #2D3748;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255, 154, 158, 0.3);
    }}
    
    .ap-content {{
        color: #4A5568;
        line-height: 1.6;
    }}
    
    /* Tab styles */
    .ap-tabs {{
        display: flex;
        margin-bottom: 20px;
    }}
    
    .ap-tab {{
        padding: 10px 20px;
        cursor: pointer;
        border-bottom: 2px solid #ddd;
        flex: 1;
        text-align: center;
        transition: all 0.3s ease;
    }}
    
    .ap-tab.active {{
        border-bottom: 2px solid #FF9A9E;
        color: #FF9A9E;
        font-weight: 500;
    }}
    
    .ap-tab-content {{
        padding: 20px 0;
    }}
    
    /* Highlights */
    .ap-highlight {{
        background: rgba(255, 229, 100, 0.2);
        padding: 2px 4px;
        border-radius: 4px;
    }}
    
    /* Icons and badges */
    .ap-icon {{
        vertical-align: middle;
        margin-right: 6px;
    }}
    
    .ap-badge {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 500;
        margin-right: 8px;
    }}
    
    .ap-badge-success {{
        background: rgba(72, 187, 120, 0.1);
        color: #48BB78;
    }}
    
    .ap-badge-warning {{
        background: rgba(237, 137, 54, 0.1);
        color: #ED8936;
    }}
    
    .ap-section-content {{
        white-space: pre-line;
    }}
    </style>
    
    <div class="ap-grader-container">
        <div class="ap-header">
            <h1 class="ap-title">AP Environmental Science FRQ Evaluation</h1>
            <div class="ap-score">
                <div class="ap-score-item">
                    <div class="ap-score-label">Final Score</div>
                    <div class="ap-score-value">{earned_points}/{total_points}</div>
                </div>
                <div class="ap-score-item">
                    <div class="ap-score-label">AP Score Estimate</div>
                    <div class="ap-score-value">{ap_score}/5</div>
                </div>
            </div>
        </div>
        
        <div class="ap-card">
            <h2 class="ap-card-title">
                <span class="ap-icon">✅</span>What You Did Well
            </h2>
            <div class="ap-content ap-section-content">
                {did_well_section}
            </div>
        </div>
        
        <div class="ap-card">
            <h2 class="ap-card-title">
                <span class="ap-icon">⚠️</span>How to Improve
            </h2>
            <div class="ap-content ap-section-content">
                {improve_section}
            </div>
        </div>
        
        <div class="ap-card">
            <h2 class="ap-card-title">
                <span class="ap-icon">📋</span>Next Steps
            </h2>
            <div class="ap-content ap-section-content">
                {next_steps_section}
                
                <div style="margin-top: 20px; text-align: center; font-weight: 500;">
                    🚀 Keep practicing! Your reasoning skills are improving, and your score will too! 🎉
                </div>
            </div>
        </div>
        
        <div class="ap-tabs">
            <div class="ap-tab active" onclick="showTab('summary')">Summary</div>
            <div class="ap-tab" onclick="showTab('detailed')">Detailed Analysis</div>
        </div>
        
        <div id="summary-tab" class="ap-tab-content" style="display: block;">
            <div class="ap-card">
                <h2 class="ap-card-title">Score Breakdown</h2>
                <div class="ap-content">
                    <p>The final score of <strong>{earned_points}/{total_points}</strong> suggests an overall AP score in the <strong>{ap_score}</strong> range.</p>
                    <p>⭐ To pass (score 3+), aim for 50% or higher on FRQs.<br>
                    ⭐ For a 4-5 (college credit level), target 70%+ per FRQ.</p>
                </div>
            </div>
        </div>
        
        <div id="detailed-tab" class="ap-tab-content" style="display: none;">
            <div class="ap-card">
                <h2 class="ap-card-title">Detailed Evaluation</h2>
                <div class="ap-content">
                    <pre style="white-space: pre-wrap; font-family: inherit;">{scoring_response}</pre>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    function showTab(tabName) {{
        const tabs = document.querySelectorAll('.ap-tab');
        const tabContents = document.querySelectorAll('.ap-tab-content');
        
        tabs.forEach(tab => tab.classList.remove('active'));
        tabContents.forEach(content => content.style.display = 'none');
        
        if (tabName === 'summary') {{
            document.querySelector('.ap-tab:nth-child(1)').classList.add('active');
            document.getElementById('summary-tab').style.display = 'block';
        }} else {{
            document.querySelector('.ap-tab:nth-child(2)').classList.add('active');
            document.getElementById('detailed-tab').style.display = 'block';
        }}
    }}
    </script>
    """
    
    return html

def create_language_literature_html(api_response_json: str) -> str:
    """
    Create HTML for AP Language or Literature FRQ responses.
    
    Args:
        api_response_json: The JSON response from the respective API endpoint
    
    Returns:
        HTML string formatted according to the Streamlit app design
    """
    try:
        # Parse the API response
        api_response = json.loads(api_response_json)
        
        # Extract key values
        rowA = api_response.get('rowA', {})
        rowB = api_response.get('rowB', {})
        rowC = api_response.get('rowC', {})
        
        row_a_score = rowA.get('score', 0)
        row_b_score = rowB.get('score', 0)
        row_c_score = rowC.get('score', 0)
        
        # Calculate total score
        total_score = row_a_score + row_b_score + row_c_score
        
        # Extract summary text
        summary_content = api_response.get("summary", {})
        if isinstance(summary_content, dict):
            overall_assessment = summary_content.get("summary", "")
        elif isinstance(summary_content, str):
            overall_assessment = summary_content
        else:
            overall_assessment = ""
        
        # Check if fact check failed
        fact_check = api_response.get('fact_check', None)
        fact_check_failed = (fact_check is not None and fact_check.get('score', 1) == 0)
        
        if fact_check_failed:
            penalized_total_score = row_a_score + 1 + row_c_score
            fact_check_html = f"""
            <div class="section-container" style="background: #fff7f7;">
                <h4 style="color: #B83232;">Fact Check Failed</h4>
                <p style="color: #4A5568;">
                    <strong>Reasoning:</strong> {fact_check.get('reasoning', '')}
                </p>
                <p style="color: #2D3748;">
                    <strong>Your penalized total score is:</strong> {penalized_total_score}
                </p>
                <p style="color: #2D3748;">
                    <strong>If you hadn't made this error, here's how you would have scored:</strong> {total_score}
                </p>
            </div>
            """
        else:
            fact_check_html = ""
        
        # Create the HTML with the styling from the Streamlit app
        html = f"""
        <style>
        /* Global styles */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(to right, #F2FCFE, #FDFCFB) !important;
        }}

        /* Ensure smooth transitions for all elements */
        * {{
            transition: all 0.2s ease-in-out;
        }}

        .section-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                        box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .section-container:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        }}

        .result-table {{
            width: 100%;
            border-collapse: collapse;
            background: #fff;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1rem;
        }}

        .result-table th,
        .result-table td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }}

        .result-table th {{
            background-color: #f9f9f9;
            text-transform: uppercase;
            font-size: 0.85rem;
            color: #444;
        }}

        .result-table tr:hover {{
            background-color: #f6faff;
            transition: background-color 0.3s ease-in-out;
        }}

        .summary-row {{
            background-color: #FFF7E6;
            color: #333;
        }}

        .total-score {{
            background: linear-gradient(135deg, #FFF5F5 0%, #FFF0F0 100%);
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            color: #2D3748;
        }}
        </style>

        <div style="padding: 20px; max-width: 1200px; margin: 0 auto;">
            {fact_check_html}
            
            <div class="section-container">
                <table class="result-table">
                    <tbody>
                        <tr class="total-score">
                            <td colspan="4" style="text-align: center; font-size: 1.5rem; font-weight: 700; color: #2D3748;">
                                Total Score: {total_score}
                            </td>
                        </tr>
                        <tr class="summary-row">
                            <td colspan="4" style="text-align: left; padding: 1rem;">
                                <div style="max-width: 800px; margin: 0 auto;">
                                    <h3 style="color: #2D3748; margin-bottom: 0.5rem; font-size: 1.25rem;">
                                        Overall Assessment
                                    </h3>
                                    <p style="line-height: 1.8; color: #4A5568;">
                                        {overall_assessment}
                                    </p>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <th>Category</th>
                            <th>Score</th>
                            <th>Commentary</th>
                            <th>Improvements</th>
                        </tr>
                        <tr>
                            <td><strong>Thesis</strong></td>
                            <td style="text-align: center; font-size: 1.25rem; font-weight: 600; color: #2D3748;">
                                {row_a_score}
                            </td>
                            <td>{rowA.get('commentary', '')}</td>
                            <td>{rowA.get('improvements', '')}</td>
                        </tr>
                        <tr>
                            <td><strong>Evidence & Commentary</strong></td>
                            <td style="text-align: center; font-size: 1.25rem; font-weight: 600; color: #2D3748;">
                                {row_b_score}
                            </td>
                            <td>{rowB.get('commentary', '')}</td>
                            <td>{rowB.get('improvements', '')}</td>
                        </tr>
                        <tr>
                            <td><strong>Sophistication</strong></td>
                            <td style="text-align: center; font-size: 1.25rem; font-weight: 600; color: #2D3748;">
                                {row_c_score}
                            </td>
                            <td>{rowC.get('commentary', '')}</td>
                            <td>{rowC.get('improvements', '')}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        return html
    except Exception as e:
        # If there's an error parsing the response, return a basic error message
        return f"""
        <div style="color: red; padding: 20px; text-align: center;">
            <h2>Error generating HTML view</h2>
            <p>There was an error processing the grading results: {str(e)}</p>
        </div>
        """

def create_general_course_html(api_response_json: str) -> str:
    """
    Create HTML for general AP course FRQ responses (like US History, World History, etc.)
    
    Args:
        api_response_json: The JSON response from the general API endpoint
    
    Returns:
        HTML string for displaying the results in a visually appealing table format
    """
    try:
        # Parse the API response
        api_response = json.loads(api_response_json)
        
        # Extract key values (adjusted based on actual API response structure)
        total_points = api_response.get('total_points', 0)
        awarded_points = api_response.get('awarded_points', 0)
        
        # Calculate score percentage if not provided
        if 'score_percentage' in api_response:
            score_percentage = api_response.get('score_percentage', 0)
        else:
            # Calculate it if not provided directly
            score_percentage = round((awarded_points / total_points * 100), 1) if total_points > 0 else 0
        
        # Try different possible field names for rationale
        rationale = api_response.get('rationale', '')
        if not rationale:
            rationale = api_response.get('rationale_for_the_score', '')
        
        # Extract feedback points - check various field names
        feedback_points = []
        if 'feedback' in api_response:
            feedback = api_response.get('feedback', [])
            if isinstance(feedback, list):
                feedback_points = feedback
            elif isinstance(feedback, str):
                feedback_points = [feedback]
        elif 'feedback_to_the_student' in api_response:
            feedback = api_response.get('feedback_to_the_student', [])
            if isinstance(feedback, list):
                feedback_points = feedback
            elif isinstance(feedback, str):
                feedback_points = [feedback]
        
        # Generate feedback HTML
        feedback_html = ""
        for i, point in enumerate(feedback_points, 1):
            feedback_html += f"<li>{point}</li>"
        
        # If no feedback points were found, provide a default message
        if not feedback_html:
            feedback_html = "<li>No specific feedback points provided.</li>"
        
        # Create the HTML
        html = f"""
        <style>
        body {{
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(to right, #F2FCFE, #FDFCFB) !important;
        }}

        .ap-general-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(6px);
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }}
        
        .ap-general-title {{
            font-size: 28px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            color: #2D3748;
            background: linear-gradient(90deg, #FF9A9E, #FAD0C4, #FFD1FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .ap-results-table {{
            width: 100%;
            border-collapse: collapse;
            background: #fff;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1rem;
        }}
        
        .ap-results-table th,
        .ap-results-table td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }}
        
        .ap-results-table th {{
            background-color: #f9f9f9;
            text-transform: uppercase;
            font-size: 0.85rem;
            color: #444;
            text-align: center;
        }}
        
        .ap-results-table tr:hover {{
            background-color: #f6faff;
            transition: background-color 0.3s ease-in-out;
        }}
        
        .ap-label {{
            font-weight: 600;
            color: #4A5568;
        }}
        
        .ap-value {{
            color: #2D3748;
        }}
        
        .ap-rationale {{
            padding: 16px 24px;
            line-height: 1.6;
        }}
        
        .ap-feedback-section {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        }}
        
        .ap-feedback-title {{
            font-size: 22px;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 20px;
            color: #2D3748;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(255, 154, 158, 0.3);
        }}
        
        .ap-feedback-list {{
            padding-left: 20px;
            margin-bottom: 0;
        }}
        
        .ap-feedback-list li {{
            margin-bottom: 12px;
            line-height: 1.6;
            color: #4A5568;
        }}
        
        .ap-feedback-list li:last-child {{
            margin-bottom: 0;
        }}
        </style>
        
        <div class="ap-general-container">
            <h1 class="ap-general-title">AP FRQ Evaluation Results</h1>
            
            <table class="ap-results-table">
                <thead>
                    <tr>
                        <th colspan="2">OVERALL RESULTS</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="ap-label">Total Points</td>
                        <td class="ap-value">{total_points}</td>
                    </tr>
                    <tr>
                        <td class="ap-label">Awarded Points</td>
                        <td class="ap-value">{awarded_points}</td>
                    </tr>
                    <tr>
                        <td class="ap-label">Score (%)</td>
                        <td class="ap-value">{score_percentage}%</td>
                    </tr>
                    <tr>
                        <td class="ap-label">Rationale</td>
                        <td class="ap-rationale">{rationale}</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="ap-feedback-section">
                <h2 class="ap-feedback-title">Feedback to the Student:</h2>
                <ol class="ap-feedback-list">
                    {feedback_html}
                </ol>
            </div>
        </div>
        """
        
        return html
    except Exception as e:
        # If there's an error parsing the response, return a basic error message
        return f"""
        <div style="color: red; padding: 20px; text-align: center;">
            <h2>Error generating HTML view</h2>
            <p>There was an error processing the grading results: {str(e)}</p>
        </div>
        """

@app.post("/grade/", response_model=GradingResponse)
async def grade_frq(
    course_title: str = Form(...),
    question: str = Form(...),
    student_answer: str = Form(...),
    scoring_rubric: str = Form(...),
    images: Optional[List[UploadFile]] = File(None)
):
    """
    Grade an AP FRQ response. 

    1. First, classify the FRQ to determine the appropriate course.
    2. If course is "Environmental Science" (case-insensitive), use Claude + OpenAI for the two-step evaluation.
    3. If course is "AP Language" (case-insensitive), call the AP Language endpoint.
    4. If course is "AP Literature" (case-insensitive), call the AP Literature endpoint.
    5. Otherwise, use the general AP grader endpoint for other courses.

    The final JSON response will include:
      - scoring_response: the detailed evaluation (as string) 
      - feedback_response: the structured feedback (if applicable, otherwise empty string).
      - html_for_app: formatted HTML for displaying in a web application
      - total_possible_points: maximum score possible
      - total_earned_points: points earned by the student
    """
    try:
        # Process images if provided: convert each image to a base64-encoded string.
        if images:
            image_b64_list = []
            for image in images:
                content = await image.read()
                b64_str = base64.b64encode(content).decode("utf-8")
                image_b64_list.append(b64_str)
            image_data = "\n".join(image_b64_list)
        else:
            image_data = ""
        
        # First, classify the FRQ to determine the course
        classified_course = await classify_frq_course(course_title, question)
        normalized_course = normalize_course_name(classified_course)
        
        # 1) If Environmental Science:
        if normalized_course == "environmental science":
            claude_prompt = create_claude_prompt(
                question=question,
                student_answer=student_answer,
                scoring_rubric=scoring_rubric,
                image_data=image_data
            )
            claude_messages = [{"role": "user", "content": claude_prompt}]
            scoring_response = await call_claude_api(claude_messages)

            openai_prompt = create_openai_prompt(scoring_response)
            openai_messages = [{"role": "user", "content": openai_prompt}]
            feedback_response = call_openai_api(openai_messages)

            # Extract score information
            total_earned_points, total_possible_points = extract_score_from_env_science(
                scoring_response, feedback_response
            )

            # Generate HTML for Environmental Science
            html_for_app = create_environmental_science_html(scoring_response, feedback_response)

            return GradingResponse(
                scoring_response=scoring_response,
                feedback_response=feedback_response,
                html_for_app=html_for_app,
                total_possible_points=float(total_possible_points),
                total_earned_points=float(total_earned_points)
            )

        # 2) If AP Language:
        elif normalized_course == "language":
            payload = {
                "full_question": question,
                "student_essay": student_answer,
                "reference_text": ""
            }
            response = requests.post(LANG_ENDPOINT, json=payload)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error from AP Language API: {response.text}"
                )
            api_response = response.json()
            api_response_json = json.dumps(api_response)
            
            # For Lang/Lit, get scores - total possible is always 6, earned is from original_total_score or sum of row scores
            total_possible_points = 6.0
            if "original_total_score" in api_response:
                total_earned_points = float(api_response["original_total_score"])
            else:
                # Sum the row scores (rowA, rowB, rowC)
                row_a_score = api_response.get('rowA', {}).get('score', 0)
                row_b_score = api_response.get('rowB', {}).get('score', 0)
                row_c_score = api_response.get('rowC', {}).get('score', 0)
                total_earned_points = float(row_a_score + row_b_score + row_c_score)
            
            # Generate HTML for Language
            html_for_app = create_language_literature_html(api_response_json)

            return GradingResponse(
                scoring_response=api_response_json,
                feedback_response="",
                html_for_app=html_for_app,
                total_possible_points=total_possible_points,
                total_earned_points=total_earned_points
            )

        # 3) If AP Literature:
        elif normalized_course == "literature":
            payload = {
                "full_question": question,
                "student_essay": student_answer,
                "reference_text": ""
            }
            response = requests.post(LIT_ENDPOINT, json=payload)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error from AP Literature API: {response.text}"
                )
            api_response = response.json()
            api_response_json = json.dumps(api_response)
            
            # For Lang/Lit, get scores - total possible is always 6, earned is from original_total_score or sum of row scores
            total_possible_points = 6.0
            if "original_total_score" in api_response:
                total_earned_points = float(api_response["original_total_score"])
            else:
                # Sum the row scores (rowA, rowB, rowC)
                row_a_score = api_response.get('rowA', {}).get('score', 0)
                row_b_score = api_response.get('rowB', {}).get('score', 0)
                row_c_score = api_response.get('rowC', {}).get('score', 0)
                total_earned_points = float(row_a_score + row_b_score + row_c_score)
            
            # Generate HTML for Literature
            html_for_app = create_language_literature_html(api_response_json)

            return GradingResponse(
                scoring_response=api_response_json,
                feedback_response="",
                html_for_app=html_for_app,
                total_possible_points=total_possible_points,
                total_earned_points=total_earned_points
            )

        # 4) Otherwise, fallback to the general AP grader.
        else:
            data = {
                "question": question,
                "student_answer": student_answer,
                "scoring_rubric": scoring_rubric
            }
            response = requests.post("https://api-ap-grader.onrender.com/score", json=data)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error from AP grader API: {response.text}"
                )
            api_response = response.json()
            api_response_json = json.dumps(api_response)
            
            # For general courses, extract score information from the response
            total_possible_points = float(api_response.get('total_points', 0))
            total_earned_points = float(api_response.get('awarded_points', 0))
            
            # Generate HTML for general courses
            html_for_app = create_general_course_html(api_response_json)

            return GradingResponse(
                scoring_response=api_response_json,
                feedback_response="",
                html_for_app=html_for_app,
                total_possible_points=total_possible_points,
                total_earned_points=total_earned_points
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "AP FRQ Grading API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
