import pandas as pd
import concurrent.futures
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

api_key = os.getenv("UPSTAGE_API_KEY")

if not api_key:
    raise ValueError("UPSTAGE_API_KEY not found in environment variables")

BASE_SYSTEM_PROMPTS = {
    "omission": """You are an expert evaluator for Classical Korean to Modern Korean conversion. Focus on identifying content omissions from the original text.""",
    "restoration": """You are an expert evaluator for Classical Korean to Modern Korean conversion. Focus on evaluating the accuracy of □ (missing character) restoration.""",
    "naturalness": """You are an expert evaluator for Classical Korean to Modern Korean conversion. Focus on evaluating the naturalness of modern Korean expression.""",
    "accuracy": """You are an expert evaluator for Classical Korean to Modern Korean conversion. Focus on identifying information distortion and inappropriate additions.""",
}


def calculate_score_from_error_count(error_count):
    """Calculate score (0-1) directly from error count"""
    if error_count == 0:
        return 1.0  # 완벽
    elif error_count == 1:
        return 0.9  # 매우 우수
    elif error_count == 2:
        return 0.7  # 양호
    elif error_count == 3:
        return 0.5  # 보통
    elif error_count == 4:
        return 0.3  # 미흡
    elif error_count == 5:
        return 0.1  # 불량
    else:  # 6+ errors
        return 0.0  # 심각한 문제


def calculate_average_severity_scores(aggregated_results):
    """Calculate average severity scores across all articles for each category and overall score"""

    # 각 카테고리별 점수 수집
    category_scores = {
        "omission": [],
        "restoration": [],
        "naturalness": [],
        "accuracy": [],
    }

    # 각 기사에서 카테고리별 점수 추출
    for result in aggregated_results:
        article_scores = {}

        # omission
        if "omission_details" in result:
            count = result["omission_details"]["count"]
            score = calculate_score_from_error_count(count)
            category_scores["omission"].append(score)
            article_scores["omission"] = score

        # restoration
        if "restoration_details" in result:
            count = result["restoration_details"]["count"]
            score = calculate_score_from_error_count(count)
            category_scores["restoration"].append(score)
            article_scores["restoration"] = score

        # naturalness
        if "naturalness_details" in result:
            count = result["naturalness_details"]["count"]
            score = calculate_score_from_error_count(count)
            category_scores["naturalness"].append(score)
            article_scores["naturalness"] = score

        # accuracy
        if "accuracy_details" in result:
            count = result["accuracy_details"]["count"]
            score = calculate_score_from_error_count(count)
            category_scores["accuracy"].append(score)
            article_scores["accuracy"] = score

    # 평균 계산
    average_scores = {}
    for category, scores in category_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            average_scores[category] = {
                "average_score": round(avg_score, 3),
                "total_articles": len(scores),
            }

    # Calculate overall average score (simple average of all categories)
    if average_scores:
        # Only include the 4 main categories, exclude overall
        main_categories = ["omission", "restoration", "naturalness", "accuracy"]
        category_scores = [
            average_scores[cat]["average_score"]
            for cat in main_categories
            if cat in average_scores
        ]
        overall_avg_score = sum(category_scores) / len(category_scores)

        average_scores["overall"] = {
            "average_score": round(overall_avg_score, 3),
            "total_articles": len(aggregated_results),
        }

    return average_scores


def get_user_prompt(category, original, golden, predicted):
    """Get user prompt for each evaluation category"""

    if category == "omission":
        return f"""
# Content Omission Evaluation for Classical Korean to Modern Korean Conversion

⚠️  **CRITICAL INSTRUCTION: This prompt contains FORMAT EXAMPLES ONLY. You must NEVER copy or reproduce any example content. Analyze ONLY the actual input data provided at the bottom of this prompt.**

## Step 1: Task Understanding
Count the number of content omissions where important information from the classical Korean original text is missing in the modern Korean conversion.

## Step 2: Conservative Evaluation Guidelines
### Critical Evaluation Rules
- **BE EXTREMELY CONSERVATIVE**: Only count OBVIOUS and CLEAR content omissions
- **WHEN IN DOUBT, DON'T COUNT**: If uncertain whether something is an omission, set omission_count to 0
- **NO HALLUCINATIONS**: Do not invent omissions that don't exist
- **VERIFY CAREFULLY**: Double-check each potential omission before counting
- **NEVER COPY EXAMPLES**: Do NOT copy or reproduce any examples from this prompt. Analyze only the actual input data provided
- **ANALYZE REAL DATA ONLY**: Base your evaluation solely on the actual Original, Golden, and Model texts provided below

## Step 3: Evaluation Focus Areas
### Types of Omissions to Check (Reference Examples Only - Analyze Your Input Data)

**Complete Sentence Omission**
- Check if entire sentences from the original are missing in the conversion
- Split original text by sentence delimiters (period, question mark, exclamation mark)
- Verify each original sentence has corresponding content in conversion
- Reference Example: Original "기술상개량(技術上改良)은역시참신(亦是嶄新)한기계(機械)를사용(使用)하야최선량(最善良)한효과(效果)를생(生)케함을유의(留意)할것은언(言)을불사(不俟)할지라" → Conversion: [completely missing] → Issue: Technical improvement sentence completely omitted

**Title Omission**
- Check if the title from the original text is missing in the conversion
- Titles are typically on separate lines, centered, or clearly distinguished from content
- Reference Example: Original "농업(農業)과사회(社會) 주의(主義) (사(四))" → Conversion starts with "시골 사람들이..." → Issue: Title "농업과 사회주의 (4)" missing

**Detail Information Omission**
- Missing modifiers or additional explanations
- Missing conditions or exceptions
- Missing technical specifications or descriptions
- Reference Example: Original "피증기우(彼蒸汽又)는전기(電氣)셔와파종기(播種機)와예취기(刈取機)와여(如)한자(者)" → Conversion "파종기와 예취기 같은 농기계" → Issue: "증기나 전기 서기" information omitted

## Step 4: Output Format
**CRITICAL WARNING: The following is just a FORMAT EXAMPLE. Do NOT copy these example values. Use ONLY the actual input data provided below.**

```json
{{
    "omission_count": [NUMBER_OF_ACTUAL_OMISSIONS_FOUND],
    "omissions": [
        {{
            "omission_type": "[ACTUAL_OMISSION_TYPE]",
            "original_content": "[ACTUAL_CONTENT_FROM_INPUT]",
            "explanation": "[YOUR_ACTUAL_ANALYSIS]"
        }}
    ]
}}
```

**REMINDER: Analyze ONLY the actual input data below. Do NOT use example content above.**

⚠️  **FINAL WARNING: The reference examples above are for understanding issue types only. You must NEVER copy these examples into your output. Identify real issues from the actual input data provided below.**

## Input Data
Original Classical Korean: {original}
Golden Reference: {golden}
Model Conversion: {predicted}
"""

    elif category == "restoration":
        return f"""
# □ (Missing Character) Restoration Accuracy Evaluation

⚠️  **CRITICAL INSTRUCTION: This prompt contains FORMAT EXAMPLES ONLY. You must NEVER copy or reproduce any example content. Analyze ONLY the actual input data provided at the bottom of this prompt.**

## Step 1: Task Understanding
Evaluate the accuracy of restoring □ (missing characters) in the classical Korean to modern Korean conversion.

## Step 2: Conservative Evaluation Guidelines
### Critical Evaluation Rules
- **BE EXTREMELY CONSERVATIVE**: Only count OBVIOUS restoration errors
- **WHEN IN DOUBT, DON'T COUNT**: If uncertain whether restoration is inappropriate, set error_count to 0
- **NO HALLUCINATIONS**: Do not invent errors that don't exist
- **VERIFY CAREFULLY**: Check contextual appropriateness of restored content
- **NEVER COPY EXAMPLES**: Do NOT copy or reproduce any examples from this prompt. Analyze only the actual input data provided
- **ANALYZE REAL DATA ONLY**: Base your evaluation solely on the actual Original, Golden, and Model texts provided below

## Step 3: Evaluation Focus Areas
### Types of Restoration Issues to Check (Reference Examples Only - Analyze Your Input Data)

**Unrestored □**
- Count □ symbols that remain unrestored in the conversion
- Compare original □ count with conversion □ count
- Reference Example: Original "향리(鄉里)의인(人)이단순(單純)한농촌(農村)의생활(生活)을기(棄)하고번화(繁華)한도시(都市)로유입(流入)함은□□일반적경향(一般的傾向)이라" → Conversion "시골 사람들이...유입되는 것은 □□ 일반적 경향이다" → Issue: □□ symbols left unrestored

**Inappropriate □ Restoration**
- Restored words that are contextually unnatural
- Grammatically incorrect restorations
- Semantically inappropriate restorations based on surrounding context
- Reference Example: Original "도회(都會)에는우(又)□능력(能力)의발휘(發揮)할기회(機會)가유(有)하나" → Conversion "도시에는 또한 돈의 능력을 발휘할 기회가 있으나" → Issue: □ restored as '돈의' but '개인의' or '자신의' would be more appropriate

**Excessive □ Restoration**
- Adding words where no □ existed in original
- Over-interpretation beyond single □ restoration
- Adding excessive meaning to single □ symbols
- Reference Example: Original "자작농(自作農)□함으로써만족(滿足)할자(者)" → Conversion "자작농으로서 풍족한 생활을 영위함으로써 만족할 자" → Issue: Single □ excessively expanded to '풍족한 생활을 영위'

## Step 4: Output Format
**CRITICAL WARNING: The following is just a FORMAT EXAMPLE. Do NOT copy these example values. Use ONLY the actual input data provided below.**

```json
{{
    "error_count": [NUMBER_OF_ACTUAL_ERRORS_FOUND],
    "errors": [
        {{
            "error_type": "[ACTUAL_ERROR_TYPE]",
            "original_context": "[ACTUAL_CONTEXT_FROM_INPUT]",
            "model_restoration": "[ACTUAL_MODEL_OUTPUT]",
            "explanation": "[YOUR_ACTUAL_ANALYSIS]"
        }}
    ]
}}
```

**REMINDER: Analyze ONLY the actual input data below. Do NOT use example content above.**

⚠️  **FINAL WARNING: The reference examples above are for understanding issue types only. You must NEVER copy these examples into your output. Identify real issues from the actual input data provided below.**

## Input Data
Original Classical Korean: {original}
Golden Reference: {golden}
Model Conversion: {predicted}
"""

    elif category == "naturalness":
        return f"""
# Modern Korean Naturalness Evaluation

⚠️  **CRITICAL INSTRUCTION: This prompt contains FORMAT EXAMPLES ONLY. You must NEVER copy or reproduce any example content. Analyze ONLY the actual input data provided at the bottom of this prompt.**

## Step 1: Task Understanding
Evaluate whether the converted text reads naturally as modern Korean that a 2025 reader can comfortably understand.

## Step 2: Conservative Evaluation Guidelines
### Critical Evaluation Rules
- **BE EXTREMELY CONSERVATIVE**: Only count OBVIOUS naturalness violations
- **WHEN IN DOUBT, DON'T COUNT**: If uncertain whether expression is unnatural, set violation_count to 0
- **NO HALLUCINATIONS**: Do not invent violations that don't exist
- **VERIFY CAREFULLY**: Check if expressions are truly problematic for modern readers
- **NEVER COPY EXAMPLES**: Do NOT copy or reproduce any examples from this prompt. Analyze only the actual input data provided
- **ANALYZE REAL DATA ONLY**: Base your evaluation solely on the actual Original, Golden, and Model texts provided below

## Step 3: Evaluation Focus Areas
### Types of Naturalness Issues to Check (Reference Examples Only - Analyze Your Input Data)

**Speech Level Inconsistency**
- Inconsistent use of formal endings (-습니다, -했습니다) vs informal endings (-다, -았다)
- Mixed speech levels within the same text
- Reference Example: "현재의 문명은 도시 중심 문명입니다. 따라서 농촌은 소외된다." → Issue: Mixed formal (-습니다) and informal (-다) endings

**Archaic Language Remnants**
- Archaic sentence endings like "~하도다", "~할지라도", "~하랴면"
- Archaic particles like "~에재하야", "~으로인하야"
- Reference Example: "문명과 그 시설의 은파를 도비에 보급케 함이 필요하도다" → Issue: Archaic "~하도다", "~케 함이" expressions remain

**Literal Translation Style**
- Awkward word order uncommon in modern Korean
- Simple sound reading of Chinese characters without natural Korean expression
- Reference Example: "농촌 사람은 단지 잔효냉배를 상할 뿐이다" → Issue: "잔효냉배" is awkward literal reading; should be "찌꺼기" or "나머지"

**Unnatural Vocabulary Choice**
- Contextually awkward modern Korean word choices
- Inappropriate translation of technical terms
- Overly literal translations that sound unnatural
- Reference Example: "농업이 개인개인의 기업으로 소규모에 분립하여" → Issue: "개인개인의 기업" should be "개별 농가의 경영"; "분립" should be "분산"

## Step 4: Output Format
**CRITICAL WARNING: The following is just a FORMAT EXAMPLE. Do NOT copy these example values. Use ONLY the actual input data provided below.**

```json
{{
    "violation_count": [NUMBER_OF_ACTUAL_VIOLATIONS_FOUND],
    "violations": [
        {{
            "violation_type": "[ACTUAL_VIOLATION_TYPE]",
            "example": "[ACTUAL_EXAMPLE_FROM_MODEL_OUTPUT]",
            "explanation": "[YOUR_ACTUAL_ANALYSIS]"
        }}
    ]
}}
```

**REMINDER: Analyze ONLY the actual input data below. Do NOT use example content above.**

⚠️  **FINAL WARNING: The reference examples above are for understanding issue types only. You must NEVER copy these examples into your output. Identify real issues from the actual input data provided below.**

## Input Data
Original Classical Korean: {original}
Golden Reference: {golden}
Model Conversion: {predicted}
"""

    elif category == "accuracy":
        return f"""
# Information Preservation Evaluation

⚠️  **CRITICAL INSTRUCTION: This prompt contains FORMAT EXAMPLES ONLY. You must NEVER copy or reproduce any example content. Analyze ONLY the actual input data provided at the bottom of this prompt.**

## Step 1: Task Understanding
Evaluate whether core information from the original classical Korean text is preserved without distortion or inappropriate additions.

## Step 2: Conservative Evaluation Guidelines
### Critical Evaluation Rules
- **BE EXTREMELY CONSERVATIVE**: Only count OBVIOUS information distortions or inappropriate additions
- **WHEN IN DOUBT, DON'T COUNT**: If uncertain whether information is distorted, set error_count to 0
- **NO HALLUCINATIONS**: Do not invent errors that don't exist
- **VERIFY CAREFULLY**: Check if meaning is truly changed or incorrectly added
- **NEVER COPY EXAMPLES**: Do NOT copy or reproduce any examples from this prompt. Analyze only the actual input data provided
- **ANALYZE REAL DATA ONLY**: Base your evaluation solely on the actual Original, Golden, and Model texts provided below

## Step 3: Evaluation Focus Areas
### Types of Information Issues to Check (Reference Examples Only - Analyze Your Input Data)

**Core Information Distortion**
- Changes to proper nouns (names, places, institutions)
- Alterations to key events or content meaning
- Conversion of specific information to vague expressions
- Reference Example: Original "구미제국농업(歐米諸國農業)" → Conversion "서구 선진국 농업" → Issue: Specific "유럽과 미국 제국들" changed to vague "서구 선진국"

**Numerical Information Errors**
- Incorrect conversion of numbers, dates, time, currency, measurements
- Errors in converting Chinese numerals to Arabic numerals
- Wrong preservation of numerical values during unit conversions
- Reference Example: Original "(사(四))" → Conversion "(3)" → Issue: "4번째" incorrectly changed to "3번째"

**Meaning Distortion**
- Translation to opposite meaning
- Confusion of positive/negative
- Mix-up of subject/object relationships
- Fundamental changes to the original intent
- Reference Example: Original "농촌(農村)이차(此)와여(如)히부단(不斷)하게원기(元氣)와능력(能力)이유(有)한분자(分子)를실(失)하는사정(事情)" → Conversion "농촌이 지속적으로 인재를 얻는 상황" → Issue: Original meaning "losing talent" distorted to "gaining talent"

**Inappropriate Content Addition**
- Adding information not present in the original
- Over-interpretation beyond the original scope
- Including modern concepts not in the classical text

## Step 4: Output Format
**CRITICAL WARNING: The following is just a FORMAT EXAMPLE. Do NOT copy these example values. Use ONLY the actual input data provided below.**

```json
{{
    "error_count": [NUMBER_OF_ACTUAL_ERRORS_FOUND],
    "errors": [
        {{
            "error_type": "[ACTUAL_ERROR_TYPE]",
            "original_content": "[ACTUAL_CONTENT_FROM_INPUT]",
            "model_conversion": "[ACTUAL_MODEL_OUTPUT]",
            "explanation": "[YOUR_ACTUAL_ANALYSIS]"
        }}
    ]
}}
```

**REMINDER: Analyze ONLY the actual input data below. Do NOT use example content above.**

⚠️  **FINAL WARNING: The reference examples above are for understanding issue types only. You must NEVER copy these examples into your output. Identify real issues from the actual input data provided below.**

## Input Data
Original Classical Korean: {original}
Golden Reference: {golden}
Model Conversion: {predicted}
"""

    return ""


def single_eval(args):
    # Each process must create its own client
    import json

    original, golden, predicted, category, model_name = args
    system_prompt = BASE_SYSTEM_PROMPTS[category]
    user_prompt = get_user_prompt(category, original, golden, predicted)

    try:
        # Use OpenAI API
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content

        # Try to extract JSON from the response
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except:
                    result = {"error_count": 0, "errors": []}
            else:
                result = {"error_count": 0, "errors": []}

        # Extract count based on category
        if category == "omission":
            count = result.get("omission_count", 0)
            details = result.get("omissions", [])
        elif category == "restoration":
            count = result.get("error_count", 0)
            details = result.get("errors", [])
        elif category == "naturalness":
            count = result.get("violation_count", 0)
            details = result.get("violations", [])
        elif category == "accuracy":
            count = result.get("error_count", 0)
            details = result.get("errors", [])
        else:
            count = 0
            details = []

    except Exception as e:
        count = 0
        details = []

    return {
        "국문기사": original,
        "골든레퍼런스": golden,
        "모델예측결과": predicted,
        "카테고리": category,
        "count": count,
        "details": details,
    }

def evaluate_correction(true_df: pd.DataFrame, pred_df: pd.DataFrame, max_workers=None):
    """
    Evaluate correction quality using multiprocessing
    
    Args:
        true_df: DataFrame with original_sentence and answer_sentence
        pred_df: DataFrame with answer_sentence
        max_workers: Maximum number of worker processes (default: CPU count)
    """
    articles_ko = true_df["original_sentence"].tolist()
    articles_en = true_df["answer_sentence"].tolist()
    articles_model = pred_df["answer_sentence"].tolist()
    categories = list(BASE_SYSTEM_PROMPTS.keys())
    
    tasks = []
    for i, (orig, gold, pred) in enumerate(
        zip(articles_ko, articles_en, articles_model)
    ):
        for category in categories:
            tasks.append((orig, gold, pred, category, "solar-pro2"))
    
    # Set default max_workers to CPU count if not specified
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(single_eval, tasks), total=len(tasks), desc="Evaluating"):
            results.append(result)
            
    # Aggregate results by article
    aggregated_results = []
    for i in range(0, len(results), len(categories)):
        article_results = results[i : i + len(categories)]
        
        # Group by category
        category_results = {}
        for result in article_results:
            category = result["카테고리"]
            category_results[category] = {
                "count": result["count"],
                "details": result["details"],
            }

        # Create aggregated result
        aggregated_result = {
            "국문기사": article_results[0]["국문기사"],
            "골든레퍼런스": article_results[0]["골든레퍼런스"],
            "모델예측결과": article_results[0]["모델예측결과"],
            # Score columns (for easy analysis)
            "omission_score": calculate_score_from_error_count(
                category_results.get("omission", {"count": 0})["count"]
            ),
            "restoration_score": calculate_score_from_error_count(
                category_results.get("restoration", {"count": 0})["count"]
            ),
            "naturalness_score": calculate_score_from_error_count(
                category_results.get("naturalness", {"count": 0})["count"]
            ),
            "accuracy_score": calculate_score_from_error_count(
                category_results.get("accuracy", {"count": 0})["count"]
            ),
            # Detailed information columns
            "omission_details": category_results.get(
                "omission", {"count": 0, "details": []}
            ),
            "restoration_details": category_results.get(
                "restoration", {"count": 0, "details": []}
            ),
            "naturalness_details": category_results.get(
                "naturalness", {"count": 0, "details": []}
            ),
            "accuracy_details": category_results.get(
                "accuracy", {"count": 0, "details": []}
            ),
        }

        aggregated_results.append(aggregated_result)

    result_df = pd.DataFrame(aggregated_results)

    # Calculate average severity scores across all articles
    average_scores = calculate_average_severity_scores(aggregated_results)
    
    return result_df, average_scores
    
    
    
    
    
    
    
