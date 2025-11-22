# Experiment 2: Multi-turn Conversion with DSP (Direct Preference Search)

3턴 대화 방식의 현대어 변환에 DSP(Direct Preference Search) 기반 힌트 생성을 추가한 실험입니다.  
Leaderboard Public Score: 0.8470  

## 개요

이 실험은 Policy LM을 학습하여 각 입력 문장에 대한 힌트를 생성하고, 이를 프롬프트에 포함시켜 변환 품질을 향상시키는 방법을 탐구합니다.

### 주요 특징

- **3턴 대화 방식**: 의미 보존 → 자연스러움 개선 → 내용 검증
- **DSP 힌트**: Policy LM이 생성한 힌트를 첫 번째 턴에 포함
- **Policy LM**: T5 모델 기반으로 학습 데이터에서 힌트 생성 패턴 학습

## 설치

### 의존성

```bash
pip install pandas tqdm python-dotenv openai torch transformers datasets
```

### 환경 변수

`.env` 파일에 Upstage API 키를 설정하세요:

```bash
UPSTAGE_API_KEY=your_api_key_here
```

## 사용법

### 기본 실행

```bash
python experiment2/generate_fit.py \
    --input data/test_dataset.csv \
    --output submission_DSP.csv \
    --evaluate
```

### Policy LM 학습 포함

처음 실행하거나 Policy LM을 재학습하려면:

```bash
python experiment2/generate_fit.py \
    --input data/test_dataset.csv \
    --output submission_DSP.csv \
    --train_csv data/train_dataset.csv \
    --train_policy_lm \
    --evaluate
```

### 전체 옵션

```bash
python experiment2/generate_fit.py \
    --input data/test_dataset.csv \
    --output submission_DSP.csv \
    --train_csv data/train_dataset.csv \
    --model solar-pro2 \
    --max_workers 8 \
    --policy_model_dir policy_lm \
    --train_policy_lm \
    --temp1 0.0 \
    --temp2 0.0 \
    --temp3 0.0 \
    --evaluate
```

## 주요 옵션

- `--input`: 입력 CSV 파일 경로 (기본값: `data/test_dataset.csv`)
- `--output`: 출력 CSV 파일 경로 (기본값: `submission_DSP.csv`)
- `--train_csv`: 학습 데이터 CSV 경로 (Policy LM 학습용, 기본값: `data/train_dataset.csv`)
- `--prompt`: 프롬프트 JSON 파일 경로 (기본값: `prompt_fit.json`)
- `--model`: 사용할 모델명 (기본값: `solar-pro2`)
- `--max_workers`: 병렬 처리 워커 수 (기본값: 8)
- `--policy_model_dir`: Policy LM 모델 저장 디렉토리 (기본값: `policy_lm`)
- `--train_policy_lm`: Policy LM을 학습할지 여부 (기본값: False, 모델이 없으면 자동 학습)
- `--temp1`, `--temp2`, `--temp3`: 각 턴의 temperature 값
- `--max_tokens1`, `--max_tokens2`, `--max_tokens3`: 각 턴의 max_tokens 값
- `--evaluate`: 생성 후 평가 실행

## 프롬프트 설정

프롬프트는 `prompt_fit.json` 파일에서 관리됩니다. JSON 구조:

```json
{
    "template_name": "프롬프트 이름",
    "description": "프롬프트 설명",
    "prompt": {
        "system_turns": [
            "1턴 system prompt",
            "2턴 system prompt",
            "3턴 system prompt"
        ],
        "user_turns": [
            {
                "template": "1턴 user prompt 템플릿 (힌트 포함)",
                "format_vars": ["text"]
            },
            {
                "template": "2턴 user prompt 템플릿",
                "format_vars": ["text", "first_result"]
            },
            {
                "template": "3턴 user prompt 템플릿",
                "format_vars": ["text", "second_result"]
            }
        ]
    },
    "dsp": {
        "enabled": true,
        "hint_format": "{text}\nHint: {hint}"
    }
}
```

### DSP 힌트 처리

첫 번째 턴의 user prompt에서 `{text}` 변수는 자동으로 힌트가 포함된 형태로 변환됩니다:
- 힌트가 있으면: `{text}\nHint: {hint}`
- 힌트가 없으면: `{text}`

## 3턴 대화 방식

1. **1턴**: 의미 보존 중심 변환 (힌트 포함)
   - Policy LM이 생성한 힌트를 프롬프트에 포함
   - 원문의 모든 의미와 정보 보존에 집중

2. **2턴**: 자연스러움 개선 + 의미 보존
   - 첫 번째 결과를 더 자연스럽고 읽기 쉽게 개선
   - 원문의 의미와 정보는 반드시 보존

3. **3턴**: 원문과 비교하여 내용 보존 검증 및 보강
   - 원문과 변환 결과를 비교하여 누락/왜곡 확인
   - 필요시 보강하여 최종 결과 생성

## Policy LM 학습

Policy LM은 학습 데이터의 원문-답변 쌍을 이용하여 힌트 생성 패턴을 학습합니다.

### 학습 데이터 형식

CSV 파일은 다음 컬럼을 포함해야 합니다:
- `original_title` / `answer_title`: 제목 쌍 (선택)
- `original_sentence` / `answer_sentence`: 문장 쌍 (선택)

### 학습 파라미터

- 모델: `t5-small` (기본값)
- Epochs: 3 (기본값)
- Batch size: 8 (기본값)

학습된 모델은 `--policy_model_dir`에 저장되며, 다음 실행 시 자동으로 로드됩니다.

## 출력 형식

생성된 CSV 파일은 다음 컬럼을 포함합니다:
- `id`: 고유 식별자
- `original_sentence`: 원문 문장
- `answer_sentence`: 변환된 문장

## 평가

`--evaluate` 옵션을 사용하면 생성 후 자동으로 평가가 실행됩니다.

평가 결과는 `{output}_eval.json` 파일로 저장되며, 다음 정보를 포함합니다:
- 카테고리별 점수 (Omission, Restoration, Naturalness, Accuracy)
- 전체 평균 점수
- 상세 요약

## 주의사항

- Policy LM 학습에는 GPU가 권장됩니다 (CUDA 사용 가능 시 자동으로 사용)
- 대용량 데이터 처리 시 `--max_workers` 값을 조정하세요
- Policy LM 모델은 첫 실행 시 자동으로 학습되며, 이후에는 저장된 모델을 사용합니다
- API 호출 시 비용이 발생할 수 있습니다

