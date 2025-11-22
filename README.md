# Mixup: Upstage 현대어 변환 프롬프톤

한자, 고어, 한문, 영어 혼용문을 현대 한국어 기사체로 변환하는 프로젝트입니다.

## 프로젝트 구조

```
.
├── generate.py          # 문장 생성 스크립트 (3턴 대화 방식)
├── evaluate.py          # 평가 스크립트
├── metrics.py           # 평가 메트릭 계산
├── prompt.json          # 프롬프트 설정 (JSON 형식)
├── prompts/             # 프롬프트 파일들 (Python)
└── README.md
```

## 설치

### 1. 의존성 설치

```bash
pip install pandas tqdm python-dotenv openai
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 Upstage API 키를 설정하세요:

```bash
UPSTAGE_API_KEY=your_api_key_here
```

## 사용법

### 문장 생성

`generate.py`를 사용하여 CSV 파일의 문장들을 변환합니다:

```bash
python generate.py \
    --input data/test_dataset.csv \
    --output submission.csv \
    --model solar-pro2 \
    --prompt prompt.json \
    --max_workers 3 \
    --temp1 0.0 \
    --temp2 0.15 \
    --temp3 0.05
```

#### 주요 옵션

- `--input`: 입력 CSV 파일 경로 (기본값: `data/test_dataset.csv`)
- `--output`: 출력 CSV 파일 경로 (기본값: `submission.csv`)
- `--model`: 사용할 모델명 (기본값: `solar-pro2`)
- `--prompt`: 프롬프트 JSON 파일 경로 (기본값: `prompt.json`)
- `--max_workers`: 병렬 처리 워커 수 (기본값: 3)
- `--temp1`, `--temp2`, `--temp3`: 각 턴의 temperature 값
- `--max_tokens1`, `--max_tokens2`, `--max_tokens3`: 각 턴의 max_tokens 값

#### 입력 CSV 형식

CSV 파일은 다음 컬럼을 포함해야 합니다:
- `id`: 고유 식별자
- `original_sentence`: 변환할 원문 문장

#### 출력 CSV 형식

생성된 CSV 파일은 다음 컬럼을 포함합니다:
- `id`: 고유 식별자
- `original_sentence`: 원문 문장
- `answer_sentence`: 변환된 문장

### 평가

`evaluate.py`를 사용하여 생성된 결과를 평가합니다:

```bash
python evaluate.py \
    --truth data/ground_truth.csv \
    --submission submission.csv \
    --output evaluation_results.csv
```

## 프롬프트 설정

프롬프트는 `prompt.json` 파일에서 관리됩니다. JSON 구조:

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
                "template": "1턴 user prompt 템플릿 (Python format string)",
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
    "multi_turn": true,
    "parameters": {
        "temperature": [0.0, 0.15, 0.05],
        "max_tokens": [null, null, null]
    }
}
```

### 3턴 대화 방식

1. **1턴**: 의미 보존 중심 변환
2. **2턴**: 자연스러움 개선 + 의미 보존
3. **3턴**: 원문과 비교하여 내용 보존 검증 및 보강

## 평가 메트릭

평가는 다음 4가지 카테고리로 진행됩니다:

1. **Omission (누락)**: 원문의 내용이 누락되었는지 평가
2. **Restoration (복원)**: □(빈 문자) 복원의 정확도 평가
3. **Naturalness (자연스러움)**: 현대 한국어 표현의 자연스러움 평가
4. **Accuracy (정확도)**: 정보 왜곡 및 부적절한 추가 평가

각 카테고리는 0-1 점수로 평가되며, 전체 점수는 평균으로 계산됩니다.

## 주의사항

- API 호출 시 비용이 발생할 수 있습니다
- 대용량 데이터 처리 시 `--max_workers` 값을 조정하세요
- 중간 결과는 10개 항목마다 자동 저장됩니다
