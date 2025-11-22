import os
import sys
import argparse
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

# Hugging Face Transformers
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()  # Hugging Face warning 제거
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Load environment variables
load_dotenv()

# Add parent directory to path for evaluate import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------ 프롬프트 로드 ------------------
def load_prompt_from_json(file_path):
    """JSON 파일에서 프롬프트를 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompt_config = json.load(f)
    
    if "prompt" not in prompt_config:
        raise ValueError(f"File {file_path} does not contain 'prompt' key")
    
    prompt_data = prompt_config["prompt"]
    
    if "system_turns" not in prompt_data or "user_turns" not in prompt_data:
        raise ValueError(f"File {file_path} must contain 'system_turns' and 'user_turns' in 'prompt'")
    
    return prompt_data

# ------------------ Policy LM 학습 ------------------
def train_policy_lm(train_csv, model_dir="policy_lm", epochs=3, batch_size=8, model_name="t5-small"):
    df = pd.read_csv(train_csv)
    
    # 결측치 처리
    df = df.fillna("")
    
    # 학습용 문장 병합: title과 sentence 모두 사용
    pairs = []
    for _, row in df.iterrows():
        if row.get("original_title") and row.get("answer_title"):
            pairs.append({"input": row["original_title"], "target": row["answer_title"]})
        if row.get("original_sentence") and row.get("answer_sentence"):
            pairs.append({"input": row["original_sentence"], "target": row["answer_sentence"]})
    
    if not pairs:
        raise ValueError("No valid training pairs found in CSV.")
    
    dataset = Dataset.from_list(pairs)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    def preprocess(examples):
        inputs = ["Generate hint: " + text for text in examples["input"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(examples["target"], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(preprocess, batched=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="no",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        push_to_hub=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_model(model_dir)
    print(f"✅ Policy LM saved at {model_dir}")
    return tokenizer, model

# ------------------ 힌트 생성 ------------------
def generate_hint(policy_tokenizer, policy_model, text, max_length=50, device="cuda" if torch.cuda.is_available() else "cpu"):
    policy_model.to(device)
    input_ids = policy_tokenizer("Generate hint: " + text, return_tensors="pt").input_ids.to(device)
    outputs = policy_model.generate(input_ids, max_length=max_length, num_beams=4)
    hint = policy_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return hint

# ------------------ 멀티턴 변환 ------------------
def call_api_multi_turn(client, model, text, prompt_data, hint="", 
                        temp1=0.0, max_tokens1=None,
                        temp2=0.0, max_tokens2=None,
                        temp3=0.0, max_tokens3=None):
    """
    3턴 변환 + DSP 힌트 포함
    1턴 - 의미 보존, 
    2턴 - 자연스러움 + 의미 보존, 
    3턴 - 원문과 비교하여 내용 보존 확인 및 보강
    """
    try:
        system_turns = prompt_data["system_turns"]
        user_turns = prompt_data["user_turns"]
        
        # 첫 번째 턴: 힌트 포함
        first_system_prompt = system_turns[0]
        first_user_template = user_turns[0]["template"]
        text_with_hint = f"{text}\nHint: {hint}" if hint else text
        first_user_prompt = first_user_template.format(text=text_with_hint)
        
        first_params = {"model": model, "temperature": temp1}
        if max_tokens1 is not None:
            first_params["max_tokens"] = max_tokens1
        
        resp_first = client.chat.completions.create(
            messages=[
                {"role": "system", "content": first_system_prompt},
                {"role": "user", "content": first_user_prompt}
            ],
            **first_params
        )
        first_result = resp_first.choices[0].message.content.strip()

        # 두 번째 턴
        second_system_prompt = system_turns[1]
        second_user_template = user_turns[1]["template"]
        second_user_prompt = second_user_template.format(text=text, first_result=first_result)
        
        second_params = {"model": model, "temperature": temp2}
        if max_tokens2 is not None:
            second_params["max_tokens"] = max_tokens2
        
        resp_second = client.chat.completions.create(
            messages=[
                {"role": "system", "content": second_system_prompt},
                {"role": "user", "content": second_user_prompt}
            ],
            **second_params
        )
        second_result = resp_second.choices[0].message.content.strip()

        # 세 번째 턴
        third_system_prompt = system_turns[2]
        third_user_template = user_turns[2]["template"]
        third_user_prompt = third_user_template.format(text=text, second_result=second_result)
        
        third_params = {"model": model, "temperature": temp3}
        if max_tokens3 is not None:
            third_params["max_tokens"] = max_tokens3
        
        resp_third = client.chat.completions.create(
            messages=[
                {"role": "system", "content": third_system_prompt},
                {"role": "user", "content": third_user_prompt}
            ],
            **third_params
        )
        final_result = resp_third.choices[0].message.content.strip()
        return final_result

    except Exception as e:
        print(f"[ERROR] {text[:40]}... - {e}")
        return text  # fallback

# ------------------ 평가 ------------------
def run_evaluate(true_df_path, pred_df_path):
    from evaluate import evaluate
    true_df = pd.read_csv(true_df_path)
    pred_df = pd.read_csv(pred_df_path)
    result_df, summary_text, average_scores = evaluate(true_df, pred_df)
    return average_scores, summary_text

# ------------------ main ------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-turn conversion + DSP")
    parser.add_argument("--train_csv", default="data/train_dataset.csv", help="Training CSV path")
    parser.add_argument("--input", default="data/test_dataset.csv", help="Input CSV path")
    parser.add_argument("--output", default="submission_DSP.csv", help="Output CSV path")
    parser.add_argument("--prompt", default="prompt_fit.json", help="Path to prompt JSON file (default: prompt_fit.json)")
    parser.add_argument("--model", default="solar-pro2", help="Model name (default: solar-pro2)")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    parser.add_argument("--policy_model_dir", default="policy_lm", help="Policy LM model directory")
    parser.add_argument("--train_policy_lm", action="store_true", help="Train Policy LM")
    
    # 1턴 파라미터
    parser.add_argument("--temp1", type=float, default=0.0, help="Temperature for 1st turn (default: 0.0)")
    parser.add_argument("--max_tokens1", type=int, default=None, help="Max tokens for 1st turn (default: None)")
    
    # 2턴 파라미터
    parser.add_argument("--temp2", type=float, default=0.0, help="Temperature for 2nd turn (default: 0.0)")
    parser.add_argument("--max_tokens2", type=int, default=None, help="Max tokens for 2nd turn (default: None)")
    
    # 3턴 파라미터
    parser.add_argument("--temp3", type=float, default=0.0, help="Temperature for 3rd turn (default: 0.0)")
    parser.add_argument("--max_tokens3", type=int, default=None, help="Max tokens for 3rd turn (default: None)")
    
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after generation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Policy LM 학습
    if args.train_policy_lm or not os.path.exists(args.policy_model_dir):
        print("🔄 Training Policy LM...")
        tokenizer, policy_model = train_policy_lm(args.train_csv, model_dir=args.policy_model_dir)
    else:
        print(f"📂 Loading Policy LM from {args.policy_model_dir}...")
        tokenizer = T5Tokenizer.from_pretrained(args.policy_model_dir)
        policy_model = T5ForConditionalGeneration.from_pretrained(args.policy_model_dir)

    # 2. 테스트 데이터 로드
    df_test = pd.read_csv(args.input)
    if "original_sentence" not in df_test.columns or "id" not in df_test.columns:
        raise ValueError("Input CSV must contain 'original_sentence' and 'id' columns")

    # 3. Upstage API client
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY missing.")
    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")

    # 4. 프롬프트 로드
    prompt_data = load_prompt_from_json(args.prompt)
    print(f"📝 Loaded prompt from {args.prompt}")

    # 5. 변환
    results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {}
        for idx, text in enumerate(df_test["original_sentence"].astype(str)):
            hint = generate_hint(tokenizer, policy_model, text, device=device)
            future = executor.submit(call_api_multi_turn, client, args.model, text, prompt_data, hint,
                                   args.temp1, args.max_tokens1,
                                   args.temp2, args.max_tokens2,
                                   args.temp3, args.max_tokens3)
            future_map[future] = idx

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Generating"):
            idx = future_map[future]
            results[idx] = future.result()

    # 6. 저장
    final_df = pd.DataFrame([
        {"id": df_test.iloc[i]["id"], "original_sentence": df_test.iloc[i]["original_sentence"],
         "answer_sentence": results[i] if results[i] else df_test.iloc[i]["original_sentence"]}
        for i in range(len(df_test))
    ])
    final_df.to_csv(args.output, index=False)
    print(f"💾 Output saved: {args.output}")

    # 7. 평가
    if args.evaluate:
        print("\n📊 Running evaluation...")
        scores, summary = run_evaluate(args.input, args.output)
        eval_result_path = args.output.replace(".csv", "_eval.json")
        json.dump({"scores": scores, "summary": summary}, open(eval_result_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\n🏁 Evaluation Completed!")
        print(f"📌 Saved: {eval_result_path}")
        print("\n===== 결과 요약 =====")
        print(summary)
        print("\n===================")

if __name__ == "__main__":
    main()

