import os
import argparse
import json

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

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


def call_api_multi_turn(client, model, text, prompt_data, 
                        temp1=0.0, max_tokens1=None,
                        temp2=0.0, max_tokens2=None,
                        temp3=0.0, max_tokens3=None):
    """3턴 API 호출 함수: 
    1턴 - 의미 보존, 
    2턴 - 자연스러움 + 의미 보존, 
    3턴 - 원문과 비교하여 내용 보존 확인 및 보강
    
    각 턴마다 다른 temperature와 max_tokens 사용 가능
    """
    try:
        system_turns = prompt_data["system_turns"]
        user_turns = prompt_data["user_turns"]
        
        # 첫 번째 턴
        first_system_prompt = system_turns[0]
        first_user_template = user_turns[0]["template"]
        first_user_prompt = first_user_template.format(text=text)
        
        first_params = {"model": model, "temperature": temp1}
        if max_tokens1 is not None:
            first_params["max_tokens"] = max_tokens1
        
        resp_first = client.chat.completions.create(
            messages=[
                {"role": "system", "content": first_system_prompt},
                {"role": "user", "content": first_user_prompt},
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
                {"role": "user", "content": second_user_prompt},
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
                {"role": "user", "content": third_user_prompt},
            ],
            **third_params
        )
        final_result = resp_third.choices[0].message.content.strip()
        
        return final_result
    except Exception as e:
        print(f"[ERROR] {text[:40]}... - {e}")
        return text  # fallback


def main():
    parser = argparse.ArgumentParser(description="Generate modified sentences using Upstage API (3-turn conversation mode)")
    parser.add_argument("--input", default="data/test_dataset.csv", help="Input CSV path containing original_sentence column")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--model", default="solar-pro2", help="Model name (default: solar-pro2)")
    parser.add_argument("--max_workers", type=int, default=3, help="Number of parallel workers (default: 3)")
    parser.add_argument("--prompt", default="prompt.json", help="Path to prompt JSON file (default: prompt.json)")
    
    # 1턴 파라미터
    parser.add_argument("--temp1", type=float, default=0.0, help="Temperature for 1st turn (default: 0.0)")
    parser.add_argument("--max_tokens1", type=int, default=None, help="Max tokens for 1st turn (default: None)")
    
    # 2턴 파라미터
    parser.add_argument("--temp2", type=float, default=0.15, help="Temperature for 2nd turn (default: 0.0)")
    parser.add_argument("--max_tokens2", type=int, default=None, help="Max tokens for 2nd turn (default: None)")
    
    # 3턴 파라미터
    parser.add_argument("--temp3", type=float, default=0.05, help="Temperature for 3rd turn (default: 0.0)")
    parser.add_argument("--max_tokens3", type=int, default=None, help="Max tokens for 3rd turn (default: None)")
    
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    
    if "original_sentence" not in df.columns:
        raise ValueError("Input CSV must contain 'original_sentence' column")
    
    if "id" not in df.columns:
        raise ValueError("Input CSV must contain 'id' column")

    # Setup Upstage client
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
    
    # Load prompt from JSON file
    prompt_data = load_prompt_from_json(args.prompt)
    
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Prompt file: {args.prompt}")
    print(f"Max workers: {args.max_workers}")
    print(f"Mode: 3-turn conversation")
    print(f"  1st: Meaning preservation (temp={args.temp1}, max_tokens={args.max_tokens1})")
    print(f"  2nd: Naturalness + meaning preservation (temp={args.temp2}, max_tokens={args.max_tokens2})")
    print(f"  3rd: Content preservation verification & reinforcement (temp={args.temp3}, max_tokens={args.max_tokens3})")

    # Initialize results dictionary to maintain original order
    results = {idx: None for idx in range(len(df))}
    completed_count = 0
    save_interval = 10  # Save every 10 completed items
    
    # Process each sentence with parallel workers (multi-turn)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(call_api_multi_turn, client, args.model, text, prompt_data,
                          args.temp1, args.max_tokens1,
                          args.temp2, args.max_tokens2,
                          args.temp3, args.max_tokens3): idx
            for idx, text in enumerate(df["original_sentence"].astype(str).tolist())
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            completed_count += 1
            
            # Periodically save intermediate results
            if completed_count % save_interval == 0:
                # Create DataFrame with completed results in original order
                completed_results = []
                for i in range(len(df)):
                    if results[i] is not None:
                        completed_results.append({
                            "id": df.iloc[i]["id"],
                            "original_sentence": df.iloc[i]["original_sentence"],
                            "answer_sentence": results[i]
                        })
                    else:
                        # For incomplete items, use original sentence as placeholder
                        completed_results.append({
                            "id": df.iloc[i]["id"],
                            "original_sentence": df.iloc[i]["original_sentence"],
                            "answer_sentence": df.iloc[i]["original_sentence"]
                        })
                
                out_df = pd.DataFrame(completed_results)
                out_df.to_csv(args.output, index=False)
                # print(f"\n[Progress] Saved {completed_count}/{len(df)} results to {args.output}")

    # Final save with all results in original order
    final_results = []
    for idx in range(len(df)):
        final_results.append({
            "id": df.iloc[idx]["id"],
            "original_sentence": df.iloc[idx]["original_sentence"],
            "answer_sentence": results[idx] if results[idx] is not None else df.iloc[idx]["original_sentence"]
        })
    
    out_df = pd.DataFrame(final_results)
    out_df.to_csv(args.output, index=False)
    print(f"\n✅ Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()

