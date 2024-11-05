from typing import List, Dict, Any
import argparse
import asyncio
import json
import os
import random

from tqdm.asyncio import tqdm_asyncio

import utils.file_operations as file_operations
import utils.judges as judges
import utils.metrics as metrics


async def judge_pairs(pairs: List[Dict[str, Any]], judge_name: str, judge_model: str, concurrency_limit: int = 1, reverse_order: int = False, output_file: str = None):
    semaphore = asyncio.Semaphore(concurrency_limit)
    judge = judges.get_judge_from_judge_name_and_model(judge_name, judge_model)
    file_lock = asyncio.Lock()
    
    async def judge_pair(pair: Dict[str, Any]):
        async with semaphore:
            
            question = pair["question"]
            response_A = pair["response_A"]
            response_B = pair["response_B"]
            
            try:
                judgment_1 = await judge.get_judgment(question, response_A, response_B)
            except Exception as e:
                print(f"Failed to judge pair {pair['pair_id']} due to the following error: {e}.")
                judgment_1 = None
            judgments = [judgment_1]
            
            if reverse_order:
                try:
                    judgment_2 = await judge.get_judgment(question, response_B, response_A)
                except Exception as e:
                    print(f"Failed to judge pair {pair['pair_id']} due to the following error: {e}.")
                    judgment_2 = None
                judgments.append(judgment_2)
            
            pair["judge_name"] = judge_name
            pair["judgments"] = judgments
            return pair

    tasks = [asyncio.create_task(judge_pair(pair)) for pair in pairs]

    for future in tqdm_asyncio.as_completed(tasks):
        pair = await future
        if output_file is not None:
            async with file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    return pairs


def main(args: argparse.Namespace) -> None:
    
    random.seed(args.seed)
    
    pairs = file_operations.read_jsonl(args.pairs)    

    dataset_name = os.path.basename(args.pairs).replace(".jsonl", "")
    file_path = f"{dataset_name},judge_name={args.judge_name},judge_model={args.judge_model.replace('/', '_')}.jsonl"
    os.makedirs("./outputs", exist_ok=True)
    file_path = os.path.join("./outputs", file_path)
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping judging pairs...")
        original_num_pairs = len(pairs)
        existing_pairs = file_operations.read_jsonl(file_path)
        existing_pair_ids = {pair["pair_id"] for pair in existing_pairs}
        pairs = [pair for pair in pairs if pair["pair_id"] not in existing_pair_ids]
        print(f"Skipped {original_num_pairs - len(pairs)} pairs.")


    if pairs: 
        print("Judging pairs ...")
        pairs = asyncio.run(
            judge_pairs(
                pairs,
                args.judge_name,
                args.judge_model,
                reverse_order=not args.single_game,
                concurrency_limit=args.concurrency_limit,
                output_file=file_path,
            )
        )

    # 7. compute final metrics
    print("Computing final metrics ...") 
    pairs = file_operations.read_jsonl(file_path)  # need to load all the history, not just the generated one.
    for source in ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", ""]:
        print(f"\n{source if source else 'Overall'}:")
        metrics.compute_final_metrics(pairs, not args.single_game, include_fn = lambda x: x["source"].startswith(source))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge_name', type=str, required=True) # name of judge, should correspond to an entry in utils/judges/get_judge_from_judge_name_and_model.
    parser.add_argument('--judge_model', type=str, required=True) # model to be used by judge.
    parser.add_argument('--single_game', action="store_true") # by default, we run each pair through twice (A,B) and (B,A). This flag will only run the original ordering, and should be used if a judge is order-independent.
    parser.add_argument('--seed', type=int, default=42) # seed to use.
    parser.add_argument('--concurrency_limit', type=int, default=1) # We use asyncio to speed things up, 10 is usally a good value here.
    parser.add_argument('--pairs', type=str, required=True) # path to jsonl containing pairs for judging
    args = parser.parse_args()
    main(args)
