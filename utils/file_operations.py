from typing import List, Any
import json

# random stuff (e.g., file operations)

def read_jsonl(file_path: str) -> List[Any]:
    res = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            res.append(json_obj)
    return res


def write_to_jsonl(file_path: str, data: List[Any]) -> None:
    with open(file_path, 'w') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
