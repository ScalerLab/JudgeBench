from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import re

import utils.prompts as prompts
import utils.models as models
from utils.rubric import rubric

# run meta-judge on judgements

class meta_judge_agent:
    def __init__(self, model_A: str, model_B: str, model_C: str) -> None:
        self.api_A = models.get_chat_api_from_model(model_A)
        #self.api_B = models.get_chat_api_from_model(model_B)
        #self.api_C = models.get_chat_api_from_model(model_C)

    def get_meta_score(self, output):
        meta_score = output
        return meta_score

    async def get_meta_judgment(self, question: str, answer_A: str, answer_B: str, judgement: str, decision: str) -> Dict[str, Any]:
        prompt = prompts.render_template(
            "meta_judge_prompt", question=question, answer_a=answer_A, answer_b=answer_B, judgement_1 = judgement, decision_1 = decision)
        prompt += f"Combine judgment and evaluation to finally assign a score for each criterion based on the following rubrics:"
        for criterion in rubric:
            prompt += f"Criterion: {criterion['criterion']}\nDescription:: {criterion['description']}\n"
            for i in range(1, 6):
                prompt += f"{i}: {criterion[f'score_{i}']}\n"
            prompt += "Please respond in the following format: \n Criterion: [criterion_name]\n Score: [1-5]\n\n"
        output = await self.api_A.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )
        
        score = self.get_meta_score(output.strip())
        print("meta_judge_score:", output)
        return {
            "meta_score": score
        }
