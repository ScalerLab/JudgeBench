from typing import List, Dict, Any

# compute final metrics
# basically just compute how accuracy the judge is

def flip_judgment(decision: str) -> str:
    if decision == "A>B":
        decision = "B>A"
    elif decision == "B>A":
        decision = "A>B"
    return decision


def compute_final_metrics(pairs: List[Dict[str, Any]], reverse_order: bool, include_fn=lambda x: x) -> None:
    
    pairs = [pair for pair in pairs if include_fn(pair)]

    n_pairs = len(pairs)

    if not reverse_order:
        n_correct = sum(
            pair["judgments"][0]["decision"] == pair["label"]
            for pair in pairs
        )
        n_incorrect = n_pairs - n_correct
        return 100*n_correct/n_pairs
        
    else:
        
        n_all_correct = 0
        n_all_incorrect = 0
        n_some_correct = 0
        
        n_correct = 0
        n_incorrect = 0
        n_tie = 0
        
        n_nulls = 0
        n_inconsistent = 0
        
        for pair in pairs:
            
            label = pair["label"]
            judgment1, judgment2 = pair["judgments"]
        
            decision1 = judgment1["decision"] if judgment1 is not None else None
            decision2 = flip_judgment(judgment2["decision"] if judgment2 is not None else None)
            
            if decision1 is None or decision2 is None:
                n_nulls += 1
            
            # consistency metrics
            if decision1 == label and decision2 == label:
                n_all_correct += 1
            elif decision1 != label and decision2 != label:
                n_all_incorrect += 1
            else:
                n_some_correct += 1
                
            if decision1 != decision2:
                n_inconsistent += 1
                
            # new metrics
            counter = 0
            for decision in [decision1, decision2]:
                if decision == label:
                    counter += 1
                elif decision == flip_judgment(label):
                    counter -= 1
                
            if counter > 0:
                n_correct += 1
            elif counter < 0:
                n_incorrect += 1
            else:
                n_tie += 1
        
        return 100*n_correct/n_pairs
        
##############################################################
