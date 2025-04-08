import re
from typing import Dict, Tuple, Optional, Union, Any


def extract_xml_answer(text: str) -> str:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, text, re.DOTALL))
    if not matches:
        return None
        
    final_answer = matches[-1].group(1).strip()
    return final_answer


def normalize_answer(answer: str) -> str:
    """Normalizes the answer text for better comparison.
    Args:
        answer: Raw answer text
    Returns:
        Normalized answer text
    """
    # Convert to lowercase  
    normalized = answer.lower()
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    # Remove punctuation that doesn't affect meaning
    normalized = re.sub(r'[.,;:!?]', '', normalized)
    return normalized

def reward_func(response, answer):
    pattern = r".*?<answer>.*?</answer>\s*\b<|im_end|>$"
    
    tags = {
        'ans_start': ('<answer>', 1),
        'ans_end': ('</answer>', 1),
    }
    counts = 0
    for tag_name, (tag_str, expected_count) in tags.items():
        count = response.count(tag_str)
        if count == expected_count:
            counts +=1
    if counts == 2:
        match = re.match(pattern, response, re.DOTALL | re.MULTILINE)
        if match:
            response_ = extract_xml_answer(response)
            norm_response = normalize_answer(response_)
            norm_answer = normalize_answer(answer)
            #ans_pattern = r"\b(?:in|at|on|inside)?\s*(?:the\s*)?" + re.escape(norm_answer) + r"\b$"
            ans_pattern = r"\b(?:in|at|on|inside|)?\s*(?:the\s*)?(?:\w+'s\s*)?" + re.escape(norm_answer) + r"\s*\b$"
            match = re.match(ans_pattern, norm_response, re.DOTALL | re.MULTILINE)
            if match:
                print(f'Right format and exactly match, score: 2, response: ({norm_response}), answer: ({norm_answer})')
                return 2
            else:
                print(f'Right format but wrong answer, score: 0, response: ({norm_response}), answer: ({norm_answer})')
                return 0
        else:
            print('Right format (tag counts correct) but no answer found, score: 0')
            return 0
    print(f'Wrong format, score: 0')
    return 0


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    # breakpoint()
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def compute_score(solution_str: str, 
                 ground_truth: Union[Dict[str, Any], str],
                 format_reward: int = 1,
                 answer_reward: float = 2.0) -> float:
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data or string with ground truth answer
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Maximum points awarded for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Processing ToM Sample ".center(80, '='))
    
    # Extract ground truth
    if isinstance(ground_truth, dict):
        gt_answer = ground_truth.get('expected_answer', '')
    else:
        gt_answer = ground_truth
    
    print(f"[Ground Truth] Expected answer: {gt_answer}")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    print(f'\n[Evaluating]')
    total_score = reward_func(processed_str, gt_answer)
    
    print("="*80 + "\n")

    return total_score


if __name__ == "__main__":
    # Test cases for different types of ExploreToM answers
    test_cases = [
        # {
        #     "ground_truth": "does not know about it",
        #     "model_response": "Assistant: <think>Let me reason through this step by step. Isabella doesn't have direct knowledge of Colton's belief about festival marketing strategies because they haven't communicated about it. Isabella has her own understanding, but without explicit communication with Colton, she cannot know his beliefs on the matter.</think><answer>Isabella does not know about it</answer>"
        # },
        # {
        #     "ground_truth": "leather briefcase",
        #     "model_response": "Assistant: <think>I need to trace Kaylee's understanding of Liam's belief. Since Liam saw the silver letter opener being moved to the leather briefcase, but Kaylee doesn't know this, she would think Liam still believes it's in the original location.</think><answer>Kaylee thinks that Liam will search for the silver letter opener in the leather briefcase.</answer>"
        # },
        # {
        #     "ground_truth": "yes",
        #     "model_response": "Assistant: <think>Based on the story, Isabella was directly involved in the festival marketing strategy discussions and contributed her ideas. She clearly has knowledge about these strategies.</think><answer>Yes</answer>"
        # },
        {
            "ground_truth": "blue_box",
            "model_response": "<answer>Lily thinks Jayden thinks the peas is in the blue_box</answer><|im_end|>"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest Case {i+1}")
        compute_score(test["model_response"], test["ground_truth"])
