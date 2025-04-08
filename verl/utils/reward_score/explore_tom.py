import re
from typing import Dict, Tuple, Optional, Union, Any

# for dpsk
# def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
#     """Extracts the final answer from the model's response string.
    
#     Args:
#         solution_str: Raw response string from the language model
        
#     Returns:
#         Tuple containing (extracted_answer, processed_string)
#     """
#     # Split response to isolate assistant output
#     if "<｜Assistant｜>" in solution_str:
#         processed_str = solution_str.split("<｜Assistant｜>", 1)[1]
#     else:
#         print("[Error] Failed to locate model response header")
#         return None, solution_str

#     # Extract final answer using XML-style tags
#     answer_pattern = r'<answer>(.*?)</answer>'
#     matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
#     if not matches:
#         print("[Error] No valid answer tags found")
#         return None, processed_str
        
#     final_answer = matches[-1].group(1).strip()
#     return final_answer, processed_str

# for qwen
def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
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

    # _ => ' '
    normalized = re.sub(r'_', ' ', normalized)
    return normalized

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

# (-2, -1.5, 2)
# def check_answer_correctness(predicted_answer: str, ground_truth: str) -> Tuple[bool, float]:

#     # ans_pattern = r".*?" + re.escape(norm_answer) + r"\s*\b$"

#     """Checks if the predicted answer matches the ground truth.
    
#     Args:
#         predicted_answer: The answer extracted from model's response
#         ground_truth: The ground truth answer
        
#     Returns:
#         Tuple containing (is_correct, score)
#     """
#     print("\n[Answer Validation]")
#     print(f"  Ground truth: '{ground_truth}'")
#     print(f"  Predicted: '{predicted_answer}'")
    
#     # Normalize both answers for better comparison
#     norm_pred = normalize_answer(predicted_answer)
#     norm_truth = normalize_answer(ground_truth)
    
#     print(f"  Normalized ground truth: '{norm_truth}'")
#     print(f"  Normalized prediction: '{norm_pred}'")
    
#     # Check exact match after normalization
#     if norm_pred == norm_truth:
#         print("  Answer validation: EXACT MATCH")
#         return True, 2.0
    
#     # Check if ground truth is a choice and prediction contains the correct choice
#     if ' / ' in ground_truth:
#         choices = [normalize_answer(choice) for choice in ground_truth.split(' / ')]
#         print(f"  Multiple choice options: {choices}")
        
#         for choice in choices:
#             if choice in norm_pred:
#                 print(f"  Answer validation: MATCH (contains correct choice: '{choice}')")
#                 return True, 2.0
    
#     # Check if prediction is in list of acceptable answers
#     # This could be extended with domain-specific lists of equivalent answers
#     if norm_pred in norm_truth or norm_truth in norm_pred:
#         print(f"  Answer validation: Partial MATCH (contains correct choice: {norm_pred}({norm_truth}))")
#         return False, -1.5
#     print("  Answer validation: MISMATCH")
#     return False, -2.0

def check_answer_correctness(predicted_answer: str, ground_truth: str) -> Tuple[bool, float]:

    # ans_pattern = r".*?" + re.escape(norm_answer) + r"\s*\b$"

    """Checks if the predicted answer matches the ground truth.
    
    Args:
        predicted_answer: The answer extracted from model's response
        ground_truth: The ground truth answer
        
    Returns:
        Tuple containing (is_correct, score)
    """
    print("\n[Answer Validation]")
    print(f"  Ground truth: '{ground_truth}'")
    print(f"  Predicted: '{predicted_answer}'")
    
    # Normalize both answers for better comparison
    norm_pred = normalize_answer(predicted_answer)
    norm_truth = normalize_answer(ground_truth)
    
    print(f"  Normalized ground truth: '{norm_truth}'")
    print(f"  Normalized prediction: '{norm_pred}'")
    
    # Check exact match after normalization
    if norm_pred == norm_truth:
        print("  Answer validation: EXACT MATCH with strict format")
        return True, 2.0
    
    ans_pattern = r".*?" + re.escape(norm_truth) + r"\s*\b$"
    if re.match(ans_pattern, norm_pred):
        print("  Answer validation: EXACT MATCH with loose format")
        return True, 2
    
    print("  Answer validation: MISMATCH")
    return False, -2.0

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
    print(" Processing Theory of Mind Sample ".center(80, '='))
    
    # Extract ground truth
    if isinstance(ground_truth, dict):
        gt_answer = ground_truth.get('expected_answer', '')
    else:
        gt_answer = ground_truth
    
    print(f"[Ground Truth] Expected answer: {gt_answer}")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    if format_correct and answer_text:
        is_correct, score_value = check_answer_correctness(answer_text, gt_answer)
        answer_score = score_value
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score


if __name__ == "__main__":
    # Test cases for different types of ExploreToM answers
    test_cases = [
        {
            "ground_truth": "does not know about it",
            "model_response": "Assistant: <think>Let me reason through this step by step. Isabella doesn't have direct knowledge of Colton's belief about festival marketing strategies because they haven't communicated about it. Isabella has her own understanding, but without explicit communication with Colton, she cannot know his beliefs on the matter.</think><answer>Isabella does not know about it</answer>"
        },
        {
            "ground_truth": "leather briefcase",
            "model_response": "Assistant: <think>I need to trace Kaylee's understanding of Liam's belief. Since Liam saw the silver letter opener being moved to the leather briefcase, but Kaylee doesn't know this, she would think Liam still believes it's in the original location.</think><answer>Kaylee thinks that Liam will search for the silver letter opener in the leather briefcase.</answer>"
        },
        {
            "ground_truth": "yes",
            "model_response": "Assistant: <think>Based on the story, Isabella was directly involved in the festival marketing strategy discussions and contributed ,her ideas. She clearly has knowledge about these strategies.</think><answer>Yes</answer>"
        }, 
        {
            "ground_truth": "green_pantry",
            "model_response": "Assistant: <think></think><answer>Emma thinks Aria thinks Isabella thinks Ethan thinks the tangerine is in the green pantry.</answer>"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest Case {i+1}")
        compute_score(test["model_response"], test["ground_truth"])
