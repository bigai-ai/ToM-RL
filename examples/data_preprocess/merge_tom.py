"""
Preprocess the Facebook/ExploreToM dataset to parquet format
"""

import os
import datasets
import argparse

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def make_prefix(story, question, template_type='base', add_hint=False, wo_think=False):
    """
    Format the prompt with appropriate instructions based on template type.
    
    Args:
        story: The infilled story text
        question: The question about the story
        template_type: The template format to use ('base' or 'qwen-instruct')
        
    Returns:
        Formatted prompt with instructions
    """
    quiz = f"Read the following story and answer the question. \nStory: {story}\nQuestion: {question}"
    # quiz = story + "\n\n" + question
    
    if template_type == 'base':
        prefix = f"""The user asks a question about a story, and the Assistant answers it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\n\nUser:{quiz}\nAssistant: <think>"""
    # TODO add hints
    elif template_type == 'qwen-instruct':
        if not add_hint:
            if not wo_think:
                prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
            else:
                prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Now the user asks you to solve a theory of mind reasoning problem. Please reason step by step, and put your final answer within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n"""
        else:
            if not wo_think:
                prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.\nNote: You should assume the following.\n(1) An agent witnesses everything and every movement before exiting a room.\n(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, or have private or public interactions.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
            else:
                prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Now the user asks you to solve a theory of mind reasoning problem. Please reason step by step, and put your final answer within <answer> </answer> tags.\nNote: You should assume the following.\n(1) An agent witnesses everything and every movement before exiting a room.\n(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, or have private or public interactions.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n"""
    elif template_type == 'dpsk-reasoning':
        prefix = "<｜begin▁of▁sentence｜><｜User｜>You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a theory of mind reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.<｜Assistant｜><think>"
    return prefix


def tom_make_map_fn():
    def process_fn(example, idx):
        question = make_prefix(
            story=example['story'], 
            question=example['question'],
            template_type=args.template_type,
            add_hint=args.add_hint,
            wo_think=args.wo_think
        )
        solution = example['answer']
        data_source = example['data_source']
        return {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "theory_of_mind",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": example.get('extra_info', {'key': 'dummy'})
        }
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/cleaned_tom')
    parser.add_argument('--hdfs_dir', default=None)
    # random seed
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--template_type', type=str, default='qwen-instruct', 
                        choices=['base', 'qwen-instruct', 'dpsk-reasoning'])
    parser.add_argument('--add_hint', action='store_true', 
                        help='Add hint to the prompt')
    parser.add_argument('--wo_think', action='store_true', 
                        help='Whether to remove the thinking tags')
    args = parser.parse_args()


    # deception,story_length,question_order,sample_id,story,question,choices,answer,question_old,answer_old
    # 3200 = 2000(hi_tom) + 1200(explore_tom)
    train_src = './data/cleaned_tom/merge/ToM_train_HiEx_3200.parquet'
    test_src = './data/cleaned_tom/merge/ToM_test_HiExTi.parquet'

    train_ds = datasets.load_dataset('parquet', data_files=train_src)['train']
    test_ds = datasets.load_dataset('parquet', data_files=test_src)['train']

    print('len of train_dataset:', len(train_ds))
    print('len of test_dataset:', len(test_ds))

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    
    train_ds = train_ds.map(function=tom_make_map_fn(), with_indices=True)   
    test_ds = test_ds.map(function=tom_make_map_fn(), with_indices=True)

    if args.add_hint:
        if args.wo_think:
            train_ds.to_parquet(os.path.join(local_dir, 'ToM_train_HiEx_hint_wo_think.parquet'))
            test_ds.to_parquet(os.path.join(local_dir, 'ToM_test_HiExTi_hint_wo_think.parquet'))
        else:
            train_ds.to_parquet(os.path.join(local_dir, 'ToM_train_HiEx_hint.parquet'))
            test_ds.to_parquet(os.path.join(local_dir, 'ToM_test_HiExTi_hint.parquet'))
    else:
        if args.wo_think:
            train_ds.to_parquet(os.path.join(local_dir, 'ToM_train_HiEx_wo_think.parquet'))
            test_ds.to_parquet(os.path.join(local_dir, 'ToM_test_HiExTi_wo_think.parquet'))
        else:
            train_ds.to_parquet(os.path.join(local_dir, 'ToM_train_HiEx.parquet'))
            test_ds.to_parquet(os.path.join(local_dir, 'ToM_test_HiExTi.parquet'))

    # Create local directory if not exists
    os.makedirs(local_dir, exist_ok=True)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
