from time import sleep

from openai import OpenAI

from typing import Dict, List

from processing.segmentation import (
    CAM_INDEX_TO_PATTERN,
    ANG_INDEX_TO_PATTERN,
    find_consecutive_chunks,
)


# ------------------------------------------------------------------------------------- #

MAX_TRIALS = 3

# ------------------------------------------------------------------------------------- #

api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key

client = OpenAI(api_key=api_key)

def call_gpt4_v(user_prompt, modelname, max_tokens=700):
    conversation_history = [({"role": "user",
                                 "content": [
                                     {"type": "text", "text": user_prompt},
                                 ]
                                 }
                                )]
    response = client.chat.completions.create(
        model=modelname,
        messages=conversation_history,
        max_tokens=max_tokens,
    )
    return response


def show_content(response):
    print(response.choices[0].message.content)

# def save_content(response, file):
#     with open(file, 'w') as f:
#         f.write(response.choices[0].message.content)

def single_test(prompt_text, modelname):            
    for k in range(MAX_TRIALS):
        try:
            response = call_gpt4_v(prompt_text, modelname)
            caption = response.choices[0].message.content
            if any(char.isdigit() for char in caption):
                assert False, f"{caption}\n"
            break
        except Exception as e:
            print(f"Frame Error: {e}\n")
            sleep(1)
            continue
    return response.choices[0].message.content 

def get_full_description(
    segment_index: int, start: int, end: int, index_to_move: Dict[int, str], index_to_angular: Dict[int, str]
) -> str:
    cam_index = segment_index//7
    ang_index = segment_index%7
    if index_to_angular[ang_index] == "static":
        return f"Between frames {start} and {end}: {index_to_move[cam_index]}"
    else:
        return f"Between frames {start} and {end}: {index_to_move[cam_index]} + {index_to_angular[ang_index]}"

def get_caption_prompt(
    traj_description: str,
    context_prompt: str,
    instruction_prompt: str,
    constraint_prompt: str,
    demonstration_prompt: str,
) -> str:
    raw_prompt = (
        f"{context_prompt}\n{instruction_prompt}\n{constraint_prompt}"
        f"{demonstration_prompt}{traj_description}\nDescription: "
    )
    return raw_prompt

# ------------------------------------------------------------------------------------- #
def has_consecutive_nonzero_same_numbers(angular_list):
    for i in range(1, len(angular_list)):
        if angular_list[i] == angular_list[i - 1] and angular_list[i] != 0:
            return True
    return False

def caption_trajectories(
    cam_segments: List[int],
    context_prompt: str,
    instruction_prompt: str,
    constraint_prompt: str,
    demonstration_prompt: str,
) -> str:
    # # Find consecutive chunks of patterns
    cam_chunks = find_consecutive_chunks(cam_segments)
    
    # Describe each chunk and join them
    cam_description = []
    angular_list = []
    for index, start, end in cam_chunks:
        description = get_full_description(index, start, end, CAM_INDEX_TO_PATTERN, ANG_INDEX_TO_PATTERN)
        angular_list.append(index%7)
        cam_description.append(description)

    traj_description = (
        f"\n\nOutline: Total frames {end+1}. "
        + "\n[Camera motion] "
        + "; ".join(cam_description)
        + ". "
    )
    
    # # Prompt the chatbot with the trajectory description
    caption_prompt = get_caption_prompt(
        traj_description,
        context_prompt,
        instruction_prompt,
        constraint_prompt,
        demonstration_prompt,
    )
    
    if has_consecutive_nonzero_same_numbers(angular_list):
        print("Consecutive same numbers")
        caption = single_test(caption_prompt, "gpt-4o")
    else:
        print("No consecutive same numbers")
        caption = single_test(caption_prompt, "gpt-4o-mini")
    print(caption)

    return caption
