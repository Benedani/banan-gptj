from transformers import GPTJForCausalLM, AutoTokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    print("loading to CPU...")
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).half()
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    # we only do inference here
    torch.no_grad()


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    try:
        # Parse out your arguments
        prompt = model_inputs.get('prompt', "")
        temperature = model_inputs.get('temperature', 0.9)
        top_p = model_inputs.get('topP', 0.9)
        repetition_penalty = model_inputs.get('repetitionPenalty', 1.0)

        # Funny steps system
        # The bot generates `tokens_per_step` tokens, then checks whether it stopped generating text for the bot or not.
        # This is repeated `steps` times
        tokens_per_step = model_inputs.get('tokensPerStep', 10)
        steps = model_inputs.get('steps', 5)
        line_start = model_inputs.get('lineStart', "Phineas ")

        if prompt == "KEEPALIVE" and length == 1:
            return {"output": "keeping alive"}

        #bad_words_ids = [
        #    tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ["??", "???", "????", "?????", "??????", "???????", "????????", "anal", "arse", "ass", "bitch", "boner", "dick", "dildo", "nigga", "nigge", "penis", "pussy", "vagina"]
        #]

        # Run the model
        current_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_text = prompt

        for _i in range(steps):
            current_tokens = model.generate(current_tokens, use_cache=True, do_sample=True, temperature=temperature, max_new_tokens=tokens_per_step, top_p=top_p, repetition_penalty=repetition_penalty)

            # Decode output tokens
            output_text = tokenizer.batch_decode(current_tokens, skip_special_tokens = True)[0]

            # Should we exit?
            ln = output_text.splitlines()[-1]
            if len(ln) >= len(line_start) and not ln.startswith(line_start):
                # ok it's generating something unrelated now
                break

        result = {"output": output_text}

        # Return the results as a dictionary
        return result
    except BaseException as skill_issue:
        return {"skill_issue": str(skill_issue)}
