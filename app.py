from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, LogitsProcessorList, BeamSearchScorer, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, HammingDiversityLogitsProcessor, StoppingCriteriaList, MaxLengthCriteria
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    print("loading to CPU...")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype="auto").half()
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")
    
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    
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
        length = model_inputs.get('length', 50)
        temperature = model_inputs.get('temperature', 0.9)
        top_p = model_inputs.get('topP', 0.9)
        repetition_penalty = model_inputs.get('repetitionPenalty', 1.0)
        
        if prompt == "KEEPALIVE" and length == 1:
            return {"output": "keeping alive"}

        # Tokenize inputs
        input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        #bad_words_ids = [
        #    tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ["??", "???", "????", "?????", "??????", "???????", "????????", "anal", "arse", "ass", "bitch", "boner", "dick", "dildo", "nigga", "nigge", "penis", "pussy", "vagina"]
        #]

        # Run the model
        output = model.generate(input_tokens, use_cache=True, do_sample=True, temperature=temperature, max_new_tokens=length, top_p=top_p, repetition_penalty=repetition_penalty)

        # Decode output tokens
        output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

        result = {"output": output_text}

        # Return the results as a dictionary
        return result
    except BaseException as skill_issue:
        return {"skill_issue": str(skill_issue)}
