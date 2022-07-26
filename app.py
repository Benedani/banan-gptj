from transformers import GPTJForCausalLM, GPT2Tokenizer, LogitsProcessorList, BeamSearchScorer, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, HammingDiversityLogitsProcessor, StoppingCriteriaList, MaxLengthCriteria
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    print("loading to CPU...")
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")


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
        diversity_penalty = model_inputs.get('diversityPenalty', 1.0)

        # Tokenize inputs
        input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=20,
            device=device,
            num_beam_groups=5,
        )

        # instantiate logits processors
        logits_processor = LogitsProcessorList(
        [
            TopPLogitsWarper(top_p=top_p),
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
            TemperatureLogitsWarper(temperature=temperature),
            HammingDiversityLogitsProcessor(diversity_penalty=diversity_penalty,num_beams=20,num_beam_groups=5)
        ]
        )

        # Run the model
        output = model.group_beam_search(input_tokens,beam_scorer,logits_processor,stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=length)]))
        # Decode output tokens
        output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

        result = {"output": output_text}

        # Return the results as a dictionary
        return result
    except BaseException as skill_issue:
        return {"skill_issue": str(skill_issue)}
