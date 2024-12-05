from fastapi import FastAPI, HTTPException
from schemas import Prompt
from imp.les2 import base_model, tokenizer, model, device, new_update


app = FastAPI()


@app.post("/generate")
def create(request: Prompt):
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)

        # Generate response
        outputs = model.generate(
            inputs["input_ids"],
            max_length=request.max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
