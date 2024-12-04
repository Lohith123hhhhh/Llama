import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline, logging
from peft import LoraConfig
from trl import SFTTrainer

base_model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
guanaco_dataset = "mlabonne/guanaco-llama2-1k"
new_model = "llama-1.1B-chat-guanaco"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset(guanaco_dataset, split="train")
model = AutoModelForCausalLM.from_pretrained(base_model)
model.to(device)
model.config.use_cache = True
model.config.pretraining_tp = 1 # basically the number of systems that can work parllely

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # padding the sequences4
tokenizer.padding_side = "right"


# run inference call
logging.set_verbosity(logging.CRITICAL)
prompt = "Who is Napoleon Bonaparte?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer,max_length=200)
result = pipe(f"{prompt}")
print(result[0]["generated_text"])

# training with LoRA

peft_params = LoraConfig(lora_alpha=16,  # this helps in controlling the scaling of the updates
                         lora_dropout=0.1, # helps in convergence , with probability of 10% will set random Lora output to 0
                         r=64,  #rank of Lora so matrices will either have LHS OR RHS dimension of 64
                         bias="none",  # no bias term
                         task_type="CAUSAL_LLM"
)
training_params = TrainingArguments(output_dir='./results',
                                    num_train_epochs=2, #2 passes over the datset
                                    per_device_train_batch_size=2, # mbs = 2
                                    gradient_accumulation_steps=16, # effective batch size = 16*2
                                    optim="adamw_torch",
                                    save_steps=25,# creates checkpoint every 25 steps
                                    logging_steps=1,
                                    learning_rate=2e-4,
                                    weight_decay=0.01,
                                    fp16=True, # 16 bit
                                    bf16=False,
                                    max_grad_norm=0.3, # improves convergence
                                    max_steps=1,
                                    warmup_ratio=0.03, # learning rate warmup
                                    group_by_length=True,
                                    lr_scheduler_type="cosine",
)
trainer = SFTTrainer(model=model,
                     train_dataset=dataset,
                     tokenizer=tokenizer,
                     peft_config=peft_params, # parameter efficient fine tuning
                     args=training_params,
                     dataset_text_field="text",
                     max_seq_length=None,
                     packing=False,

 )
import gc # garbage collectiion
gc.collect()
torch.cuda.empty_cache() # clean the cache

trainer.train()   # train the model
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
prompt = "Who is Napoleon Bonaparte?"
pipe = pipeline(task="text-generation", model=new_model, tokenizer=tokenizer,)
result = pipe(f"{prompt}")
print(result[0]["generated_text"])