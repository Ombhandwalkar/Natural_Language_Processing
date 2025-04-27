from vllm import LLM,SamplingParams

conversation=[
    {'role':'system','content':'You are helpful assistant'},
    {'role':'user','content':'hello'},
    {'role':'system','content':'Hello, How can I help you today?'},
    {'role':'user','content':'Write an essay about importance of higher education'}
]

sampling_params=SamplingParams(temperature=0.8,top_p=0.,max_tokens=512)

llm=LLM(
     model="/mnt/wolverine/home/samtukra/juan/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf",
    tokenizer="/mnt/wolverine/home/samtukra/juan/models/Meta-Llama-3.1-8B-Instruct-GGUF",
    max_model_len=50000,
    quantization="int8"
)

outputs=llm.chat(conversation,sampling_params)

for output in outputs:
    prompt=output.prompt
    generated_text=output.outputs[0].text
    print('\n')
    print(f"Prompt: {prompt!r},\n\n Generated text: {generated_text!r}")