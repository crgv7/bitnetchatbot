# Path to your model weights
local_model = "/workspaces/bitnetchatbot/gemma/BitNet-b1.58-2B-4T/gemma-3-1b-it-q4_0.gguf"

import multiprocessing

from langchain_community.chat_models import ChatLlamaCpp

llm = ChatLlamaCpp(
    temperature=0.5,
    model_path=local_model,
    n_ctx=10000,
    n_gpu_layers=8,
    n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    max_tokens=512,
    n_threads=multiprocessing.cpu_count() - 1,
    repeat_penalty=1.5,
    top_p=0.5,
    verbose=True,
)

messages = [
    (
        "system",
        "Eres un asistente útil que ayuda a los usuarios a encontrar información y resolver problemas.",
    ),
    ("human", "Que es la inteligencia artificial?"),
]

ai_msg = llm.invoke(messages)
ai_msg

print(ai_msg.content)