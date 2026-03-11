import ollama

def chatbot_reply(message):

    prompt = f"""
    You are a supportive mental wellness assistant.
    Reply in a kind, optimistic and encouraging way.

    User: {message}
    """

    response = ollama.chat(
        model="llama3",
        messages=[{"role":"user","content":prompt}]
    )

    return response["message"]["content"]