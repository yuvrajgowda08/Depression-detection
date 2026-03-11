import ollama

def explain_prediction(text, prediction):

    label = "Depression Risk" if prediction == 1 else "No Depression Risk"

    prompt = f"""
A mental health AI predicted: {label}

User text:
{text}

Explain in a supportive and helpful way why this text might indicate emotional distress.
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]