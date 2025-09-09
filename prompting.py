def build_prompt(context: str, query: str, examples: list = None, instructions: str = None) -> str:
    prompt = ''
    if instructions:
        prompt += instructions + "\n"
    if examples:
        prompt += "Examples:\n"
        for ex in examples:
            prompt += f"Q: {ex['query']}\nA: {ex['answer']}\n"
        prompt += "---\n"
    prompt += "Context:\n" + context + "\n---\n"
    prompt += f"Customer Query: {query}\nAgent Answer:"
    return prompt
