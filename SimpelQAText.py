KnowledgeBase = """LangChain is a framework designed to facilitate the development of applications that leverage large language models (LLMs). It provides tools and abstractions to make it easier to build, customize, and deploy applications that integrate LLMs, such as GPT, into workflows, data pipelines, and other systems.

LangChain has 7 key features:
1. Integration with LLMs: It supports integration with various LLMs, including OpenAI's GPT, Hugging Face models, and others.
2. Prompt Management: Helps design, optimize, and manage prompts for interacting with LLMs.
3. Memory: Enables LLMs to retain context across interactions, which is useful for conversational applications.
4. Chains: Allows chaining multiple LLM calls or other operations to create complex workflows.
5. Agents: Provides tools for building autonomous agents that can make decisions, interact with external tools, and perform tasks dynamically.
6. Data Augmentation: Integrates with external data sources to enhance LLM outputs with relevant information.
7. Customization: Offers flexibility to tailor LLM behavior for specific use cases."""

from transformers import pipeline

# Loading a Question/Answers Pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_question(question, context=KnowledgeBase):
    response = qa_pipeline(question=question, context=context)
    answer = response['answer']
    score = response['score']
    return answer, score

def main():
    print("Welcome to the Question and answer session, QA_Bot")
    while True:
        question = input("Please enter your question or type 'exit' to quit: ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        else:
            answer, score = answer_question(question)
            print(f"Answer: {answer}")
            print(f"Confidence: {score:.2f}")
            print("\n")

main()