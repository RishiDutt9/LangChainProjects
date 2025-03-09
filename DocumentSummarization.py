from transformers import pipeline 

# Load the Summarization pipeline
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    summary = summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

text = """LangChain is a framework to build and deploy language models. 
It is designed to be easy to use and flexible. 
The framework is built on top of the Hugging Face Transformers library. 
It provides a simple API to train, fine-tune, 
and deploy models. LangChain is open-source and can be used for a wide 
range of natural language processing tasks. It is designed to be fast,
 efficient, and scalable. The framework is built with performance in mind and 
 can handle large datasets with ease. LangChain is designed to be easy to use and flexible.
   It is built on top of the Hugging Face Transformers library. The framework provides a simple API to train, fine-tune,
     and deploy models. LangChain is open-source and can be used for a wide range of natural language processing tasks.
       It is designed to be fast, efficient, and scalable. The framework is built with performance in mind and can handle 
       large datasets with ease."""

summary = summarize_text(text)
print("Original Text before Summarizing:", text)
print("Summary of the text:", summary)


