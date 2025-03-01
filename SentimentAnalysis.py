from transformers import pipeline

#Load the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    #Analyze the sentiment of the text
    response = sentiment_analysis(text)
    sentiment = response[0]['label']
    confidence = response[0]['score']
    return sentiment, confidence




def main():
    print("Welcome to the Sentiment session, Sneti_Bot")
    while True:
        text = input("Please enter your sentence to check the Sentiment or type 'exit' to quit: ")
        if text.lower() == 'exit':
            print("Goodbye!")
            break
        else:
            sentiment, confidence = analyze_sentiment(text)
            print(f"Sentiment for the Provided text : {text} is : {sentiment}")
            print(f"Confidence: {confidence:.2f}")
            print("\n")

main()
