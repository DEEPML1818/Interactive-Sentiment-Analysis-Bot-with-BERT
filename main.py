from transformers import pipeline

def analyze_sentiment_bert(text):
    classifier = pipeline('sentiment-analysis')
    result = classifier(text)
    
    # Get the label ('POSITIVE' or 'NEGATIVE') from the result
    label = result[0]['label']
    
    # Convert label to 'Positive' or 'Negative'
    sentiment = 'Positive' if label == 'POSITIVE' else 'Negative'
    
    return sentiment


def main():
    print("Sentiment Analysis Bot with BERT")
    print("---------------------------------")
    
    while True:
        # Get user input
        user_input = input("Enter a sentence (or type 'exit' to quit): ")
        
        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Exiting the Sentiment Analysis Bot. Goodbye!")
            break
        
        # Analyze sentiment using BERT
        sentiment = analyze_sentiment_bert(user_input)
        
        # Display the result
        print(f"Sentiment: {sentiment}")
        print("---------------------------------")

if __name__ == "__main__":
    main()
