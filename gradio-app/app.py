import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="Tirendaz/my_distilbert_model")

def text_classification(text):
    result= classifier(text)
    sentiment_label = result[0]['label']
    sentiment_score = result[0]['score']
    formatted_output = f"This sentiment is {sentiment_label} with the probability {sentiment_score*100:.2f}%"
    return formatted_output

examples=["This is wonderful movie!", "The movie was really bad; I didn't like it."]

io = gr.Interface(fn=text_classification, 
                         inputs= gr.Textbox(lines=2, label="Text", placeholder="Enter title here..."), 
                         outputs=gr.Textbox(lines=2, label="Text Classification Result"),
                         title="Text Classification",
                         description="Enter a text and see the text classification result!",
                         examples=examples)

io.launch()

