from fastapi import FastAPI,Request, Form
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from lime.lime_text import LimeTextExplainer
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

#Configuration
MODEL_PATH = "/mnt/c/ai_detection/saved_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

#Define FastAPI
app = FastAPI(title="AI Text Detection API with LIME")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

#Health Check
@app.get("/health")
def health_check():
    return {"status": "ok"}

#Request Body
class InputText(BaseModel):
    paragraph: str
    explain: bool = False

#Prediction function
def predict_text(paragraph: str, chunk_size: int = 128):
    # Tokenize paragraph into smaller parts
    tokens = tokenizer.tokenize(paragraph)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    all_probs = []
    label_map = {0: "Human", 1: "AI-generated"}  # adjust to your label encoding

    for chunk in chunks:
        text_chunk = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(
            text_chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=chunk_size
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu())

    # Average the probabilities from all chunks
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    confidence, predicted_class = torch.max(avg_probs, dim=1)

    return {
        "label": label_map[predicted_class.item()],
        "confidence": float(confidence.item())
    }

#LIME explanation
def explain_text_with_lime(paragraph: str, num_features: int = 10):
    label_map = {0: "Human", 1: "AI-generated"}
    explainer = LimeTextExplainer(class_names=list(label_map.values()))
    # Prediction function wrapper for LIME
    def predict_proba(texts):
        all_probs = []
        for t in texts:
            inputs = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy().flatten())
        return np.array(all_probs)
    # Explain instance
    explanation = explainer.explain_instance(
        paragraph,
        predict_proba,
        num_features=num_features
    )
    # Generate HTML representation
    html_data = explanation.as_html()
    return html_data

#Prediction endpoint
@app.post("/predict")
def predict(input_text: InputText):
    result = predict_text(input_text.paragraph)
    return {"result": result}

#LIME endpoint
@app.post("/explain", response_class=HTMLResponse)
def explain(input_text: InputText):
    html_explanation = explain_text_with_lime(input_text.paragraph)
    file_path = "lime_explanation.html"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_explanation)

    return HTMLResponse(content=f"""
        <h3>LIME explanation saved!</h3>
        <p>Open <a href="http://127.0.0.1:8000/static/lime_explanation.html" target="_blank">LIME Visualization</a></p>
    """)
# Note - Open the HTML file to view the LIME explanation

#To open the home page
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

#To open the results page
@app.post("/result", response_class=HTMLResponse)
def show_result(request: Request, paragraph: str = Form(...)):
    # Run prediction
    result = predict_text(paragraph)
    html_explanation = explain_text_with_lime(paragraph)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "paragraph": paragraph,
        "label": result["label"],
        "confidence": f"{result['confidence']*100:.2f}%",
        "lime_html": html_explanation
    })

