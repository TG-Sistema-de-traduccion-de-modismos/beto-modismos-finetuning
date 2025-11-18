import json
import regex
import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def marcar_contexto(contexto, raiz):
    patron = regex.compile(rf"\b{raiz}\p{{L}}*\b", regex.IGNORECASE)
    return patron.sub(lambda m: f"[TGT] {m.group(0)} [TGT]", contexto, 1)

def procesar(data):
    ejemplos = []
    for entrada in data:
        raiz = entrada["raiz"]
        contextos = entrada["contextos"]
        etiquetas = entrada["etiquetas"]

        for i, contexto in enumerate(contextos):
            if i < len(etiquetas):
                etiqueta = etiquetas[i]
                ejemplos.append({
                    "entrada": marcar_contexto(contexto, raiz),
                    "label": etiqueta
                })
    return ejemplos

ejemplos = procesar(data)

ejemplos_train, ejemplos_test = train_test_split(
    ejemplos,
    test_size=0.2,
    random_state=42,
    stratify=[e["label"] for e in ejemplos]
)

ds_train = Dataset.from_list(ejemplos_train)
ds_test = Dataset.from_list(ejemplos_test)

le = LabelEncoder()
ds_train = ds_train.add_column("label_id", le.fit_transform(ds_train["label"]))
ds_test = ds_test.add_column("label_id", le.transform(ds_test["label"]))

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

all_texts = list(ds_train["entrada"]) + list(ds_test["entrada"])
max_len = min(max(len(tokenizer(x)["input_ids"]) for x in all_texts), 128)

def tokenize(example):
    return tokenizer(example["entrada"], truncation=True, padding="max_length", max_length=max_len)

ds_train = ds_train.map(tokenize)
ds_test = ds_test.map(tokenize)

ds_train = ds_train.remove_columns("label").rename_column("label_id", "label")
ds_test = ds_test.remove_columns("label").rename_column("label_id", "label")

class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(ds_train["label"]),
    y=ds_train["label"]
)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

model = AutoModelForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased",
    num_labels=len(le.classes_)
).to(device)

def custom_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
    loss = loss_fct(logits, labels)
    return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./beto-wsd",
    
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    num_train_epochs=8,                 
    per_device_train_batch_size=8,      
    per_device_eval_batch_size=8,
    learning_rate=2e-05,               
    weight_decay=0.01,                
    lr_scheduler_type="linear",        
    warmup_steps=50,                    

    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    report_to="all",

    seed=42,
    dataloader_num_workers=2,
    run_name="beto-wsd-training"
)

def compute_metrics(pred):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
