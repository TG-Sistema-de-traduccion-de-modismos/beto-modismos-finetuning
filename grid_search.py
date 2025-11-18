import json
import regex
import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from itertools import product
import pandas as pd
from datetime import datetime
import os
import traceback
import multiprocessing


multiprocessing.set_start_method('spawn', force=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

MODELOS = {
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",
    "BERTIN": "bertin-project/bertin-roberta-base-spanish",
    "ALBERTO": "CenIA/albert-base-spanish"
}

GRID_PARAMS = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "num_train_epochs": [3, 5, 8, 10],
    "per_device_train_batch_size": [8, 16],
    "weight_decay": [0.0, 0.01, 0.05],
    "warmup_steps": [0, 50, 100],
    "lr_scheduler_type": ["linear", "cosine"]
}

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
                ejemplos.append({
                    "entrada": marcar_contexto(contexto, raiz),
                    "label": etiquetas[i]
                })
    return ejemplos

def preparar_datos():
    with open("dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    ejemplos = procesar(data)
    
    ejemplos_train, ejemplos_test = train_test_split(
        ejemplos, test_size=0.2, random_state=42,
        stratify=[e["label"] for e in ejemplos]
    )
    
    return ejemplos_train, ejemplos_test

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def entrenar_configuracion(args):
    modelo_nombre, modelo_path, params, ejemplos_train, ejemplos_test, run_id = args
    
    try:
        print(f"\n{'='*60}")
        print(f"Iniciando experimento {run_id}: {modelo_nombre}")
        print(f"Parámetros: {params}")
        print(f"{'='*60}\n")
        
        ds_train = Dataset.from_list(ejemplos_train)
        ds_test = Dataset.from_list(ejemplos_test)
        
        le = LabelEncoder()
        ds_train = ds_train.add_column("label_id", le.fit_transform(ds_train["label"]))
        ds_test = ds_test.add_column("label_id", le.transform(ds_test["label"]))
        
        tokenizer = AutoTokenizer.from_pretrained(modelo_path)
        all_texts = list(ds_train["entrada"]) + list(ds_test["entrada"])
        max_len = min(max(len(tokenizer(x)["input_ids"]) for x in all_texts), 128)
        
        def tokenize(example):
            return tokenizer(
                example["entrada"],
                truncation=True,
                padding="max_length",
                max_length=max_len
            )
        
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
            modelo_path,
            num_labels=len(le.classes_)
        ).to(device)
        
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss
        
        output_dir = f"./runs/{modelo_nombre}_run_{run_id}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="no",  
            num_train_epochs=params["num_train_epochs"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=params["per_device_train_batch_size"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            lr_scheduler_type=params["lr_scheduler_type"],
            warmup_steps=params["warmup_steps"],
            logging_dir=None,
            logging_strategy="no",
            report_to="none",
            seed=42,
            dataloader_num_workers=2,
        )
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_test,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        
        train_result = trainer.train()
        
        eval_result = trainer.evaluate()
        
        resultado = {
            "run_id": run_id,
            "modelo": modelo_nombre,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **params,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "accuracy": eval_result["eval_accuracy"],
            "precision": eval_result["eval_precision"],
            "recall": eval_result["eval_recall"],
            "f1": eval_result["eval_f1"],
            "status": "completado"
        }
        
        print(f"\nExperimento {run_id} completado: F1={eval_result['eval_f1']:.4f}")
        
        del model, trainer, ds_train, ds_test
        torch.cuda.empty_cache()
        
        return resultado
        
    except Exception as e:
        print(f"\nError en experimento {run_id}: {str(e)}")
        traceback.print_exc()
        return {
            "run_id": run_id,
            "modelo": modelo_nombre,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **params,
            "status": "error",
            "error": str(e)
        }

def main():
    print("="*80)
    print("GRID SEARCH PARA MODELOS BERT EN ESPAÑOL")
    print("="*80)
    
    print("\nPreparando datasets...")
    ejemplos_train, ejemplos_test = preparar_datos()
    print(f"Tamaño train: {len(ejemplos_train)}, test: {len(ejemplos_test)}")
    
    param_names = list(GRID_PARAMS.keys())
    param_values = list(GRID_PARAMS.values())
    all_combinations = list(product(*param_values))
    
    max_per_model = 67  
    if len(all_combinations) > max_per_model:
        import random
        random.seed(42)
        all_combinations = random.sample(all_combinations, max_per_model)
    
    print(f"\nTotal de combinaciones por modelo: {len(all_combinations)}")
    print(f"Total de experimentos: {len(all_combinations) * len(MODELOS)}")
    
    configuraciones = []
    run_id = 1
    for modelo_nombre, modelo_path in MODELOS.items():
        for combo in all_combinations:
            params = dict(zip(param_names, combo))
            configuraciones.append((
                modelo_nombre,
                modelo_path,
                params,
                ejemplos_train,
                ejemplos_test,
                run_id
            ))
            run_id += 1
    
    resultados = []
    
    print(f"\nEjecutando {len(configuraciones)} experimentos secuencialmente...")
    print("Esto puede tomar varias horas...\n")
    
    for i, config in enumerate(configuraciones, 1):
        resultado = entrenar_configuracion(config)
        resultados.append(resultado)
        
        if i % 10 == 0:
            df_temp = pd.DataFrame(resultados)
            df_temp.to_excel("resultados_parciales.xlsx", index=False)
            print(f"\n{'='*60}")
            print(f"✓ Progreso: {i}/{len(configuraciones)} completados ({i/len(configuraciones)*100:.1f}%)")
            print(f"{'='*60}\n")
    
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel("resultados_grid_search.xlsx", index=False)
    
    print("\n" + "="*80)
    print("ANÁLISIS DE RESULTADOS")
    print("="*80)
    
    df_ok = df_resultados[df_resultados["status"] == "completado"].copy()
    
    print("\nMEJORES RESULTADOS POR MODELO:\n")
    
    mejores_por_modelo = {}
    for modelo in MODELOS.keys():
        df_modelo = df_ok[df_ok["modelo"] == modelo]
        if len(df_modelo) > 0:
            mejor = df_modelo.loc[df_modelo["f1"].idxmax()]
            mejores_por_modelo[modelo] = mejor
            
            print(f"\n{modelo}:")
            print(f"  F1 Score: {mejor['f1']:.4f}")
            print(f"  Accuracy: {mejor['accuracy']:.4f}")
            print(f"  Precision: {mejor['precision']:.4f}")
            print(f"  Recall: {mejor['recall']:.4f}")
            print(f"  Learning Rate: {mejor['learning_rate']}")
            print(f"  Epochs: {mejor['num_train_epochs']}")
            print(f"  Batch Size: {mejor['per_device_train_batch_size']}")
            print(f"  Weight Decay: {mejor['weight_decay']}")
            print(f"  Warmup Steps: {mejor['warmup_steps']}")
            print(f"  Scheduler: {mejor['lr_scheduler_type']}")
    
    print("\n" + "="*80)
    print("COMPARACIÓN FINAL")
    print("="*80)
    
    df_mejores = pd.DataFrame([mejores_por_modelo[m] for m in mejores_por_modelo.keys()])
    df_mejores = df_mejores.sort_values("f1", ascending=False)
    
    print("\nRanking por F1 Score:")
    for idx, row in df_mejores.iterrows():
        print(f"{row['modelo']:12s} - F1: {row['f1']:.4f}, Acc: {row['accuracy']:.4f}")
    
    df_mejores.to_excel("mejores_modelos.xlsx", index=False)
    
    print("\n" + "="*80)
    print("ESTADÍSTICAS GENERALES")
    print("="*80)
    
    for modelo in MODELOS.keys():
        df_modelo = df_ok[df_ok["modelo"] == modelo]
        if len(df_modelo) > 0:
            print(f"\n{modelo}:")
            print(f"  F1 promedio: {df_modelo['f1'].mean():.4f} ± {df_modelo['f1'].std():.4f}")
            print(f"  F1 máximo: {df_modelo['f1'].max():.4f}")
            print(f"  F1 mínimo: {df_modelo['f1'].min():.4f}")
            print(f"  Experimentos exitosos: {len(df_modelo)}")
    
    print("\nGrid search completado!")
    print(f"Resultados guardados en:")
    print(f"  - resultados_grid_search.xlsx (todos los experimentos)")
    print(f"  - mejores_modelos.xlsx (mejores de cada modelo)")

if __name__ == "__main__":
    main()