# BETO Fine-tuned para Desambiguación de Modismos

Este proyecto implementa un sistema de desambiguación léxica (Word Sense Disambiguation) para modismos en español, utilizando el modelo BETO (BERT en español) fine-tuneado con contextos marcados.

##  Descripción

El modelo identifica el sentido correcto de modismos en diferentes contextos, marcando la palabra objetivo con tokens especiales `[TGT]` para ayudar al modelo a enfocarse en la palabra que necesita desambiguar.

## Modelo Pre-entrenado

El modelo fine-tuneado está disponible en Hugging Face:

**[pescobarg/BETO-finetuned-modismos](https://huggingface.co/pescobarg/BETO-finetuned-modismos)**

Modelo base utilizado: [dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)

##  Características

- Fine-tuning de BETO para desambiguación léxica
- Marcado automático de palabras objetivo con tokens `[TGT]`
- Balanceo de clases mediante pesos computados
- Soporte para entrenamiento con GPU (CUDA)
- Configuración optimizada con early stopping
- Métricas completas: accuracy, precision, recall y F1-score

##  Requisitos

- Docker con soporte NVIDIA GPU
- CUDA 12.8
- Dataset en formato JSON (`dataset.json`)

### Estructura del Dataset

```json
[
  {
    "raiz": "palabra_raiz",
    "contextos": [
      "Contexto 1 con la palabra_raiz en uso",
      "Contexto 2 con otra palabra_raiz"
    ],
    "etiquetas": ["sentido_1", "sentido_2"]
  }
]
```

##  Instalación y Uso

### 1. Construir la imagen Docker

```bash
docker build -t beto-train .
```

### 2. Ejecutar el entrenamiento

```bash
docker run --gpus all -it --name beto-run beto-train
```

### 3. Acceder al contenedor (opcional)

```bash
docker exec -it beto-run bash
```

##  Configuración del Entrenamiento

- **Épocas**: 8
- **Batch size**: 8
- **Learning rate**: 2e-5
- **Weight decay**: 0.01
- **Warmup steps**: 50
- **Max length**: 128 tokens (o longitud máxima del dataset)
- **Estrategia de evaluación**: Por época
- **Split**: 80% entrenamiento, 20% test

##  Estructura del Proyecto

```
.
├── Dockerfile              # Configuración del contenedor
├── requirements.txt        # Dependencias Python
├── train.py               # Script de entrenamiento
├── dataset.json           # Dataset de entrenamiento
└── README.md              # Este archivo
```

##  Dependencias Principales

- `transformers`: Framework para modelos de lenguaje
- `torch`: Framework de deep learning
- `datasets`: Manejo de datasets
- `scikit-learn`: Preprocesamiento y métricas
- `regex`: Procesamiento de texto con Unicode

##  Métricas de Evaluación

El modelo se evalúa con las siguientes métricas:

- **Accuracy**: Precisión general
- **Precision**: Precisión ponderada por clase
- **Recall**: Recall ponderado por clase
- **F1-score**: Media armónica ponderada

##  Resultados

El modelo guarda automáticamente:

- Checkpoints cada época en `./beto-wsd/`
- Logs de entrenamiento en `./logs/`
- Mejor modelo según F1-score

##  Funcionamiento

1. **Marcado de Contexto**: Las palabras objetivo se marcan automáticamente con `[TGT]` usando expresiones regulares Unicode-aware
2. **Tokenización**: Los contextos se tokenizan con el tokenizer de BETO
3. **Balanceo**: Se calculan pesos de clase para manejar desbalanceo
4. **Fine-tuning**: El modelo se entrena con CrossEntropyLoss ponderado
5. **Evaluación**: Se selecciona el mejor modelo según F1-score en el set de validación

## Notas

- El modelo utiliza una longitud máxima adaptativa basada en el dataset
- Se implementa estratificación en el split para mantener distribución de clases
- Compatible con CUDA 12.8 y cuDNN
- El entrenamiento requiere GPU NVIDIA

---

**Desarrollado para desambiguación de modismos en español con BETO**
