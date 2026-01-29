from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss
)
import time
import psutil
import os

# --- Función de métricas mejorada ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Métricas básicas
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    # Precisión, Recall y F1 por clase
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(labels, predictions)
    
    # Log Loss (requiere probabilidades)
    from scipy.special import softmax
    probs = softmax(logits, axis=-1)
    logloss = log_loss(labels, probs)
    
    # ROC-AUC (multiclase one-vs-rest)
    try:
        roc_auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_neg": float(precision_per_class[0]),
        "precision_neu": float(precision_per_class[1]),
        "precision_pos": float(precision_per_class[2]),
        "recall_neg": float(recall_per_class[0]),
        "recall_neu": float(recall_per_class[1]),
        "recall_pos": float(recall_per_class[2]),
        "f1_neg": float(f1_per_class[0]),
        "f1_neu": float(f1_per_class[1]),
        "f1_pos": float(f1_per_class[2]),
        "log_loss": logloss,
        "roc_auc": roc_auc,
    }

# --- Función de entrenamiento y evaluación con métricas de rendimiento ---
def train_and_evaluate(model_name, train_dataset, test_dataset):
    print(f"\n{'='*60}")
    print(f"Procesando modelo: {model_name}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["news_headline"],
            padding=True,
            truncation=True,
            max_length=128
        )

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # NEG, NEU, POS
    )

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name.replace('/', '_')}",
        eval_strategy="epoch",
        save_strategy="no",  # No guardar checkpoints para ahorrar espacio
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Entrenamiento
    print(f"Iniciando entrenamiento...")
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    # Evaluación
    print(f"Iniciando evaluación...")
    eval_start = time.time()
    metrics = trainer.evaluate()
    eval_time = time.time() - eval_start
    
    # Medición de tiempo de inferencia y uso de memoria
    print(f"Calculando tiempo de inferencia...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Medir tiempo de inferencia en todo el conjunto de test
    num_iterations = 5
    inference_times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = trainer.predict(tokenized_test)
        total_time = (time.time() - start) * 1000  # ms
        # Tiempo promedio por muestra
        inference_times.append(total_time / len(tokenized_test))
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Agregar métricas de rendimiento
    metrics['train_time_seconds'] = train_time
    metrics['eval_time_seconds'] = eval_time
    metrics['inference_time_ms_mean'] = np.mean(inference_times)
    metrics['inference_time_ms_std'] = np.std(inference_times)
    metrics['memory_mb'] = mem_after
    metrics['memory_increase_mb'] = mem_after - mem_before
    
    print(f"\nTiempo de entrenamiento: {train_time:.2f}s")
    print(f"Tiempo de evaluación: {eval_time:.2f}s")
    print(f"Tiempo de inferencia (media): {np.mean(inference_times):.2f}ms ± {np.std(inference_times):.2f}ms")
    print(f"Uso de memoria: {mem_after:.2f} MB")
    
    return metrics

# --- Función principal ---
def main():
    # Cargar dataset
    dataset = load_dataset("prithvi1029/sentiment-analysis-for-financial-news")
    print("Filas del dataset original:", dataset["train"].num_rows)

    # Limitar a 1000 muestras
    dataset_small = dataset["train"].select(range(1000))
    print("Filas del dataset limitado a 1000:", len(dataset_small))

    # División entrenamiento/prueba 80/20
    dataset_split = dataset_small.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset_split["train"].rename_column("sentiment", "labels")
    test_dataset = dataset_split["test"].rename_column("sentiment", "labels")

    # Mapeo de etiquetas a números
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    train_dataset = train_dataset.map(
        lambda batch: {"labels": [label_map[l] for l in batch["labels"]]},
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda batch: {"labels": [label_map[l] for l in batch["labels"]]},
        batched=True
    )

    print("Columnas del dataset tras renombrar:", train_dataset.column_names)

    # Evaluar BERT
    print("\n🔹 Evaluando BERT...")
    bert_metrics = train_and_evaluate("bert-base-uncased", train_dataset, test_dataset)
    
    # Evaluar DistilBERT
    print("\n🔹 Evaluando DistilBERT...")
    distilbert_metrics = train_and_evaluate("distilbert-base-uncased", train_dataset, test_dataset)
    
    # Evaluar FinBERT
    print("\n🔹 Evaluando FinBERT...")
    finbert_metrics = train_and_evaluate("yiyanghkust/finbert-tone", train_dataset, test_dataset)

    # Resultados Comparativos
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO DE RESULTADOS")
    print("="*80)
    
    models_results = {
        "BERT": bert_metrics,
        "DistilBERT": distilbert_metrics,
        "FinBERT": finbert_metrics
    }
    
    # Tabla comparativa
    print("\n📊 TABLA COMPARATIVA DE MÉTRICAS PRINCIPALES")
    print("-" * 80)
    print(f"{'Modelo':<15} {'Accuracy':<10} {'F1-Score':<10} {'Log Loss':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    for model_name, metrics in models_results.items():
        print(f"{model_name:<15} {metrics['eval_accuracy']:<10.4f} {metrics['eval_f1']:<10.4f} "
              f"{metrics['eval_log_loss']:<10.4f} {metrics['eval_roc_auc']:<10.4f}")
    
    print("\n📊 MÉTRICAS DE RENDIMIENTO")
    print("-" * 80)
    print(f"{'Modelo':<15} {'Tiempo Entrena (s)':<20} {'Inferencia (ms)':<25} {'Memoria (MB)':<15}")
    print("-" * 80)
    for model_name, metrics in models_results.items():
        print(f"{model_name:<15} {metrics['train_time_seconds']:<20.2f} "
              f"{metrics['inference_time_ms_mean']:.2f} ± {metrics['inference_time_ms_std']:.2f}{'':<10} "
              f"{metrics['memory_mb']:<15.2f}")
    
    print("\n📊 MÉTRICAS POR CLASE (F1-Score)")
    print("-" * 80)
    print(f"{'Modelo':<15} {'Negativo':<12} {'Neutral':<12} {'Positivo':<12}")
    print("-" * 80)
    for model_name, metrics in models_results.items():
        print(f"{model_name:<15} {metrics['eval_f1_neg']:<12.4f} {metrics['eval_f1_neu']:<12.4f} "
              f"{metrics['eval_f1_pos']:<12.4f}")
    
    print("\n" + "="*80)
    print("ANÁLISIS FINALIZADO")
    print("="*80)

if __name__ == "__main__":
    main()
