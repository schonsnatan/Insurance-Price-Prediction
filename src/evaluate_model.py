import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Gera métricas e gráficos para avaliar o modelo"""
    
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"📊 Avaliação do {model_name}:")
    print(f"🔹 R² Score: {r2:.4f}")
    print(f"🔹 Mean Squared Error (MSE): {mse:.4f}")

    # Gráfico de valores reais vs previstos
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel("Valores Reais")
    plt.ylabel("Valores Previstos")
    plt.title(f"Valores Reais vs Previstos - {model_name}")
    plt.show()

    # Histograma do erro
    errors = y_true - y_pred
    plt.figure(figsize=(8,5))
    sns.histplot(errors, bins=20, kde=True)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel("Erro (Real - Previsto)")
    plt.ylabel("Frequência")
    plt.title(f"Distribuição dos Erros - {model_name}")
    plt.show()
    
    return {"R2": r2, "MSE": mse}
