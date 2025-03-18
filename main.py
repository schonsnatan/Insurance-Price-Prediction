from src.data_preprocessing import load_data, clean_data
from src.model_training import train_model
from src.evaluate_model import evaluate_model

# Definir o caminho dos dados
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'

# Carregar e limpar os dados
df = load_data(filepath)
df = clean_data(df)

# Treinar modelo e exibir o resultado
model, y_pred, y_test = train_model(df, use_ridge=False)

# Avaliar modelo
metrics = evaluate_model(y_test, y_pred, model_name="Linear Regression")
