import os

# Configuration logic
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
FEATURE_COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

SPLITS_DIR = os.path.join(DATA_DIR, "splits")  # Pour les données de test

# Chemins des fichiers selon votre architecture
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_detection_xgboost.pkl")
TEST_DATA_PATH = os.path.join(SPLITS_DIR, "test_X.parquet")
TEST_LABELS_PATH = os.path.join(SPLITS_DIR, "test_y.parquet")

# Chemins supplémentaires utiles
TRAIN_DATA_PATH = os.path.join(SPLITS_DIR, "train_X.parquet")
TRAIN_LABELS_PATH = os.path.join(SPLITS_DIR, "train_y.parquet")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "creditcard.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "X_processed.parquet")
PROCESSED_LABELS_PATH = os.path.join(DATA_DIR, "processed", "y_processed.parquet")
SCALE_POS_WEIGHT_PATH = os.path.join(SPLITS_DIR, "scale_pos_weight.json")