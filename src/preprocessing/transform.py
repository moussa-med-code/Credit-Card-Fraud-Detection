"""
Module de preprocessing pour le dataset Credit Card Fraud Detection.
Transforme les données brutes en features normalisées prêtes pour XGBoost.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

# Configuration du logging structuré
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class CreditCardPreprocessor:
    """
    Préprocesseur pour le dataset Credit Card Fraud Detection.
    
    Étapes :
    1. Nettoyage (doublons, valeurs manquantes)
    2. Feature engineering (log, scaling)
    3. Split stratifié train/val/test
    4. Sauvegarde en format parquet
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise le préprocesseur.
        
        Args:
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.random_state = random_state
        self.time_scaler: Optional[StandardScaler] = None
        self.amount_scaler: Optional[RobustScaler] = None
        self.features_scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[list] = None
        self.scale_pos_weight: Optional[float] = None
        self._fitted = False
        
    def _validate_no_missing(self, df: pd.DataFrame) -> None:
        """
        Vérifie l'absence de valeurs manquantes dans le DataFrame.
        
        Args:
            df: DataFrame à vérifier
            
        Raises:
            ValueError: Si des valeurs manquantes sont détectées
        """
        missing_count = df.isnull().sum().sum()
        logger.info(f"Vérification des valeurs manquantes : {missing_count} trouvées")
        if missing_count != 0:
            raise ValueError(
                f"Dataset contient {missing_count} valeurs manquantes. "
                f"Attendu : 0. Nettoyage requis avant preprocessing."
            )
        logger.info("✓ Aucune valeur manquante détectée")
    
    def _remove_duplicates(self, df: pd.DataFrame, expected_duplicates: int = 1081) -> pd.DataFrame:
        """
        Supprime les doublons du DataFrame.
        
        Args:
            df: DataFrame d'entrée
            expected_duplicates: Nombre attendu de doublons (vérification d'intégrité)
            
        Returns:
            DataFrame sans doublons
            
        Raises:
            UserWarning: Si le nombre de doublons diffère de l'attendu
        """
        n_before = len(df)
        duplicates = df.duplicated().sum()
        
        if duplicates != expected_duplicates:
            logger.warning(
                f"Nombre de doublons ({duplicates}) différent de l'attendu "
                f"({expected_duplicates}). Vérifiez l'intégrité des données."
            )
        
        df_clean = df.drop_duplicates(keep='first')
        n_after = len(df_clean)
        
        logger.info(f"Forme avant nettoyage : {df.shape}")
        logger.info(f"Doublons supprimés : {n_before - n_after}")
        logger.info(f"Forme après nettoyage : {df_clean.shape}")
        
        return df_clean
    
    def fit(self, df: pd.DataFrame) -> 'CreditCardPreprocessor':
        """
        Apprend les paramètres de transformation sur les données.
        
        Args:
            df: DataFrame complet avec colonnes Time, Amount, V1-V28, Class
            
        Returns:
            self pour chaînage
            
        Raises:
            KeyError: Si colonnes requises manquantes
        """
        logger.info("Début du fit des transformations...")
        
        # Validation des colonnes requises
        required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise KeyError(f"Colonnes manquantes dans le DataFrame : {missing_cols}")
        
        # Nettoyage
        self._validate_no_missing(df)
        df = self._remove_duplicates(df)
        
        # Séparation features/target
        self.feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Fit Time scaler
        self.time_scaler = StandardScaler()
        self.time_scaler.fit(df[['Time']])
        logger.info("✓ StandardScaler ajusté sur Time")
        
        # Fit Amount scaler (log + RobustScaler)
        amount_log = np.log1p(df['Amount'])
        self.amount_scaler = RobustScaler()
        self.amount_scaler.fit(amount_log.values.reshape(-1, 1))
        logger.info("✓ RobustScaler ajusté sur log(Amount)")
        
        # Fit features V1-V28 scaler
        v_features = [f'V{i}' for i in range(1, 29)]
        self.features_scaler = StandardScaler()
        self.features_scaler.fit(df[v_features])
        logger.info("✓ StandardScaler ajusté sur V1-V28")
        
        self._fitted = True
        logger.info("Fit terminé avec succès")
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transforme les données selon les paramètres appris.
        
        Args:
            df: DataFrame à transformer
            
        Returns:
            Tuple (X_processed, y)
            
        Raises:
            RuntimeError: Si fit() n'a pas été appelé avant
        """
        if not self._fitted:
            raise RuntimeError("Le préprocesseur doit être fitté avant d'appeler transform()")
        
        logger.info("Début de la transformation...")
        
        # Nettoyage
        self._validate_no_missing(df)
        df = self._remove_duplicates(df)
        
        # Extraction target
        y = df['Class'].copy()
        
        # Transformation Time
        time_scaled = self.time_scaler.transform(df[['Time']])
        
        # Transformation Amount
        amount_log = np.log1p(df['Amount'])
        amount_scaled = self.amount_scaler.transform(amount_log.values.reshape(-1, 1))
        
        # Transformation V1-V28
        v_features = [f'V{i}' for i in range(1, 29)]
        v_scaled = self.features_scaler.transform(df[v_features])
        
        # Assemblage final
        X_processed = np.hstack([time_scaled, v_scaled, amount_scaled])
        X_processed = pd.DataFrame(
            X_processed,
            columns=self.feature_names,
            index=df.index
        )
        
        logger.info(f"Transformation terminée. X shape: {X_processed.shape}")
        return X_processed, y
    
    def split_and_save(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: Path,
        train_size: float = 0.8,
        val_size: float = 0.0,
        test_size: float = 0.2
    ) -> None:
        """
        Split stratifié et sauvegarde en parquet.
        Pour ce dataset déséquilibré (0.17% fraude) : 80/20 recommandé.
        Validation gérée via cross-validation dans l'entraînement.
        
        Args:
            X: Features transformées
            y: Target
            output_dir: Dossier racine de sortie
            train_size: Proportion train (default: 0.8)
            val_size: Proportion validation (default: 0.0, utiliser CV)
            test_size: Proportion test (default: 0.2)
            
        Raises:
            ValueError: Si les proportions ne somment pas à 1
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-10:
            raise ValueError(
                f"Les proportions doivent sommer à 1. Actuel : "
                f"{train_size + val_size + test_size}"
            )
        
        logger.info(f"Split stratifié {train_size:.0%}/{test_size:.0%} (validation via CV)")
        
        if val_size == 0.0:
            # Split 80/20 recommandé pour ce dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=self.random_state
            )
            
            # Pas de set de validation séparé
            X_val, y_val = None, None
            
            # Calcul scale_pos_weight pour XGBoost
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            self.scale_pos_weight = n_neg / n_pos
            
            logger.info(f"Split effectué :")
            logger.info(f"  Train : {X_train.shape[0]} ({y_train.mean()*100:.4f}% fraudes)")
            logger.info(f"  Test  : {X_test.shape[0]} ({y_test.mean()*100:.4f}% fraudes)")
            logger.info(f"  Validation : gérée via StratifiedKFold (5-fold CV)")
            logger.info(f"  scale_pos_weight pour XGBoost : {self.scale_pos_weight:.2f}")
            
        else:
            # Fallback si l'utilisateur veut vraiment un set de validation
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=val_size + test_size,
                stratify=y,
                random_state=self.random_state
            )
            
            relative_val_size = val_size / (val_size + test_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=1 - relative_val_size,
                stratify=y_temp,
                random_state=self.random_state
            )
            
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            self.scale_pos_weight = n_neg / n_pos
            
            logger.info(f"Split effectué :")
            logger.info(f"  Train : {X_train.shape[0]} ({y_train.mean()*100:.4f}% fraudes)")
            logger.info(f"  Val   : {X_val.shape[0]} ({y_val.mean()*100:.4f}% fraudes)")
            logger.info(f"  Test  : {X_test.shape[0]} ({y_test.mean()*100:.4f}% fraudes)")
            logger.info(f"  scale_pos_weight pour XGBoost : {self.scale_pos_weight:.2f}")
        
        # Création des dossiers
        processed_dir = output_dir / "processed"
        splits_dir = output_dir / "splits"
        processed_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde données complètes
        logger.info(f"Sauvegarde dans {output_dir}")
        X.to_parquet(processed_dir / "X_processed.parquet", index=False)
        y.to_frame('Class').to_parquet(processed_dir / "y_processed.parquet", index=False)
        logger.info("✓ Données complètes sauvegardées en parquet")
        
        # Sauvegarde splits
        X_train.to_parquet(splits_dir / "train_X.parquet", index=False)
        y_train.to_frame('Class').to_parquet(splits_dir / "train_y.parquet", index=False)
        X_test.to_parquet(splits_dir / "test_X.parquet", index=False)
        y_test.to_frame('Class').to_parquet(splits_dir / "test_y.parquet", index=False)
        
        if X_val is not None:
            X_val.to_parquet(splits_dir / "val_X.parquet", index=False)
            y_val.to_frame('Class').to_parquet(splits_dir / "val_y.parquet", index=False)
        
        logger.info("✓ Tous les splits sauvegardés en parquet")
        
        # Sauvegarde scale_pos_weight
        pd.Series({'scale_pos_weight': self.scale_pos_weight}).to_json(
            splits_dir / "scale_pos_weight.json"
        )
        logger.info("✓ scale_pos_weight sauvegardé")


    def save(self, path: Path) -> None:
        """
        Sauvegarde le préprocesseur (scalers et paramètres).
        
        Args:
            path: Chemin de sauvegarde (.joblib)
        """
        if not self._fitted:
            raise RuntimeError("Le préprocesseur doit être fitté avant d'être sauvegardé.")
        
        import joblib
        path = Path(path).with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "time_scaler": self.time_scaler,
            "amount_scaler": self.amount_scaler,
            "features_scaler": self.features_scaler,
            "feature_names": self.feature_names,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "_fitted": self._fitted
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"Préprocesseur sauvegardé dans {path}")

    @classmethod
    def load(cls, path: Path) -> 'CreditCardPreprocessor':
        """
        Charge un préprocesseur sauvegardé.
        
        Args:
            path: Chemin du fichier .joblib
            
        Returns:
            Instance de CreditCardPreprocessor chargée
        """
        import joblib
        path = Path(path).with_suffix(".joblib")
        
        if not path.exists():
            raise FileNotFoundError(f"Préprocesseur introuvable : {path}")
        
        save_dict = joblib.load(path)
        
        instance = cls(random_state=save_dict["random_state"])
        instance.time_scaler = save_dict["time_scaler"]
        instance.amount_scaler = save_dict["amount_scaler"]
        instance.features_scaler = save_dict["features_scaler"]
        instance.feature_names = save_dict["feature_names"]
        instance.scale_pos_weight = save_dict["scale_pos_weight"]
        instance._fitted = save_dict["_fitted"]
        
        logger.info(f"Préprocesseur chargé depuis {path}")
        return instance

def main(raw_path: Path, output_dir: Path) -> None:
    """
    Point d'entrée principal du preprocessing.
    
    Args:
        raw_path: Chemin vers creditcard.csv
        output_dir: Dossier de sortie pour les données traitées
        
    Raises:
        FileNotFoundError: Si le fichier source n'existe pas
    """
    # Vérification fichier source
    if not raw_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {raw_path}")
    
    logger.info(f"Chargement des données depuis {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Dataset chargé : {df.shape}")
    
    # Preprocessing
    preprocessor = CreditCardPreprocessor(random_state=42)
    preprocessor.fit(df)
    X_processed, y = preprocessor.transform(df)
    
    # Split et sauvegarde
    preprocessor.split_and_save(X_processed, y, output_dir)
    
    logger.info("✅ Preprocessing terminé avec succès")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing du dataset Credit Card Fraud Detection"
    )
    parser.add_argument(
        "--raw_path",
        type=Path,
        default=Path("data/raw/creditcard.csv"),
        help="Chemin vers le fichier CSV brut"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Dossier racine pour les sorties"
    )
    
    args = parser.parse_args()
    
    try:
        main(args.raw_path, args.output_dir)
    except FileNotFoundError as e:
        logger.error(f"Erreur de fichier : {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Erreur de validation : {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Erreur de colonnes : {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}", exc_info=True)
        sys.exit(1)