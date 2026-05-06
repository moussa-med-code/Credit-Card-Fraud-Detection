"""
Modèle XGBoost pour la détection de fraude bancaire.
Entraînement avec cross-validation, early stopping et gestion du déséquilibre.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Configuration du logging
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Modèle de détection de fraude basé sur XGBoost.
    
    Caractéristiques :
    - Chargement des splits depuis parquet
    - Cross-validation stratifiée
    - Early stopping sur AUPRC
    - Gestion automatique du scale_pos_weight
    - Sauvegarde complète (modèle + métadonnées)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialise le modèle avec configuration YAML.
        
        Args:
            config_path: Chemin vers hyperparameters.yaml
        """
        self.config = self._load_config(config_path)
        self.model: Optional[XGBClassifier] = None
        self.feature_names: Optional[list] = None
        self.scale_pos_weight: Optional[float] = None
        self.best_threshold: float = 0.5
        self.cv_scores: list = []
        self.test_metrics: Dict[str, float] = {}
        
        # Fusion des paramètres
        self.params = {
            **self.config.get("base", {}),
            **self.config.get("architecture", {}),
            **self.config.get("regularization", {})
        }
        
        # Extraction early_stopping_rounds (pas un paramètre XGBoost direct)
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 30)
        
        logger.info("Modèle FraudDetectionModel initialisé")
        logger.info(f"Paramètres : {self.params}")
    
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """
        Charge la configuration YAML.
        
        Args:
            config_path: Chemin du fichier YAML
            
        Returns:
            Dictionnaire de configuration
        """
        if config_path is None:
            config_path = Path(__file__).parent / "hyperparameters.yaml"
        
        if not config_path.exists():
            logger.warning(f"Config introuvable à {config_path}, utilisation des valeurs par défaut")
            return {}
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration chargée depuis {config_path}")
        return config
    
    def load_data(self, splits_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Charge les splits train/test depuis les fichiers parquet.
        
        Args:
            splits_dir: Dossier contenant les fichiers parquet
            
        Returns:
            X_train, y_train, X_test, y_test
            
        Raises:
            FileNotFoundError: Si fichiers de splits manquants
        """
        logger.info(f"Chargement des données depuis {splits_dir}")
        
        # Vérification fichiers requis
        required_files = ["train_X.parquet", "train_y.parquet", 
                         "test_X.parquet", "test_y.parquet"]
        for f in required_files:
            if not (splits_dir / f).exists():
                raise FileNotFoundError(f"Fichier requis manquant : {splits_dir / f}")
        
        # Chargement
        X_train = pd.read_parquet(splits_dir / "train_X.parquet")
        y_train = pd.read_parquet(splits_dir / "train_y.parquet")["Class"]
        X_test = pd.read_parquet(splits_dir / "test_X.parquet")
        y_test = pd.read_parquet(splits_dir / "test_y.parquet")["Class"]
        
        self.feature_names = X_train.columns.tolist()
        
        # Chargement scale_pos_weight si disponible
        scale_pos_path = splits_dir / "scale_pos_weight.json"
        if scale_pos_path.exists():
            with open(scale_pos_path, "r") as f:
                data = json.load(f)
                self.scale_pos_weight = data.get("scale_pos_weight", None)
                logger.info(f"scale_pos_weight chargé : {self.scale_pos_weight:.2f}")
        else:
            # Calcul manuel
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            self.scale_pos_weight = n_neg / n_pos
            logger.info(f"scale_pos_weight calculé : {self.scale_pos_weight:.2f}")
        
        logger.info(f"Train : {X_train.shape} - Test : {X_test.shape}")
        logger.info(f"Fraudes train : {y_train.sum()} ({y_train.mean()*100:.4f}%)")
        logger.info(f"Fraudes test  : {y_test.sum()} ({y_test.mean()*100:.4f}%)")
        
        return X_train, y_train, X_test, y_test
    
    def train_with_cv(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_folds: int = 5
    ) -> None:
        """
        Entraîne le modèle avec cross-validation stratifiée.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            n_folds: Nombre de folds pour la CV
        """
        logger.info(f"Début entraînement avec CV {n_folds}-fold")
        
        # Configuration CV
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.params.get("random_state", 42)
        )
        
        # Paramètres finaux du modèle
        model_params = {
            **self.params,
            "scale_pos_weight": self.scale_pos_weight,
            "n_estimators": 500,  # Sera réduit par early stopping
            "early_stopping_rounds": self.early_stopping_rounds
        }
        
        # Entraînement par fold
        self.cv_scores = []
        cv_models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Nouveau modèle par fold
            fold_model = XGBClassifier(**model_params)
            
            fold_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Vérification si early stopping a été déclenché
            if hasattr(fold_model, 'best_iteration'):
                best_iter = fold_model.best_iteration
            else:
                best_iter = model_params["n_estimators"]
                logger.warning(f"Fold {fold}: early stopping non déclenché, "
                             f"utilisation de n_estimators={best_iter}")
            
            # Prédictions sur validation
            val_proba = fold_model.predict_proba(X_val)[:, 1]
            val_aucpr = average_precision_score(y_val, val_proba)
            
            self.cv_scores.append(val_aucpr)
            cv_models.append(fold_model)
            
            logger.info(f"  Fold {fold}/{n_folds} - AUPRC: {val_aucpr:.4f} - "
                       f"Best iteration: {best_iter}")
        
        # Sélection du meilleur modèle (meilleur AUPRC)
        best_fold_idx = np.argmax(self.cv_scores)
        self.model = cv_models[best_fold_idx]
        
        # Réentraînement final sur tout le train avec le nombre optimal d'arbres
        if hasattr(self.model, 'best_iteration'):
            best_n_estimators = self.model.best_iteration
        else:
            best_n_estimators = model_params["n_estimators"]
        
        logger.info(f"Réentraînement final avec n_estimators={best_n_estimators}")
        
        final_params = {
            **model_params,
            "n_estimators": best_n_estimators
        }
        # Pas d'early stopping pour le réentraînement final
        final_params.pop("early_stopping_rounds", None)
        
        final_model = XGBClassifier(**final_params)
        final_model.fit(X_train, y_train, verbose=False)
        self.model = final_model
        
        # Statistiques CV
        mean_aucpr = np.mean(self.cv_scores)
        std_aucpr = np.std(self.cv_scores)
        
        logger.info(f"CV terminée - AUPRC moyen: {mean_aucpr:.4f} (+/- {std_aucpr:.4f})")
        logger.info(f"Meilleur fold: {best_fold_idx+1} (AUPRC: {self.cv_scores[best_fold_idx]:.4f})")
    
    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Optimise le seuil de décision pour maximiser le F1-score.
        
        Args:
            X_val: Features de validation
            y_val: Target de validation
            
        Returns:
            Seuil optimal
        """
        if self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelez train_with_cv() d'abord.")
        
        proba = self.model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, proba)
        
        # Calcul du F1-score pour chaque seuil
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Meilleur seuil (ignore le dernier élément car thresholds a n-1 éléments)
        best_idx = np.argmax(f1_scores[:-1])
        self.best_threshold = thresholds[best_idx]
        
        logger.info(f"Seuil optimal: {self.best_threshold:.4f} "
                   f"(F1: {f1_scores[best_idx]:.4f})")
        
        return self.best_threshold
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Évalue le modèle sur le jeu de test.
        
        Args:
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            Dictionnaire des métriques de test
        """
        if self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelez train_with_cv() d'abord.")
        
        logger.info("Évaluation sur le jeu de test")
        
        # Prédictions
        test_proba = self.model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= self.best_threshold).astype(int)
        
        # Métriques
        test_aucpr = average_precision_score(y_test, test_proba)
        test_f1 = f1_score(y_test, test_pred)
        
        self.test_metrics = {
            "auprc": test_aucpr,
            "f1_score": test_f1,
            "threshold": self.best_threshold,
            "cv_aucpr_mean": float(np.mean(self.cv_scores)),
            "cv_aucpr_std": float(np.std(self.cv_scores))
        }
        
        # Rapport détaillé
        logger.info(f"Test AUPRC: {test_aucpr:.4f}")
        logger.info(f"Test F1-Score: {test_f1:.4f}")
        logger.info(f"Seuil utilisé: {self.best_threshold:.4f}")
        logger.info("\nRapport de classification:\n" + 
                   classification_report(y_test, test_pred, target_names=["Normal", "Fraude"]))
        
        return self.test_metrics
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit la probabilité de fraude pour de nouvelles transactions.
        
        Args:
            X: Features d'entrée
            
        Returns:
            DataFrame avec probabilités et prédictions
        """
        if self.model is None:
            raise RuntimeError("Modèle non entraîné.")
        
        proba = self.model.predict_proba(X)[:, 1]
        predictions = (proba >= self.best_threshold).astype(int)
        
        return pd.DataFrame({
            "probability": proba,
            "prediction": predictions,
            "threshold": self.best_threshold
        })
    
    def save(self, path: Path) -> None:
        """
        Sauvegarde le modèle et ses métadonnées.
        
        Args:
            path: Chemin de sauvegarde (sans extension)
        """
        if self.model is None:
            raise RuntimeError("Aucun modèle à sauvegarder.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "scale_pos_weight": self.scale_pos_weight,
            "best_threshold": self.best_threshold,
            "cv_scores": self.cv_scores,
            "test_metrics": self.test_metrics
        }
        
        joblib.dump(save_dict, path.with_suffix(".pkl"))
        logger.info(f"Modèle sauvegardé dans {path.with_suffix('.pkl')}")
    
    @classmethod
    def load(cls, path: Path) -> "FraudDetectionModel":
        """
        Charge un modèle sauvegardé.
        
        Args:
            path: Chemin du fichier .pkl
            
        Returns:
            Instance de FraudDetectionModel chargée
        """
        path = Path(path).with_suffix(".pkl")
        
        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {path}")
        
        save_dict = joblib.load(path)
        
        instance = cls()
        instance.model = save_dict["model"]
        instance.params = save_dict["params"]
        instance.feature_names = save_dict["feature_names"]
        instance.scale_pos_weight = save_dict["scale_pos_weight"]
        instance.best_threshold = save_dict["best_threshold"]
        instance.cv_scores = save_dict["cv_scores"]
        instance.test_metrics = save_dict["test_metrics"]
        
        logger.info(f"Modèle chargé depuis {path}")
        logger.info(f"Test AUPRC: {instance.test_metrics.get('auprc', 'N/A'):.4f}")
        
        return instance


if __name__ == "__main__":
    # Exemple d'utilisation
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Chemins
    splits_dir = Path("data/splits")
    model_dir = Path("saved_models")
    
    try:
        # Initialisation
        model = FraudDetectionModel()
        
        # Chargement données
        X_train, y_train, X_test, y_test = model.load_data(splits_dir)
        
        # Entraînement avec CV
        model.train_with_cv(X_train, y_train, n_folds=5)
        
        # Optimisation du seuil sur une portion train
        model.optimize_threshold(X_train, y_train)
        
        # Évaluation finale
        metrics = model.evaluate(X_test, y_test)
        
        # Sauvegarde
        model.save(model_dir / "fraud_detection_xgboost")
        
        logger.info("✅ Entraînement terminé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur : {e}", exc_info=True)
        sys.exit(1)