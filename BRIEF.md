# Projet : Machine Learning for Volatility Forecasting

## Contexte (à lire avant tout)

Je suis étudiant en 4e année de mathématiques appliquées (Polytech Nice Sophia, MAM). Ce projet ira sur mon CV et dans mon dossier pour le Mastère Spécialisé Finance et Gestion des Risques de l'ENSAE. Je dois pouvoir **expliquer chaque ligne en entretien**.

Règles de travail :

1. On avance **phase par phase** (voir plus bas). À la fin de chaque phase, tu t'arrêtes, tu résumes ce qui a été fait et tu attends ma validation avant la phase suivante.
2. Avant de coder une phase, tu proposes un plan court. Pas de code avant que je l'aie validé.
3. À la fin de chaque phase, tu me donnes : les 3 notions que je dois comprendre, et 3 questions qu'un recruteur quant pourrait me poser sur cette partie.
4. Le code, les commentaires et le README sont **en anglais**. Tes explications pour moi sont en français.
5. **Honnêteté des résultats** : aucun réglage d'hyperparamètres sur la période de test, aucun chiffre arrangé. Si le ML ne bat pas le benchmark, on le dit et on l'analyse. Un résultat négatif bien expliqué vaut mieux qu'un résultat faux.
6. Tout tourne sur CPU (ordinateur portable) en un temps raisonnable.

## Objectif

Prévoir la **volatilité réalisée à 5 jours** (horizon h = 5 jours de bourse) d'actions du S&P 500, comparer des modèles économétriques et de machine learning en validation walk-forward, puis tester l'intérêt économique des prévisions dans une stratégie de ciblage de volatilité.

## Données

- Source : `yfinance`, données journalières OHLCV de 2005 à fin 2025, mises en cache en local (parquet) pour ne pas retélécharger.
- Univers : SPY + environ 20 actions liquides du S&P 500, de secteurs variés (ex. AAPL, MSFT, JPM, XOM, JNJ, PG, ...). Mentionner le biais de survie dans le README.
- Variable exogène : niveau du VIX (`^VIX`), toujours décalé d'un jour.
- Cible : variance quotidienne estimée par **Garman-Klass** à partir des prix OHLC, sommée sur les 5 jours suivants, puis annualisée. Le README doit préciser que c'est un proxy, car la vraie volatilité réalisée demande des données intrajournalières.
- Les modèles prédisent le **log** de la volatilité réalisée.

## Modèles à comparer

1. **Naïf** : la volatilité réalisée des 5 derniers jours.
2. **HAR-RV** (Corsi, 2009) : régression linéaire sur la volatilité réalisée journalière, hebdomadaire (5 j) et mensuelle (22 j). C'est le benchmark clé de la littérature.
3. **GARCH(1,1)** et **GJR-GARCH** avec innovations de Student, via le package `arch`. Prévision multi-pas sommée sur 5 jours, réestimation périodique (par ex. tous les 21 jours).
4. **Gradient boosting** (LightGBM ou `HistGradientBoostingRegressor` de scikit-learn). Variables : volatilité réalisée retardée (1, 5, 22 j), rendements et valeurs absolues des rendements retardés, estimateur de Parkinson, VIX décalé, variation de volume.
5. **LSTM** (PyTorch) sur des séquences de 22 à 60 jours des mêmes variables. Petit réseau, early stopping sur un jeu de validation pris **à la fin** de la fenêtre d'entraînement, graines aléatoires fixées.

## Validation (point le plus important)

- **Walk-forward à fenêtre croissante** : on entraîne sur toutes les données jusqu'à t, on prévoit le bloc suivant, puis on avance. Jamais de mélange aléatoire.
- **Embargo de 5 jours** entre entraînement et test, car les cibles à 5 jours se chevauchent.
- Les normalisations (scalers) sont ajustées sur l'entraînement uniquement.
- Première période de test : à partir de 2012 environ, pour avoir assez d'historique.
- **Tests automatiques (pytest)** qui vérifient l'absence de fuite d'information : une variable à la date t n'utilise que des données jusqu'à t, la cible à t n'utilise que t+1 à t+5, etc.

## Évaluation

- Métriques : **QLIKE** (métrique principale pour la volatilité), MSE sur la variance, MAE sur la volatilité, R² de la régression de Mincer-Zarnowitz.
- **Test de Diebold-Mariano** de chaque modèle contre GARCH et contre HAR.
- Résultats par action, agrégés sur l'univers, et par régime de marché (2015-2019 calme, 2020 Covid, 2022 inflation).
- Figures : prévisions contre réalisé sur SPY, erreur cumulée par modèle, importance des variables du gradient boosting.

## Backtest de ciblage de volatilité (sur SPY)

- Poids à la date t : `w_t = min(cible_vol / vol_prévue_t, 2)` (levier plafonné à 2), cible de 10 % annualisée.
- Prévision connue à la clôture de t, position prise à la clôture de t+1 (aucun regard vers le futur).
- Rééquilibrage hebdomadaire, coûts de transaction de 5 points de base par unité de rotation.
- Comparaison avec le buy-and-hold **ramené à la même volatilité moyenne** (sinon la comparaison des Sharpe est trompeuse).
- Une stratégie par source de prévision (naïf, HAR, GARCH, gradient boosting, LSTM).
- Métriques : rendement et volatilité annualisés, Sharpe, perte maximale (max drawdown), rotation.

## Structure du dépôt

```
vol-forecasting/
├── README.md
├── pyproject.toml (ou requirements.txt)
├── configs/default.yaml
├── src/volforecast/
│   ├── data.py
│   ├── features.py
│   ├── targets.py
│   ├── walkforward.py
│   ├── models/ (naive.py, har.py, garch.py, gbm.py, lstm.py)
│   ├── evaluation.py
│   └── backtest.py
├── scripts/run_all.py
├── tests/
└── reports/figures/
```

Une seule commande doit tout relancer de bout en bout : `python scripts/run_all.py`.

## Phases

0. **Plan** : tu lis ce brief, tu proposes l'architecture, les versions de Python et des librairies, et tu signales tout ce qui te paraît flou ou discutable. Tu attends ma validation.
1. **Données et cible** : téléchargement, cache, calcul de la cible Garman-Klass, tests d'alignement.
2. **Moteur walk-forward et benchmarks** : naïf, HAR, GARCH, métriques, premiers résultats.
3. **Gradient boosting**.
4. **LSTM**.
5. **Évaluation complète** : Diebold-Mariano, régimes, figures.
6. **Backtest** de ciblage de volatilité.
7. **README final** : objectif, méthode, tableau de résultats, figures, limites, comment relancer. Puis tu me proposes deux lignes de CV en anglais fondées **uniquement sur les résultats obtenus**.
