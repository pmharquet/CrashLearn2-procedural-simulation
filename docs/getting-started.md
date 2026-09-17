# Démarrer

[Accueil](../README.md) · [Utilisation](usage.md) · [Dépannage](troubleshooting.md)

## 1. Installer

Ouvrir un terminal à la racine du dépôt. Avec Python 3.12 :

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e .
```

Sous Linux/macOS, utiliser `python3` pour créer l’environnement, puis `.venv/bin/python`. L’interface nécessite un affichage graphique. Le premier lancement peut prendre plus longtemps : Numba compile les calculs physiques.

## 2. Préparer un pilote

Le dépôt contient les circuits et le simulateur, mais aucun modèle entraîné.

| Sélection | Contenu attendu |
|---|---|
| `agent.py` | Une classe `Agent` avec `predict(obs, info)`, ses dépendances et ses poids |
| `model.onnx` | Un `agent.py` dans le même dossier pour adapter observations et actions |
| PPO `.zip` | Un modèle du simulateur, 102 ou 131 entrées |

`predict` retourne `(vitesse, direction)` : vitesse en m/s, direction en radians. Chaque pilote de compétition tourne dans son propre processus. Ses fichiers restent à leur emplacement.

## 3. Lancer la compétition

```powershell
.venv/Scripts/python -m crashlearn_sim
```

1. Choisir un circuit fixe ou le procédural et la durée de course.
2. Préparer 1 à 4 voitures : nom, couleur et pilote.
3. Démarrer et attendre le compte à rebours.
4. Utiliser **Échap / Arrêter** pour revenir au paddock, ou **R** pour recommencer.

Le classement final permet de relancer la course ou de revenir au menu. Voir les [raccourcis](usage.md#commandes).

## 4. Entraîner un pilote

```powershell
.venv/Scripts/python -m crashlearn_sim.training --help
```

Sélectionner ensuite le fichier PPO `.zip` produit dans le paddock.

## Exécutable Windows

Lancer `CrashLearn.exe` dans son dossier complet. Python n’est pas nécessaire sur le poste cible. Les enregistrements de compétition vont dans `%LOCALAPPDATA%/CrashLearn/recordings`.

Pour construire une distribution, suivre le [guide développeur](development.md#construire-lexécutable-windows).
