# Développer

[Accueil](../README.md) · [Architecture](architecture.md)

## Installer

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,legacy,freeze]"
```

`pyproject.toml` est la source unique des dépendances et des commandes. Les options `dev`, `legacy` et `freeze` ajoutent respectivement les outils de qualité, le replay historique Matplotlib et PyInstaller. L’installation éditable prend immédiatement en compte les modifications dans `src/`.

L’option `monitoring` ajoute TensorBoard pour consulter les journaux d’entraînement.

## Commandes

| Usage | Commande avec le Python de `.venv` |
|---|---|
| Application | `python -m crashlearn_sim` |
| Entraînement | `python -m crashlearn_sim.training` |
| Enregistrement historique | `python -m crashlearn_sim.legacy.record` |
| Replay historique | `python -m crashlearn_sim.legacy.replay` |

Après activation de l’environnement, les raccourcis installés `crashlearn`, `crashlearn-train`, `crashlearn-record` et `crashlearn-replay` sont également disponibles. Les anciens scripts `procedural_*.py`, `race_*.py`, `sim_recorder.py` et `viz_replay.py` ne sont plus des points d’entrée.

Pour les imports, utiliser par exemple `from crashlearn_sim.simulation.world import ProceduralSimulation`. L’API historique se trouve dans `crashlearn_sim.legacy.environment`.

## Vérifier une modification

```powershell
.venv/Scripts/python -m ruff format .
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m pytest
```

Les tests couvrent simulation, circuits, pilotes, processus, enregistrement et installation. `tests/fixtures/simulation-baseline.json` contient trois scénarios de référence exécutés sur `95d72b8` avant réorganisation. Ne pas le régénérer pour masquer une régression.

Le contrat historique **T08** reste marqué `xfail(strict=True)` : l’adhérence du moteur historique reste à 1.0 au lieu de diminuer. Il échouait avant cette réorganisation. S’il passe, le marquage doit être réexaminé. La météo procédurale a ses propres tests.

### Interface native

Avec un affichage OpenGL 3.3 :

```powershell
$env:CRASHLEARN_GPU_TESTS = '1'
.venv/Scripts/python -m pytest tests/test_rendering.py tests/test_competition_ui.py
Remove-Item Env:CRASHLEARN_GPU_TESTS
```

Ces tests ouvrent une fenêtre et vérifient plein écran, redimensionnement et parcours paddock → course → résultats → reprise. Ils sont désactivés par défaut en CI sans affichage.

## Construire un paquet Python

```powershell
.venv/Scripts/python -m build
```

Le wheel et l’archive source sont créés dans `dist/`. Le wheel contient l’application, le moteur F110 et les circuits ; il peut être installé hors du checkout. Il ne contient pas de modèles IA.

## Construire l’exécutable Windows

```powershell
./deployment/windows/build.ps1
```

Distribuer tout `dist/CrashLearn`. Le spec embarque les ressources et les sources utilisées par le cache Numba. Le hook de démarrage dirige les caches et journaux vers le dossier utilisateur.

Vérification avec un pilote réel :

```powershell
dist/CrashLearn/CrashLearn.exe --verify-agent "C:/chemin/agent.py" --verify-output logs/distribution.json
```

Examiner le rapport avant livraison et essayer la distribution sur la machine cible. Un build réussi ne remplace pas cette vérification.

## Docker et intégration continue

```sh
docker compose build
docker compose run --rm sim
docker compose run --rm recorder
docker compose run --rm replay
docker compose up tensorboard
```

Le conteneur utilise Python 3.12 et le même `pyproject.toml`. Le service `sim` lance les tests sans affichage ; le replay historique génère une image. Le workflow GitHub vérifie le style, les tests et la construction du wheel sous Windows et Linux.

Les règles Git restent dans [AGENTS.md](../AGENTS.md).
