# Dépannage

[Accueil](../README.md) · [Démarrer](getting-started.md) · [Guide complet](usage.md)

| Symptôme | Action |
|---|---|
| `ModuleNotFoundError` | Utiliser le Python de `.venv` et installer le projet avec `pip install -e .` ; pour pytest, `pip install -e ".[dev]"`. |
| Aucun agent trouvé | Sélectionner le pilote dans le paddock ; les modèles ne sont pas inclus. |
| ONNX refusé | Placer son `agent.py` à côté pour le prétraitement et les actions. |
| Pilote en erreur | Vérifier ses dépendances, chemins de poids et le retour de `predict` : deux nombres finis. |
| OpenGL indisponible | Essayer `--renderer software`. `auto` annonce son repli ; `gpu` exige OpenGL. |
| Premier démarrage lent | Attendre la compilation Numba ; les lancements suivants utilisent le cache. |
| Replay introuvable | Vérifier le chemin choisi et `%LOCALAPPDATA%/CrashLearn/recordings` pour la compétition Windows. |
| Exécutable incomplet après copie | Copier tout `dist/CrashLearn`. |
| Tests « skipped » | Activer `CRASHLEARN_GPU_TESTS=1` avec un affichage natif. |
| T08 « xfailed » | Anomalie historique documentée dans le guide développeur. |

Pour signaler un problème, conserver la commande, le message complet, le système, la version Python et les paramètres reproductibles (graine, circuit, météo, pilote). Le bundle Windows écrit aussi dans `%LOCALAPPDATA%/CrashLearn/application.log`.
