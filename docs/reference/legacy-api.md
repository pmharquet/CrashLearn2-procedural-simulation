# Moteur historique

[Accueil](../../README.md) · [Développement](../development.md)

L’application de compétition utilise son propre parcours de course. Les outils ci-dessous conservent l’API historique et son format d’enregistrement NPZ.

## Enregistrer puis rejouer

Installer l’option `legacy` pour le rendu Matplotlib :

```powershell
.venv/Scripts/python -m pip install -e ".[legacy]"
.venv/Scripts/python -m crashlearn_sim.legacy.record --steps 500 --controller random --out recordings/episode.npz
.venv/Scripts/python -m crashlearn_sim.legacy.replay --npz recordings/episode.npz --no-display
```

Le replay sans affichage produit une image à côté du fichier NPZ. Sous Linux avec un affichage disponible, retirer `--no-display` pour utiliser le lecteur interactif : pause, curseur temporel, flèches, zoom à la souris et déplacement de caméra.

Avec Docker : `docker compose run --rm recorder`, puis `docker compose run --rm replay`.

## Utiliser un pilote

```powershell
.venv/Scripts/python -m crashlearn_sim.legacy.record --submissions "C:/pilotes/equipe" --num-cars 4 --out recordings/course.npz
```

Le dossier contient `agent.py` et `model.onnx`. Le chargeur accepte aussi un dossier parent regroupant plusieurs équipes. `Agent.predict(obs, info)` retourne une vitesse cible et un angle de direction.

| Option d’enregistrement | Fonction |
|---|---|
| `--steps N` | Nombre maximal de décisions, 2 000 par défaut |
| `--laps N` | Nombre de tours cible, 2 par défaut |
| `--num-cars N` | De 1 à 4 voitures |
| `--submissions CHEMIN` | Pilote ou dossier d’équipes |
| `--controller` | `auto`, `submission`, `random`, `onnx` ou `pure_pursuit` |
| `--model CHEMIN` | Modèle ONNX direct pour la première voiture |
| `--out CHEMIN` | Fichier NPZ de destination |

Le replay accepte `--npz`, `--map`, `--follow` et `--no-display`. Consulter `--help` pour les paramètres complets.

## API Python

```python
from crashlearn_sim.legacy.environment import (
    set_map, get_available_maps, reset, get_obs,
    apply_action, simulation_step, get_step_info, close,
)

print(get_available_maps())
set_map("Spa")
reset(num_cars=1)
try:
    for _ in range(500):
        observation = get_obs(0)
        apply_action(0, 3.0, 0.0)
        simulation_step()
        if get_step_info()["lap_complete"][0]:
            break
finally:
    close()
```

L’environnement historique repose sur un singleton : le fermer entre deux utilisations indépendantes. Les tests de contrat sont dans `tests/test_contract.py` et `tests/test_integration.py`. L’anomalie d’adhérence T08 est documentée dans le guide développeur.
