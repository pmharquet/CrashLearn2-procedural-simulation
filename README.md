# CrashLearn2 — Procedural Simulation

Base du simulateur du projet T-AIA-901 (Crash & Learn), importée le 16 septembre 2026 depuis `T-AIA-901/simulation/simulation`.

Les 165 fichiers fournis sont conservés à l'identique, hors caches Python/Numba et métadonnées macOS. Cette version correspond aussi à la copie `vendor/simulation` du projet `T-AIA-901-NCY-9-1-crashlearn-2`. Elle constitue le point de départ pour les évolutions de simulation procédurale.

## Démarrage

Prérequis : Docker avec prise en charge des conteneurs Linux.

```bash
docker compose build sim
docker compose run --rm sim
```

Le service `sim` lance `demo.py` sans interface graphique.

## Tests fournis

```bash
docker compose run --rm sim python test_contract.py
docker compose run --rm sim python test_integration.py
```

La version fournie présente un échec connu sur le test T08 de mise à jour de l'adhérence : `_update_friction` ne modifie pas l'adhérence. Le code d'origine est conservé pour cette base.

## Enregistrement et visualisation

```bash
docker compose run --rm recorder
docker compose run --rm replay
```

Consulter [les instructions fournies](INSTRUCTIONS.md) pour l'API et les règles du simulateur. `env_simulation.py` expose l'environnement, `gym/f110_gym` contient le moteur, et `maps` contient les circuits et leurs données.

## Licences

Les licences d'origine sont conservées : [MIT pour le moteur](LICENSE) et [GPL v3 pour les circuits](maps/LICENSE). Les ressources restent attribuées à leurs auteurs respectifs.
