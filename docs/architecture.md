# Architecture

[Accueil](../README.md) · [Développement](development.md)

## Organisation

```text
src/
├── crashlearn_sim/           Application
│   ├── app.py               Arguments et orchestration du parcours de course
│   ├── agents.py            Chargement et validation des pilotes
│   ├── runtime.py           Processus de simulation et échanges avec l’interface
│   ├── recording.py         Enregistrement et relecture SQLite
│   ├── training.py          Environnement Gymnasium et entraînement PPO
│   ├── paths.py             Ressources installées et configuration des pilotes
│   ├── simulation/          Physique, route procédurale et état du monde
│   ├── competition/         Configuration, circuits et processus des pilotes
│   ├── ui/                  Menu, navigation et vues de compétition
│   ├── rendering/           Scène, panneaux, effets, fenêtre et rendu GPU
│   ├── legacy/              API, enregistrement et replay historiques
│   └── resources/maps/      Circuits livrés avec l’application
└── f110_gym/                Moteur tiers d’origine

tests/                      Contrats et régressions applicatives
docs/                       Guides et références
deployment/
├── windows/                Construction PyInstaller et démarrage du bundle
└── docker/                 Image des outils sans affichage
pyproject.toml              Installation, commandes, dépendances et qualité
```

La racine ne contient pas de scripts applicatifs ni de copies de compatibilité. L’application s’installe avec `pip install -e .` ; les imports utilisent exclusivement les packages de `src/`.

## Responsabilités

| Modifier… | Fichier ou dossier |
|---|---|
| Intégration physique, LiDAR, contacts | `simulation/physics.py` |
| Génération du tracé | `simulation/road.py` |
| Météo, progression et abandon | `simulation/world.py` |
| Configuration et limites de course | `competition/models.py` |
| Isolation et erreurs des pilotes | `competition/pilots.py` |
| Édition et navigation du paddock | `ui/menu.py` |
| Dessin du paddock, circuits et résultats | `ui/menu_view.py` |
| Compte à rebours et présentation de course | `ui/race_view.py` |
| Dessin de la piste et des voitures | `rendering/scene.py`, `rendering/gpu.py` |
| Panneaux et commandes du HUD | `rendering/dashboard.py` |
| Pluie, zones humides et couleurs | `rendering/effects.py`, `rendering/theme.py` |

Les vues du menu lisent l’état du contrôleur ; elles ne lancent pas les pilotes. Le tableau de bord fournit les panneaux communs aux deux moteurs de rendu. Les calculs physiques ne chargent ni Pygame ni PyTorch.

## Flux d’une course

`app` traite les événements utilisateur. `LiveRace` démarre le processus de simulation. `PilotPool` interroge chaque pilote dans un processus isolé. Le monde applique les actions, avance la physique et publie un instantané pour l’affichage.

Les décisions arrivent toutes les 50 ms simulées, avec cinq sous-pas physiques de 10 ms. L’enregistrement conserve les états physiques ; l’affichage interpole les poses. Sa file est limitée à deux instantanés pour ne pas accumuler du retard.

## Ressources et sorties

Les cartes sont chargées relativement au package installé, sans dépendre du dossier courant ni d’une modification de `sys.path`. Elles sont incluses dans le wheel et dans le bundle Windows. Le moteur tiers et les cartes conservent leurs sources et leurs licences ; leur code n’est pas soumis au formatage applicatif.

`build/`, `dist/`, `logs/`, `recordings/` et les caches sont des sorties locales ignorées par Git. Les enregistrements de compétition Windows vont dans `%LOCALAPPDATA%/CrashLearn/recordings`. Les poids des pilotes restent à l’emplacement choisi par l’utilisateur.
