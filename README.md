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


## Grand Prix procédural

Installation locale (Python 3.12 recommandé) :

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-procedural.txt
.venv/Scripts/python procedural_demo.py
```

La fenêtre démarre avec **4 voitures, météo cyclique, route mixte et vitesse ×3**.
Chaque voiture utilise une instance indépendante du véritable agent ONNX
`../T-AIA-901-NCY-9-1-crashlearn-2/submission/champion`. Le modèle n'est pas copié
et le dépôt voisin n'est pas modifié. `--agent CHEMIN` charge une autre soumission.
Aucun pilote de remplacement n'est activé silencieusement.

### Systèmes adaptés

| Système | Mode procédural |
|---|---|
| Physique | F1TENTH single-track, RK4 à 100 Hz, commandes à 20 Hz, délai de braquage de 20 ms |
| Météo | Adhérence réellement appliquée aux pneus, transitions progressives, pluie et orage visibles |
| LiDAR | 100 rayons de −π à +π, intersections avec les rails, portée 15 m, bruit et pertes de mesure |
| Adversaires | 1 à 4 voitures ; positions, angles et vitesses relatives dans les observations |
| Collisions | Empreinte du véhicule contrôlée à 100 Hz ; arrêt contre les rails, séparation et impulsion entre voitures |
| Course | Classement, secteurs de 100 m, derniers chronos et meilleur secteur, arrivée optionnelle |
| Abandon | 4 s sans progression ; retrait également si retard supérieur à 100 m |
| Cartes | Graines reproductibles, profils mixed / flowing / technical, changement à chaud avec nouvelle course |
| Enregistrement | Écriture progressive SQLite, relecture sans modèle IA, pause, recherche et lecture en boucle |

Le LiDAR reste **wall-only**, comme celui fourni à l'agent d'origine : les autres
voitures sont transmises dans `opponents`. L'interface affiche les rayons réellement
mesurés, y compris le bruit et les mesures défaillantes. Aucune géométrie de carte
n'est fournie au pilote. Les contacts ne terminent pas immédiatement un épisode :
la voiture peut reculer pour se dégager. La démo suit un pilote encore actif si
celui observé abandonne, et relance une course quand tous ont terminé ou abandonné.

La course infinie utilise des **secteurs**, pas des tours de circuit fermé.
`lap_count` correspond au nombre de secteurs et `progress` à l'avancement normalisé
dans le secteur. `distance_m` contient la distance totale. Les 32 derniers temps
par voiture sont conservés. `--finish-distance 1000` définit une arrivée à 1 km
mesurée depuis la position de départ de chaque voiture.

La route conserve 160 m devant la tête et 50 m derrière la dernière voiture active.
Le retard est limité à 100 m. Un tampon supplémentaire de 180 m est planifié hors
champ : en cas d'impasse, le générateur peut le reconstruire sans déplacer les rails
déjà publiés. La carte, les traces, événements et chronos restent bornés en mémoire.

Le générateur compose des arcs à courbure constante, reliés par des transitions de
courbure sur 1 m. Il n'impose plus d'axe global ni de retour à une direction cible.
Dix familles sont disponibles : lignes droites, courbes ouvertes, angles droits,
doubles coudes, chicanes, triples chicanes, esses, épingles à 180° doubles épingles et boucles ouvertes de 195 à 250°.
Les changements de sens dans une chicane s'enchaînent sans ligne droite obligatoire.
Les profils mixed / flowing / technical modifient leurs probabilités ; tous gardent
des portions faciles et techniques. Largeur : 4,8 m ; rayon minimal : 2,55 m pour les épingles (environ 36 cm entre les rails des deux branches au minimum).

Chaque section est contrôlée contre les portions existantes et contre elle-même,
avec une marge de 0,25 m entre les bandes de route non voisines. Les candidats qui
se croisent ou se rapprochent trop sont rejetés. Des arcs de raccord et une recherche
dans le tampon invisible permettent de contourner les impasses. Les cartes fixes du
simulateur historique restent accessibles via son API `set_map()`.

### Interface et commandes

| Commande | Action |
|---|---|
| F11 / Alt+Entrée | Basculer en plein écran ; retour à la taille de fenêtre précédente |
| Espace / bouton Pause | Suspendre ou reprendre |
| `+`, `-`, flèches haut/bas | Régler ×1 à ×10 sans changer le pas physique |
| `1` à `9`, `0` | Vitesse directe, `0` = ×10 |
| Tab | Vue large de toute la route active, avec panneaux latéraux ; jaune : génération, rouge : suppression |
| Bouton Infos / H | Afficher ou masquer les panneaux ; la vue large utilise la place libérée |
| L | Afficher les mesures LiDAR du véhicule suivi |
| T | Afficher / masquer les trajectoires colorées des IA |
| C | Changer de voiture suivie |
| W | Changer le mode météo |
| P | Changer de profil de route et démarrer une nouvelle graine |
| R | Nouvelle course |
| E | Démarrer / arrêter et sauvegarder l'enregistrement |
| V | Lire le dernier enregistrement / revenir au direct |
| Gauche / droite en replay | Reculer / avancer de 1 s ; Shift : 10 s |
| Début / Fin en replay | Première / dernière image |
| Échap | Quitter le plein écran, ou fermer en mode fenêtre |

Le HUD montre le multiplicateur demandé **et réellement atteint** selon la machine,
la vitesse, la position, les secteurs, les contacts, la météo, l'adhérence et la
qualité du LiDAR. La pluie, les reflets de route, les projections d'eau, les éclairs
et les impacts sont des effets visuels indépendants du hasard de simulation.
Les voitures sont légèrement agrandies à l'écran, sans changer leurs collisions.

### Météo

- `original` : sec, adhérence 1 et bruit de ±0,1 %.
- `dynamic` : régimes sec (0,92–1), pluie (0,70–0,87) et orage (0,55–0,65)
  tirés aléatoirement, sans répétition immédiate du régime. Durée variable de 12 à
  32 secondes simulées ; bruit et pertes LiDAR liés à l’intensité comme en mode stress.
- `stress` : adhérence cible entre 0,55 et 1 toutes les 20 secondes ; bruit jusqu'à
  ±5 % et jusqu'à 1 % de pertes LiDAR selon l'intensité.
- `cycle` : alternance sec / pluie / orage / éclaircie toutes les 20 secondes,
  avec les mêmes perturbations de capteur que le mode stress.

Les transitions d'adhérence sont progressives. Les mesures sont stables entre
plusieurs lectures d'un même pas ; les générateurs météo, route et capteurs sont
séparés. Le rendu et l'enregistrement ne changent donc pas le résultat d'une course.

```powershell
.venv/Scripts/python procedural_demo.py --cars 4 --weather stress --profile technical --speed 5
.venv/Scripts/python procedural_demo.py --record recordings/course.sqlite
.venv/Scripts/python procedural_demo.py --replay recordings/course.sqlite
```

Les enregistrements sauvegardent la géométrie, les poses, la météo, les mesures et
les événements réels. Ils sont écrits progressivement et n'accumulent pas la course
en mémoire ; leur taille sur disque augmente avec la durée. Les données sont
validées sur disque tous les 100 pas et à la fermeture. Un fichier existant n'est
jamais écrasé. La relecture fonctionne sans accès au dépôt de l'IA.

### Entraînement

```powershell
.venv/Scripts/python procedural_train.py --steps 200000 --cars 4 --weather cycle --output recordings/procedural_ppo
.venv/Scripts/python procedural_demo.py --ppo recordings/procedural_ppo.zip
```

`ProceduralEnv` expose Gymnasium avec **131 observations** normalisées : LiDAR,
vitesse, braquage, 4 × 7 caractéristiques d'adversaires et adhérence. Les deux actions
normalisées pilotent vitesse et braquage. Les adversaires utilisent l'agent importé.
La récompense valorise l'avancement et pénalise les contacts et légèrement le braquage.
L'épisode finit quand le véhicule entraîné abandonne ou atteint l'arrivée optionnelle ;
il est tronqué à 4 000 décisions par défaut. Les graines changent entre épisodes.
Le script entraîne une nouvelle politique PPO ; il ne réentraîne pas l'ONNX importé.
La démo accepte aussi les anciennes politiques à 102 observations, sans informations
d'adversaires ni d'adhérence.

Le paquet `gym` historique est requis pour la détection optionnelle de
Stable-Baselines3 dans ce dépôt contenant un dossier `gym/`. L'environnement utilise
bien Gymnasium. Le simulateur historique n'a pas été modifié.

### Vérification

```powershell
.venv/Scripts/python -m unittest test_procedural -v
.venv/Scripts/python procedural_demo.py --headless --cars 4 --weather stress --steps 10000
```

Les tests couvrent géométrie et mémoire, graines, contrats LiDAR, transitions météo,
impact physique de l'adhérence, adversaires, contacts et dégagement, abandon, secteurs,
arrivée, observations Gymnasium et fidélité de l'enregistrement / relecture.

La fenêtre se redimensionne librement et accepte l’agrandissement natif du système.
L’interface est mise à l’échelle pour rester visible dans les petites fenêtres.
`--fullscreen` démarre en plein écran ; `--window-size 1000 700` définit la taille initiale.

`--overview` démarre directement en vue large. Cette vue ajuste le cadrage à toute
la portion active entre les panneaux d’information, y compris les retours à 180°,
et garde une orientation fixe. Les panneaux sont visibles par défaut dans les deux vues et peuvent être masqués avec le bouton Infos.
Les références de circuits servent à comparer les formes ; elles ne sont pas recopiées.

Pour examiner une série de graines sans sélectionner les meilleurs résultats :

```powershell
.venv/Scripts/python capture_road_overviews.py --output recordings/overviews
.venv/Scripts/python procedural_demo.py --overview --seed 0 --weather dynamic
```

Le script produit les graines 0 à 5, à l'arrêt, avec le même rendu que l'application.
Les tests vérifient notamment les demi-tours à 180°, les quatre inversions d'une
triple chicane, les intersections des rails et l'immuabilité de la route publiée
pendant une réparation du tampon. Ces mesures ne jugent pas la qualité visuelle.

Les zones humides sont de petites surfaces hors du marquage central, ancrées à
la distance absolue du circuit pour rester fixes pendant la suppression de segments.
Les gouttes sont des particules aux positions, vitesses et longueurs indépendantes,
reproductibles à la relecture et sans effet sur le hasard du simulateur.
