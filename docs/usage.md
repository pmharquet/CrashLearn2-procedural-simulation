# Utiliser le simulateur

[Accueil](../README.md) · [Installation](getting-started.md) · [Dépannage](troubleshooting.md)

## Préparer une course

Lancer `python -m crashlearn_sim`, choisir un circuit fixe ou procédural, puis préparer les voitures dans le paddock : nom, couleur et pilote. Dans la sélection du circuit, `COLLISIONS RC` active ou désactive les contacts entre voitures ; lorsqu’elles sont désactivées, les pilotes ne perçoivent plus les autres RC au LiDAR ni dans leurs observations d’adversaires. Les rails restent actifs. Le départ suit le compte à rebours. À l’arrivée, recommencer avec les mêmes paramètres ou revenir au menu.

## Commandes

| Commande | Action |
|---|---|
| **F11** / **Alt+Entrée** | Plein écran ; retour à la taille de fenêtre précédente |
| **Espace** / bouton Pause | Suspendre ou reprendre |
| **+ / −**, **↑ / ↓** | Régler la vitesse de ×1 à ×10 |
| **1…9**, **0** | Choisir directement la vitesse ; `0` = ×10 |
| **Tab** | Alterner caméra embarquée et vue large de toute la route active |
| **H** / bouton Infos | Afficher ou masquer les panneaux ; le cadrage large s’adapte |
| **C** | Changer de voiture suivie |
| **L** / **T** | Afficher les mesures LiDAR / les trajectoires des IA |
| **W** | Changer de mode météo |
| **R** | Recommencer avec les mêmes paramètres |
| **E** | Démarrer ou arrêter l’enregistrement |
| **V** | Lire le dernier enregistrement / revenir au direct |
| **← / →** en replay | Reculer ou avancer de 1 s ; **Shift** : 10 s |
| **Début / Fin** en replay | Première / dernière image |
| **Échap** | Quitter le plein écran ou revenir au menu pendant la course |

La fenêtre accepte le redimensionnement et l’agrandissement natifs. `--fullscreen` démarre en plein écran ; `--window-size 1000 700` choisit uniquement la taille initiale.

Le HUD indique la vitesse demandée **et celle réellement atteinte**, qui dépend de la machine. L’accélération ne change pas le pas physique. En vue large, la limite jaune marque la génération et la limite rouge la suppression de la route.


## Météo et circuits

### Choisir la météo

| Mode | Comportement |
|---|---|
| `original` | Sec : adhérence 1, bruit LiDAR de ±0,1 % |
| `cycle` *(défaut)* | Sec → pluie → orage → éclaircie, toutes les 20 s simulées |
| `dynamic` | Régime aléatoire différent du précédent, toutes les 12 à 32 s simulées |
| `stress` | Adhérence cible aléatoire entre 0,55 et 1 toutes les 20 s simulées |

En mode dynamique, l’adhérence cible vaut 0,92–1 au sec, 0,70–0,87 sous la pluie et 0,55–0,65 pendant l’orage. Les transitions sont progressives. Les modes perturbés peuvent atteindre ±5 % de bruit LiDAR et 1 % de pertes de mesure.

Pluie, zones humides fixes sur la chaussée, projections et éclairs accompagnent la météo. Le hasard du rendu est indépendant de celui de la physique et des capteurs.

### Choisir le tracé

La graine permet de retrouver une génération. Le profil `mixed` mélange les difficultés ; `flowing` favorise les portions faciles ; `technical` favorise les enchaînements serrés. Tous conservent plusieurs types de sections.

Le générateur combine dix familles : lignes droites, courbes ouvertes, angles droits, doubles coudes, chicanes, triples chicanes, esses, épingles à 180°, doubles épingles et boucles ouvertes de 195 à 250°. Les inversions de courbure peuvent s’enchaîner sans ligne droite intermédiaire.

La route mesure **4,8 m de large**. Les épingles descendent à **2,55 m de rayon**, soit environ 36 cm entre les rails des deux branches au minimum. Les candidats qui se croisent ou se rapprochent trop sont rejetés.

Le simulateur conserve 160 m devant la tête et 50 m derrière la dernière voiture active. Un tampon de 180 m supplémentaires peut être recalculé hors champ en cas d’impasse, sans déplacer les rails déjà publiés. Le retard maximal de 100 m entre voitures borne la carte en mémoire.

## Enregistrer et revoir une course

Pendant la course, **E** démarre ou arrête l’enregistrement. **V** ouvre le dernier enregistrement disponible dans la session.

Sous Windows, les fichiers de compétition sont dans `%LOCALAPPDATA%/CrashLearn/recordings`. Ils contiennent géométrie, poses, météo, mesures et événements. La relecture fonctionne sans charger de pilote. Un fichier existant n’est jamais écrasé.

L’écriture SQLite est progressive, validée tous les 100 pas et à la fermeture. La taille du fichier augmente avec la durée ; la course n’est pas accumulée en mémoire.

## Entraîner un pilote

Depuis la racine du dépôt :

```powershell
.venv/Scripts/python -m crashlearn_sim.training --steps 200000 --cars 1 --weather cycle --output recordings/procedural_ppo
```

Ajouter `--no-vehicle-collisions` pour entraîner sans contacts ni perception des autres véhicules. Sans cette option, les collisions et leur perception sont activées.

Sélectionner ensuite `recordings/procedural_ppo.zip` comme pilote dans le paddock. Avec plusieurs voitures, passer `--agent CHEMIN` pour charger les adversaires ; sans chemin explicite, définir `CRASHLEARN_AGENT_DIR` avec le dossier du pilote.

`ProceduralEnv` expose 131 observations normalisées : LiDAR, vitesse, braquage, adversaires et adhérence. Deux actions pilotent vitesse et direction. La récompense valorise la progression et pénalise les contacts ainsi que, légèrement, le braquage.

L’épisode finit lorsque la voiture entraînée abandonne ou atteint l’arrivée optionnelle (`--finish-distance`). Il est tronqué à 4 000 décisions par défaut. PPO apprend une nouvelle politique ; il ne réentraîne pas l’ONNX importé. Le paddock accepte aussi les anciennes politiques à 102 observations.

## Affichage et performances

`--renderer auto` choisit OpenGL 3.3 et se replie sur le rendu logiciel si nécessaire. `--renderer gpu` exige OpenGL ; `--renderer software` utilise Pygame.

La simulation tourne dans un processus séparé : décisions à 20 Hz et physique à 100 Hz. L’affichage interpole les poses sans modifier les observations, les collisions ni les enregistrements. Le HUD distingue les FPS, les pas de simulation par seconde et les appels IA par seconde. La vitesse réellement atteinte dépend de la machine et des pilotes.

Pour les tests et la construction Windows, consulter le [guide développeur](development.md). Le [contrat historique](reference/legacy-api.md) documente les outils d’enregistrement et de relecture du moteur d’origine.
