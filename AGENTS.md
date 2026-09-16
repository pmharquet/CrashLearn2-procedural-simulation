\## Git workflow



\*\*Aucune initiative git non demandée\*\* : ne créer de branche, commiter ou pousser que sur demande explicite de l'utilisateur ; par défaut, travailler sur la branche courante. Si la branche courante est `main` ou `dev`, ne jamais y commiter : demander à l'utilisateur sur quelle branche mettre le travail (nouvelle ou existante) avant tout commit.



\### Commits



\- Suivre la convention \*\*Conventional Commits\*\* (style Angular) : `type(scope): description`.

\- Types : `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `style`, `perf`, `build`, `ci`.

\- Description à l'impératif, concise, en minuscule (ex. `fix(samsic): corrige les candidatures en 404`).

\- Un commit correspond à \*\*une seule modification précise\*\*, entièrement décrite par son titre.

\- \*\*Ne jamais\*\* s'ajouter en co-auteur (pas de trailer `Co-Authored-By: Claude`).



\### Branches



\- Format : `type/description-en-kebab-case` (ex. `feat/search-section`, `fix/episode-enrichment-feed`).

\- Types alignés sur les Conventional Commits : `feat`, `fix`, `chore`, `refactor`, `hotfix`, `docs`, `test`, `perf`, `build`, `ci`.

\- Description en \*\*minuscules\*\*, mots séparés par des tirets ; pas d'accents ni d'espaces.

\- Brancher depuis `dev` (branche principale) et cibler `dev` pour la PR.

\- Branches longue durée réservées : `dev`, `main`, ne pas les recréer en branche de travail.



\### Pull requests



\- Remplir le template `.github/pull\_request\_template.md` (structure et titres inchangés).

\- Laisser \*\*vides\*\* les champs qui ne peuvent pas être connus depuis le code/diff.

\- Cocher uniquement les cases de la checklist réellement vérifiables ; laisser les autres décochées.

\- Rester \*\*bref et efficace\*\* : pas de pavés, des puces courtes et factuelles.

\- \*\*Ne jamais\*\* s'ajouter en co-auteur.



\### Labels



\- Toute issue et toute PR porte \*\*au moins un label\*\*.

\- Récupérer la liste réelle avant d'en poser un (`gh label list`) : elle évolue avec le projet, ne jamais la deviner.

\- \*\*Ne jamais\*\* créer de label sans demande explicite.

