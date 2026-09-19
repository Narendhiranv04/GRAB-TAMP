# GRAB-TAMP project page

Static site. To publish on GitHub Pages:

1. Copy everything in this folder to the repository root (or to a `docs/` folder).
2. Commit and push.
3. Repo Settings > Pages > Build and deployment > Source: "Deploy from a branch",
   branch `main`, folder `/ (root)` (or `/docs`).

Files:

- `index.html` — the page
- `support.js`, `diagram-stage.js`, `timeline.js`, `timeline-overview.js` — runtime + animated diagram player
- `assets/*.xml` — the two diagrams
- `videos/*.mp4` — execution clips (lazy-loaded when scrolled into view)
- `.nojekyll` — keeps Pages from filtering files

React, Babel and the diagrams.net viewer load from CDN, so the page needs an
internet connection. The RRC and IIIT logos are hotlinked from robotics.iiit.ac.in.
