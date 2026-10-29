# CPLC Face Search — desktop UI prototype

A front-end-only prototype of the CPLC Face Search investigation workspace, built with
plain HTML, CSS and vanilla JavaScript (no framework, no build step). All data is
mock/synthetic and lives in `data.js`; nothing here talks to a real recognition engine.

## Files

- `index.html` — app shell and markup for all pages
- `styles.css` — dark investigative-tool design system (tokens, components)
- `data.js` — mock datasets, sources, canonical images/occurrences, watchlists,
  search history, index jobs and system health, plus a simulated async `runSearch()`
- `app.js` — navigation, state and all page rendering/interaction logic

## Running it locally

No install or build step is required.

**Option A — just open it**
Double-click `index.html` (or open it from your browser with File → Open).

**Option B — a local server (recommended, avoids any browser file:// quirks)**
From this folder, run one of:

```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

or, with Node installed:

```bash
npx serve .
```

## What's implemented

- Persistent app shell: header, collapsible nav rail, resizable/collapsible left panel
- **Search**: dataset multi-select, "search all" toggle, drag-and-drop or browse image
  upload with live preview, max-results and minimum-similarity controls, simulated
  search progress, sortable/filterable results table, grouped source occurrences with
  expandable per-source metadata cards
- **Match Review**: full-page side-by-side comparison, previous/next + keyboard
  arrow navigation, per-occurrence metadata panels, review status, notes with an
  unsaved-changes warning on navigation away
- **Datasets & Sources**: dataset table with health badges, dataset detail/source
  view, 5-step Create Dataset wizard
- **Watchlists**: list navigation, entry table, create-watchlist modal, "add to
  watchlist" drawer reachable from Search and Match Review
- **Search History**: audit-style table of past searches
- **Import Centre**: 5-step metadata importer mock (source → sheets → map/join →
  validate → confirm) using a schema-agnostic column-mapping UI
- **Index Management**: model/detector/device config, per-dataset index status,
  force-rebuild with a confirmation dialog, job history
- **Administration**: user/role table, audit log, simulated system health
- A "View as" role selector in the header demonstrates permission-aware navigation
  (visual only — note the on-screen reminder that the real backend must enforce this)
- Loading, empty, error and success states throughout; toasts for simulated actions

## Known simplifications (intentional, per the design brief)

- Images are generated placeholder portraits (inline SVG), not real photos
- All "saves", "imports" and "index jobs" are simulated with timers — nothing persists
  beyond the browser session, and nothing is shared between users
- Role switching is a visual demo only — it is not a security boundary

## Next steps toward the real system

Mock-data calls are isolated in `data.js` behind a small service-like API
(`DB.runSearch`, `DB.addWatchlistEntry`, `DB.createWatchlist`, …) so each one can be
swapped for a `fetch()` call to the Flask REST API without restructuring the UI layer.
