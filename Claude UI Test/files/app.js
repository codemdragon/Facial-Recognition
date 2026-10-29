/* =========================================================================
   CPLC Face Search — application logic (vanilla JS, no framework)
   ========================================================================= */

const App = (() => {

  const state = {
    role: 'Admin',
    page: 'search',
    selectedDatasets: new Set(DB.datasets.map(d => d.id)),
    searchAll: true,
    queryImage: null,           // { name, dims, dataUrl }
    maxResults: 50,
    minSimilarity: 70,
    results: null,              // array | 'loading' | 'error' | null
    expandedRows: new Set(),
    sortKey: 'similarity',
    sortDir: 'desc',
    filterText: '',

    reviewList: [],             // current result set used for review nav
    reviewIndex: 0,
    reviewNotesDirty: false,

    activeWatchlist: null,
    activeDataset: null,

    pendingWatchlistTarget: null, // image for "add to watchlist" drawer
  };

  // ------------------------------------------------------------------ utils
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'class') node.className = v;
      else if (k === 'html') node.innerHTML = v;
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    });
    (Array.isArray(children) ? children : [children]).forEach(c => {
      if (c == null) return;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  };
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));

  function toast(message, kind = '') {
    const stack = $('#toastStack');
    const t = el('div', { class: `toast ${kind}` }, message);
    stack.appendChild(t);
    setTimeout(() => t.remove(), 3600);
  }

  // Deterministic placeholder "photo" — an SVG data URI so the prototype needs no network images.
  function placeholderPortrait(seed, w = 160, h = 160) {
    let hash = 0;
    for (let i = 0; i < String(seed).length; i++) hash = (hash * 31 + String(seed).charCodeAt(i)) >>> 0;
    const hue = hash % 360;
    const hue2 = (hue + 40) % 360;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 160 160">
      <defs><linearGradient id="g${hash}" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="hsl(${hue},28%,24%)"/><stop offset="1" stop-color="hsl(${hue2},22%,16%)"/>
      </linearGradient></defs>
      <rect width="160" height="160" fill="url(#g${hash})"/>
      <circle cx="80" cy="64" r="30" fill="hsl(${hue},18%,42%)"/>
      <ellipse cx="80" cy="150" rx="52" ry="46" fill="hsl(${hue},18%,38%)"/>
    </svg>`;
    return 'data:image/svg+xml;utf8,' + encodeURIComponent(svg);
  }

  // ------------------------------------------------------------------ nav
  function applyRoleVisibility() {
    $$('.nav-item[data-page]').forEach(btn => {
      const roles = btn.dataset.roles.split(',');
      btn.style.display = roles.includes(state.role) ? '' : 'none';
    });
    $('#headerUserRole').textContent = state.role;
    // if current page no longer permitted, bounce to search
    const activeBtn = $(`.nav-item[data-page="${state.page}"]`);
    if (activeBtn && activeBtn.style.display === 'none') goToPage('search');
  }

  function goToPage(page) {
    if (state.page === 'review' && state.reviewNotesDirty) {
      const ok = confirm('You have unsaved review notes on this match. Leave without saving?');
      if (!ok) return;
      state.reviewNotesDirty = false;
    }
    state.page = page;
    $$('.page').forEach(p => p.classList.remove('active'));
    const target = $(`#page-${page}`);
    if (target) target.classList.add('active');
    $$('.nav-item[data-page]').forEach(b => b.classList.toggle('active', b.dataset.page === page));
    const crumbs = {
      search: 'Search', datasets: 'Datasets & Sources', review: 'Match Review',
      watchlists: 'Watchlists', history: 'Search History', import: 'Import Centre',
      index: 'Index Management', admin: 'Administration & System Health', wizard: 'Datasets & Sources / New dataset',
    };
    $('#crumb').textContent = crumbs[page] || '';
    renderPage(page);
  }

  function renderPage(page) {
    if (page === 'search') renderSearchLeft();
    if (page === 'datasets') renderDatasets();
    if (page === 'watchlists') renderWatchlists();
    if (page === 'history') renderHistory();
    if (page === 'import') renderImport();
    if (page === 'index') renderIndexMgmt();
    if (page === 'admin') renderAdmin();
    if (page === 'review') renderReview();
  }

  // ================================================================== SEARCH
  function renderSearchLeft() {
    const list = $('#datasetChecklist');
    list.innerHTML = '';
    DB.datasets.forEach(ds => {
      const row = el('label', { class: 'checkbox-row' });
      const cb = el('input', { type: 'checkbox' });
      cb.checked = state.selectedDatasets.has(ds.id);
      cb.disabled = state.searchAll;
      cb.addEventListener('change', () => {
        if (cb.checked) state.selectedDatasets.add(ds.id); else state.selectedDatasets.delete(ds.id);
        updateRunButton();
      });
      row.appendChild(cb);
      row.appendChild(document.createTextNode(ds.name));
      row.appendChild(el('span', { class: 'count' }, `${ds.images.toLocaleString()} imgs`));
      list.appendChild(row);
    });
    $('#searchAllToggle').checked = state.searchAll;
    updateRunButton();
    renderSearchResults();
  }

  function updateRunButton() {
    const btn = $('#runSearchBtn');
    const dsOk = state.searchAll || state.selectedDatasets.size > 0;
    btn.disabled = !(state.queryImage && dsOk) || state.results === 'loading';
  }

  function wireSearchPage() {
    $('#searchAllToggle').addEventListener('change', (e) => {
      state.searchAll = e.target.checked;
      renderSearchLeft();
    });

    const dz = $('#dropzone');
    const fileInput = $('#fileInput');
    $('#browseBtn').addEventListener('click', () => fileInput.click());
    dz.addEventListener('click', () => fileInput.click());
    ['dragenter', 'dragover'].forEach(evt => dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach(evt => dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
    dz.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) handleFile(f); });
    fileInput.addEventListener('change', (e) => { const f = e.target.files[0]; if (f) handleFile(f); });

    $('#replaceBtn').addEventListener('click', () => fileInput.click());
    $('#removeBtn').addEventListener('click', () => {
      state.queryImage = null;
      $('#previewArea').style.display = 'none';
      $('#dropzone').style.display = '';
      updateRunButton();
    });

    $('#maxResults').addEventListener('change', (e) => state.maxResults = e.target.value);
    const simSlider = $('#simSlider'), simValue = $('#simValue');
    simSlider.addEventListener('input', () => { state.minSimilarity = Number(simSlider.value); simValue.textContent = simSlider.value + '%'; });

    $('#runSearchBtn').addEventListener('click', runSearch);
    $('#resetSearchBtn').addEventListener('click', () => {
      state.selectedDatasets = new Set(DB.datasets.map(d => d.id));
      state.searchAll = true;
      state.maxResults = 50; $('#maxResults').value = '50';
      state.minSimilarity = 70; simSlider.value = 70; simValue.textContent = '70%';
      state.results = null;
      renderSearchLeft();
    });

    // resizable divider
    const divider = $('#searchDivider'), leftPanel = $('#searchLeft');
    let dragging = false;
    divider.addEventListener('mousedown', () => { dragging = true; divider.classList.add('dragging'); document.body.style.userSelect = 'none'; });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const railWidth = $('#rail').getBoundingClientRect().width;
      const w = Math.min(520, Math.max(280, e.clientX - railWidth));
      leftPanel.style.width = w + 'px';
    });
    window.addEventListener('mouseup', () => { dragging = false; divider.classList.remove('dragging'); document.body.style.userSelect = ''; });

    $('#searchLeftCollapse').addEventListener('click', () => {
      $('#searchSplit').classList.toggle('collapsed');
    });
  }

  function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        state.queryImage = { name: file.name, dims: `${img.naturalWidth} × ${img.naturalHeight}px`, dataUrl: e.target.result };
        $('#previewImg').src = e.target.result;
        $('#previewName').textContent = file.name;
        $('#previewDims').textContent = state.queryImage.dims;
        $('#previewArea').style.display = 'flex';
        $('#dropzone').style.display = 'none';
        updateRunButton();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  async function runSearch() {
    state.results = 'loading';
    state.expandedRows = new Set();
    renderSearchResults();
    updateRunButton();
    try {
      const datasetIds = state.searchAll ? DB.datasets.map(d => d.id) : Array.from(state.selectedDatasets);
      const results = await DB.runSearch({ datasetIds, maxResults: state.maxResults, minSimilarity: state.minSimilarity });
      state.results = results;
      toast(`Search complete — ${results.length} unique matches found.`, 'success');
    } catch (err) {
      state.results = 'error';
      toast('Search failed — simulated backend error.', 'error');
    }
    renderSearchResults();
    updateRunButton();
  }

  function renderSearchResults() {
    const area = $('#searchResults');
    area.innerHTML = '';

    if (!state.queryImage) {
      area.appendChild(stateBlock('⌕', 'No query image yet', 'Upload or drop an image in the left panel, then run a search to see candidate matches here.'));
      return;
    }
    if (state.results === null) {
      area.appendChild(stateBlock('▷', 'Ready to search', 'Review your dataset and threshold selections, then click Run search.'));
      return;
    }
    if (state.results === 'loading') {
      const block = el('div', { class: 'state-block' }, [
        el('div', { class: 'spinner' }),
        el('div', { class: 'state-block__title', style: 'margin-top:16px;' }, 'Searching permitted datasets…'),
        el('div', { class: 'state-block__body' }, 'Comparing the query image against indexed faces. This is simulated latency.'),
      ]);
      area.appendChild(block);
      return;
    }
    if (state.results === 'error') {
      const block = stateBlock('⚠', 'Search failed', 'A simulated backend error occurred. Try running the search again.');
      const retry = el('button', { class: 'btn btn-primary', style: 'margin-top:12px;', onclick: runSearch }, 'Retry search');
      block.appendChild(retry);
      area.appendChild(block);
      return;
    }

    let results = [...state.results];
    if (state.filterText) {
      const q = state.filterText.toLowerCase();
      results = results.filter(r => r.id.toLowerCase().includes(q) ||
        r.occurrences.some(o => o.linkedName.toLowerCase().includes(q) || o.caseRef.toLowerCase().includes(q)));
    }
    results.sort((a, b) => {
      let va, vb;
      if (state.sortKey === 'similarity') { va = a.similarity; vb = b.similarity; }
      else if (state.sortKey === 'occurrences') { va = a.occurrences.length; vb = b.occurrences.length; }
      else { va = a.id; vb = b.id; }
      const dir = state.sortDir === 'asc' ? 1 : -1;
      return va > vb ? dir : va < vb ? -dir : 0;
    });
    state.reviewList = results;

    const totalOcc = state.results.reduce((s, r) => s + r.occurrences.length, 0);

    // summary
    const summary = el('div', {}, [
      el('div', { class: 'page-title' }, 'Search results'),
      el('div', { class: 'page-subtitle' }, `Datasets: ${state.searchAll ? 'All permitted' : Array.from(state.selectedDatasets).map(id => DB.getDataset(id).name).join(', ')} · Completed just now`),
    ]);
    area.appendChild(summary);

    const stats = el('div', { class: 'stat-row' }, [
      el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value' }, String(state.results.length)), el('div', { class: 'stat-card__label' }, 'Unique matches')]),
      el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value' }, String(totalOcc)), el('div', { class: 'stat-card__label' }, 'Source occurrences')]),
    ]);
    area.appendChild(stats);

    // toolbar
    const toolbar = el('div', { class: 'toolbar' });
    const searchBox = el('div', { class: 'search-box' });
    const si = el('input', { class: 'input', placeholder: 'Search within results…' });
    si.value = state.filterText;
    si.addEventListener('input', () => { state.filterText = si.value; renderSearchResults(); });
    searchBox.appendChild(si);
    toolbar.appendChild(searchBox);
    toolbar.appendChild(el('div', { class: 'toolbar__spacer' }));
    toolbar.appendChild(el('span', { class: 'text-small text-secondary' }, 'Illustrative similarity scores — ranking only, not identity probability.'));
    area.appendChild(toolbar);

    if (results.length === 0) {
      area.appendChild(stateBlock('∅', 'No matches at this threshold', 'Try lowering the minimum similarity or widening the dataset selection.'));
      return;
    }

    area.appendChild(buildResultsTable(results));
  }

  function stateBlock(icon, title, body) {
    return el('div', { class: 'state-block' }, [
      el('div', { class: 'state-block__icon', style: 'font-size:34px;' }, icon),
      el('div', { class: 'state-block__title' }, title),
      el('div', { class: 'state-block__body' }, body),
    ]);
  }

  function buildResultsTable(results) {
    const wrap = el('div', { class: 'table-wrap' });
    const table = el('table');
    const cols = [
      ['preview', 'Preview', false], ['id', 'Image ID', true], ['similarity', 'Similarity', true],
      ['occurrences', 'Occurrences', true], ['source', 'Dataset / source', false],
      ['linked', 'Linked details', false], ['status', 'Review status', false], ['actions', 'Actions', false],
    ];
    const thead = el('thead');
    const tr = el('tr');
    cols.forEach(([key, label, sortable]) => {
      const th = el('th', {}, label);
      if (sortable) {
        th.classList.toggle('sorted', state.sortKey === key);
        th.addEventListener('click', () => {
          if (state.sortKey === key) state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
          else { state.sortKey = key; state.sortDir = 'desc'; }
          renderSearchResults();
        });
      }
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);

    const tbody = el('tbody');
    results.forEach((img, idx) => {
      const primary = img.occurrences[0];
      const row = el('tr', { class: 'clickable' });
      row.appendChild(el('td', {}, el('img', { class: 'thumb', src: placeholderPortrait(img.id), alt: '' })));
      row.appendChild(el('td', {}, img.id));
      row.appendChild(el('td', {}, `${img.similarity.toFixed(1)}%`));
      const occCell = el('td');
      const occBtn = el('button', { class: 'btn btn-sm btn-ghost', onclick: (e) => { e.stopPropagation(); toggleRow(img.id); } },
        `${img.occurrences.length} occurrence${img.occurrences.length > 1 ? 's' : ''} ${state.expandedRows.has(img.id) ? '▴' : '▾'}`);
      occCell.appendChild(occBtn);
      row.appendChild(occCell);
      row.appendChild(el('td', {}, [
        document.createTextNode(primary.dataset),
        el('div', { class: 'cell-sub' }, primary.source),
      ]));
      row.appendChild(el('td', {}, [
        document.createTextNode(primary.linkedName),
        el('div', { class: 'cell-sub' }, primary.caseRef),
      ]));
      row.appendChild(el('td', {}, el('span', { class: 'badge neutral' }, img.reviewStatus)));
      const actionsCell = el('td');
      const openBtn = el('button', { class: 'btn btn-sm', onclick: (e) => { e.stopPropagation(); openReview(results, idx); } }, 'Open review');
      const wlBtn = el('button', { class: 'btn btn-sm btn-ghost', onclick: (e) => { e.stopPropagation(); openWatchlistDrawer(img); } }, 'Watchlist');
      actionsCell.appendChild(openBtn); actionsCell.appendChild(wlBtn);
      row.appendChild(actionsCell);
      row.addEventListener('click', () => openReview(results, idx));
      tbody.appendChild(row);

      if (state.expandedRows.has(img.id)) {
        const occRow = el('tr', { class: 'occ-row' });
        const td = el('td', { colspan: '8' });
        const wrapOcc = el('div', { class: 'occ-wrap' });
        const grid = el('div', { class: 'occ-grid' });
        img.occurrences.forEach(o => grid.appendChild(occurrenceCard(o)));
        wrapOcc.appendChild(el('div', { class: 'text-small text-secondary', style: 'margin-bottom:10px;' },
          'Each card below is a separate source record. Names and case references are not assumed to refer to the same identity.'));
        wrapOcc.appendChild(grid);
        td.appendChild(wrapOcc);
        occRow.appendChild(td);
        tbody.appendChild(occRow);
      }
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function occurrenceCard(o) {
    const card = el('div', { class: 'occ-card' });
    card.appendChild(el('div', { class: 'occ-card__title' }, `${o.dataset} · ${o.caseRef}`));
    const dl = el('dl');
    [['Source', o.source], ['Original file', o.filename], ['Source path', o.path],
     ['Linked name', o.linkedName], ['Record ID', o.recordId], ['Imported', o.imported]].forEach(([k, v]) => {
      dl.appendChild(el('dt', {}, k)); dl.appendChild(el('dd', {}, v));
    });
    card.appendChild(dl);
    card.appendChild(el('button', { class: 'btn btn-sm btn-ghost', style: 'margin-top:8px;', onclick: () => toast(`Opening source record ${o.recordId} — simulated.`) }, 'Open source record'));
    return card;
  }

  function toggleRow(id) {
    if (state.expandedRows.has(id)) state.expandedRows.delete(id); else state.expandedRows.add(id);
    renderSearchResults();
  }

  // ================================================================== REVIEW
  function openReview(list, index) {
    state.reviewList = list;
    state.reviewIndex = index;
    state.reviewNotesDirty = false;
    goToPage('review');
  }

  function renderReview() {
    const area = $('#reviewContent');
    area.innerHTML = '';
    if (!state.reviewList.length) {
      area.appendChild(stateBlock('◧', 'No match selected', 'Run a search and open a result to review it here.'));
      return;
    }
    const img = state.reviewList[state.reviewIndex];

    const nav = el('div', { class: 'review-nav' }, [
      el('button', { class: 'btn btn-sm', onclick: () => stepReview(-1) }, '‹ Previous'),
      el('div', { class: 'text-small text-secondary' }, `Match ${state.reviewIndex + 1} of ${state.reviewList.length}`),
      el('button', { class: 'btn btn-sm', onclick: () => stepReview(1) }, 'Next ›'),
    ]);
    area.appendChild(nav);

    const images = el('div', { class: 'review-images' }, [
      el('div', { class: 'review-image-card' }, [
        el('div', { class: 'review-image-card__frame' }, el('img', { src: state.queryImage ? state.queryImage.dataUrl : placeholderPortrait('query') })),
        el('div', { class: 'review-image-card__bar' }, [
          el('span', { class: 'text-small text-secondary' }, state.queryImage ? state.queryImage.name : 'Sample query image'),
          el('button', { class: 'btn btn-sm btn-ghost' }, 'Enlarge'),
        ]),
      ]),
      el('div', { class: 'review-image-card' }, [
        el('div', { class: 'review-image-card__frame' }, el('img', { src: placeholderPortrait(img.id) })),
        el('div', { class: 'review-image-card__bar' }, [
          el('span', { class: 'text-small text-secondary' }, `${img.id} · ${img.occurrences.length} source occurrence${img.occurrences.length > 1 ? 's' : ''}`),
          el('button', { class: 'btn btn-sm btn-ghost' }, 'Enlarge'),
        ]),
      ]),
    ]);
    area.appendChild(images);

    const twoCol = el('div', { class: 'two-col' });

    // left: occurrences
    const left = el('div');
    left.appendChild(el('div', { class: 'section-title' }, 'Source occurrences'));
    left.appendChild(el('div', { class: 'text-small text-secondary', style: 'margin-bottom:10px;' }, 'Select an occurrence to view its full metadata independently. Names and references are kept separate per source.'));
    const grid = el('div', { class: 'occ-grid' });
    img.occurrences.forEach(o => grid.appendChild(occurrenceCard(o)));
    left.appendChild(grid);
    twoCol.appendChild(left);

    // right: review decision panel
    const right = el('div');
    const panel = el('div', { class: 'panel' });
    panel.appendChild(el('div', { class: 'section-title' }, 'Candidate details'));
    panel.appendChild(el('div', { class: 'kv' }, [el('dt', {}, 'Similarity ranking'), el('dd', {}, `${img.similarity.toFixed(1)}%`)]));
    panel.appendChild(el('div', { class: 'text-small text-secondary', style: 'margin:6px 0 14px;' }, 'This is a ranking score, not a probability that the two images depict the same person.'));

    panel.appendChild(el('div', { class: 'section-title' }, 'Review decision'));
    ['Unreviewed', 'Inconclusive', 'Reviewed'].forEach(status => {
      const row = el('label', { class: 'radio-row' });
      const radio = el('input', { type: 'radio', name: 'reviewStatus' });
      radio.checked = img.reviewStatus === status;
      radio.addEventListener('change', () => { img.reviewStatus = status; state.reviewNotesDirty = true; });
      row.appendChild(radio);
      row.appendChild(document.createTextNode(status === 'Inconclusive' ? 'Inconclusive — needs further examination' : status === 'Reviewed' ? 'Reviewed — assessment recorded' : status));
      panel.appendChild(row);
    });

    const notesLabel = el('label', { class: 'field__label', style: 'margin-top:12px;display:block;' }, 'Reviewer notes');
    const notes = el('textarea', { class: 'textarea' });
    notes.value = img.notes || '';
    notes.addEventListener('input', () => { img.notes = notes.value; state.reviewNotesDirty = true; });
    panel.appendChild(notesLabel);
    panel.appendChild(notes);

    const actions = el('div', { style: 'display:flex;gap:8px;margin-top:14px;' }, [
      el('button', { class: 'btn btn-primary', onclick: () => { state.reviewNotesDirty = false; toast('Review saved — simulated.', 'success'); } }, 'Save review'),
      el('button', { class: 'btn btn-ghost', onclick: () => openWatchlistDrawer(img) }, 'Add to watchlist'),
    ]);
    panel.appendChild(actions);
    panel.appendChild(el('div', { class: 'simulated-note' }, 'Preview save-review behaviour — not connected to a backend.'));
    right.appendChild(panel);
    twoCol.appendChild(right);

    area.appendChild(twoCol);
  }

  function stepReview(dir) {
    if (state.reviewNotesDirty) {
      const ok = confirm('You have unsaved review notes on this match. Continue without saving?');
      if (!ok) return;
      state.reviewNotesDirty = false;
    }
    const next = state.reviewIndex + dir;
    if (next < 0 || next >= state.reviewList.length) return;
    state.reviewIndex = next;
    renderReview();
  }

  document.addEventListener('keydown', (e) => {
    if (state.page !== 'review') return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea') return;
    if (e.key === 'ArrowLeft') stepReview(-1);
    if (e.key === 'ArrowRight') stepReview(1);
  });

  // ================================================================== DATASETS
  function renderDatasets() {
    const area = $('#datasetsTableArea');
    area.innerHTML = '';
    if (state.activeDataset) { renderDatasetDetail(area, state.activeDataset); return; }

    area.appendChild(el('div', { class: 'page-title' }, 'Dataset overview'));
    area.appendChild(el('div', { class: 'page-subtitle' }, 'Create datasets, configure sources, and monitor indexing health.'));

    const wrap = el('div', { class: 'table-wrap' });
    const table = el('table');
    table.appendChild(el('thead', {}, el('tr', {}, ['Dataset', 'Sources', 'Indexed images', 'Health', 'Last scan', 'Actions'].map(h => el('th', {}, h)))));
    const tbody = el('tbody');
    DB.datasets.forEach(ds => {
      const tr = el('tr', { class: 'clickable' });
      tr.appendChild(el('td', {}, [document.createTextNode(ds.name), el('div', { class: 'cell-sub' }, ds.description)]));
      tr.appendChild(el('td', {}, String(ds.sources.length)));
      tr.appendChild(el('td', {}, ds.images.toLocaleString()));
      tr.appendChild(el('td', {}, healthBadge(ds.health)));
      tr.appendChild(el('td', {}, ds.lastScan));
      const actions = el('td');
      actions.appendChild(el('button', { class: 'btn btn-sm', onclick: (e) => { e.stopPropagation(); state.activeDataset = ds.id; renderDatasets(); } }, 'View'));
      tr.appendChild(actions);
      tr.addEventListener('click', () => { state.activeDataset = ds.id; renderDatasets(); });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    area.appendChild(wrap);
  }

  function healthBadge(status) {
    const map = { Healthy: 'ok', Warning: 'warn', Offline: 'err', 'Index pending': 'info', Importing: 'info', Failed: 'err' };
    return el('span', { class: `badge ${map[status] || 'neutral'}` }, status);
  }

  function renderDatasetDetail(area, id) {
    const ds = DB.getDataset(id);
    area.appendChild(el('button', { class: 'btn btn-sm btn-ghost', onclick: () => { state.activeDataset = null; renderDatasets(); } }, '‹ All datasets'));
    area.appendChild(el('div', { class: 'page-title', style: 'margin-top:10px;' }, ds.name));
    area.appendChild(el('div', { class: 'page-subtitle' }, ds.description));

    const stats = el('div', { class: 'stat-row' }, [
      el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value' }, ds.images.toLocaleString()), el('div', { class: 'stat-card__label' }, 'Indexed images')]),
      el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value' }, String(ds.sources.length)), el('div', { class: 'stat-card__label' }, 'Sources')]),
      el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value', style: 'font-size:16px;' }, ds.health), el('div', { class: 'stat-card__label' }, 'Health')]),
    ]);
    area.appendChild(stats);

    area.appendChild(el('div', { class: 'section-title' }, 'Sources'));
    const wrap = el('div', { class: 'table-wrap' });
    const table = el('table');
    table.appendChild(el('thead', {}, el('tr', {}, ['Source', 'Type', 'Path', 'Images', 'Status'].map(h => el('th', {}, h)))));
    const tbody = el('tbody');
    ds.sources.forEach(s => {
      const tr = el('tr');
      tr.appendChild(el('td', {}, s.name));
      tr.appendChild(el('td', {}, s.type));
      tr.appendChild(el('td', {}, [document.createTextNode(s.path), s.note ? el('div', { class: 'cell-sub' }, s.note) : null]));
      tr.appendChild(el('td', {}, s.images.toLocaleString()));
      tr.appendChild(el('td', {}, healthBadge(s.status)));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    area.appendChild(wrap);
    area.appendChild(el('div', { class: 'simulated-note' }, 'A path like D:\\Images refers to the server\'s filesystem, not your browser. Source directories must be configured by an authorised administrator.'));
  }

  function wireDatasetsPage() {
    $('#newDatasetBtn').addEventListener('click', () => { wizardState.step = 1; wizardState.data = { name: '', description: '', owner: 'u1', sources: [] }; goToPage('wizard'); renderWizard(); });
  }

  // ---------------- Create dataset wizard ----------------
  const wizardState = { step: 1, data: { name: '', description: '', owner: 'u1', sources: [] } };

  function renderWizard() {
    const area = $('#wizardArea');
    area.innerHTML = '';
    area.appendChild(el('div', { class: 'page-title' }, 'Create dataset'));
    area.appendChild(el('div', { class: 'page-subtitle' }, 'Step-by-step configuration for a new dataset and its sources.'));

    const stepsBar = el('div', { class: 'wizard-steps' });
    const labels = ['Dataset details', 'Add sources', 'Metadata config', 'Validate sources', 'Review & save'];
    labels.forEach((label, i) => {
      const n = i + 1;
      const cls = n === wizardState.step ? 'active' : n < wizardState.step ? 'done' : '';
      stepsBar.appendChild(el('div', { class: `wizard-step ${cls}` }, [el('span', { class: 'wizard-step__dot' }, n < wizardState.step ? '✓' : String(n)), el('span', {}, label)]));
      if (n < labels.length) stepsBar.appendChild(el('div', { class: 'wizard-step__sep' }));
    });
    area.appendChild(stepsBar);

    const body = el('div', { class: 'wizard-body' });
    if (wizardState.step === 1) {
      body.appendChild(fieldInput('Dataset name', 'wzName', wizardState.data.name, v => wizardState.data.name = v));
      body.appendChild(fieldTextarea('Description', 'wzDesc', wizardState.data.description, v => wizardState.data.description = v));
      body.appendChild(el('div', { class: 'field' }, [
        el('label', { class: 'field__label' }, 'Owner'),
        selectEl(['A. Farooqi', 'N. Siddiqui'], ['u1', 'u4'], wizardState.data.owner, v => wizardState.data.owner = v),
      ]));
      body.appendChild(el('div', { class: 'field' }, [
        el('label', { class: 'field__label' }, 'Access permissions'),
        el('label', { class: 'checkbox-row' }, [el('input', { type: 'checkbox', checked: 'checked' }), 'Power Users']),
        el('label', { class: 'checkbox-row' }, [el('input', { type: 'checkbox', checked: 'checked' }), 'Admins']),
        el('label', { class: 'checkbox-row' }, [el('input', { type: 'checkbox' }), 'Normal Users']),
      ]));
    } else if (wizardState.step === 2) {
      body.appendChild(el('div', { class: 'text-small text-secondary', style: 'margin-bottom:12px;' }, 'Add one or more sources. Paths refer to the server\'s filesystem, configured by an administrator.'));
      const list = el('div', { id: 'wzSourceList' });
      renderWizardSources(list);
      body.appendChild(list);
      body.appendChild(el('button', { class: 'btn btn-sm', style: 'margin-top:8px;', onclick: () => { wizardState.data.sources.push({ type: 'Local directory', path: '' }); renderWizardSources(list); } }, '+ Add source'));
    } else if (wizardState.step === 3) {
      body.appendChild(el('div', { class: 'field' }, [
        el('label', { class: 'field__label' }, 'Attach a metadata import'),
        el('label', { class: 'checkbox-row' }, [el('input', { type: 'checkbox' }), 'Associate a spreadsheet import after creation']),
      ]));
      body.appendChild(el('div', { class: 'field' }, [
        el('label', { class: 'field__label' }, 'Saved mapping template'),
        selectEl(['None', 'Missing Persons intake (standard)', 'Legacy report import'], ['none', 'tpl1', 'tpl2'], 'none', () => {}),
      ]));
      body.appendChild(el('div', { class: 'text-small text-secondary' }, 'This can also be configured later from Import Centre.'));
    } else if (wizardState.step === 4) {
      body.appendChild(el('div', { class: 'text-small text-secondary', style: 'margin-bottom:12px;' }, 'Simulated validation of connectivity, permissions, formats and duplicates.'));
      const results = wizardState.data.sources.length ? wizardState.data.sources : [{ type: 'Local directory', path: '(none configured)' }];
      results.forEach(s => {
        const row = el('div', { class: 'panel', style: 'margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;' });
        row.appendChild(el('div', {}, [document.createTextNode(s.type), el('div', { class: 'cell-sub' }, s.path || '—')]));
        row.appendChild(s.path ? healthBadge('Healthy') : healthBadge('Warning'));
        body.appendChild(row);
      });
    } else if (wizardState.step === 5) {
      body.appendChild(el('div', { class: 'panel' }, [
        el('div', { class: 'kv' }, [el('dt', {}, 'Name'), el('dd', {}, wizardState.data.name || '(untitled)')]),
        el('div', { class: 'kv' }, [el('dt', {}, 'Description'), el('dd', {}, wizardState.data.description || '—')]),
        el('div', { class: 'kv' }, [el('dt', {}, 'Sources'), el('dd', {}, String(wizardState.data.sources.length))]),
      ]));
    }
    area.appendChild(body);

    const footer = el('div', { class: 'overlay__footer', style: 'justify-content:space-between;max-width:640px;' });
    footer.appendChild(el('button', { class: 'btn btn-ghost', onclick: () => goToPage('datasets') }, 'Cancel'));
    const rightBtns = el('div', { style: 'display:flex;gap:8px;' });
    if (wizardState.step > 1) rightBtns.appendChild(el('button', { class: 'btn', onclick: () => { wizardState.step--; renderWizard(); } }, '‹ Back'));
    if (wizardState.step < 5) rightBtns.appendChild(el('button', { class: 'btn btn-primary', onclick: () => { wizardState.step++; renderWizard(); } }, 'Next ›'));
    else rightBtns.appendChild(el('button', { class: 'btn btn-primary', onclick: () => { toast(`Dataset "${wizardState.data.name || 'Untitled'}" created — simulated.`, 'success'); goToPage('datasets'); } }, 'Create dataset'));
    footer.appendChild(rightBtns);
    area.appendChild(footer);
  }

  function renderWizardSources(list) {
    list.innerHTML = '';
    wizardState.data.sources.forEach((s, i) => {
      const row = el('div', { class: 'panel', style: 'margin-bottom:8px;' });
      row.appendChild(el('div', { class: 'field' }, [el('label', { class: 'field__label' }, 'Type'), selectEl(['Local directory', 'Network share'], ['Local directory', 'Network share'], s.type, v => s.type = v)]));
      const pathField = el('div', { class: 'field', style: 'margin-bottom:0;' });
      pathField.appendChild(el('label', { class: 'field__label' }, 'Server-side path'));
      const pathInput = el('input', { class: 'input', placeholder: '\\\\fileserver\\share\\path or D:\\Images\\folder' });
      pathInput.value = s.path; pathInput.addEventListener('input', () => s.path = pathInput.value);
      pathField.appendChild(pathInput);
      row.appendChild(pathField);
      row.appendChild(el('button', { class: 'btn btn-sm btn-ghost', style: 'margin-top:8px;', onclick: () => { wizardState.data.sources.splice(i, 1); renderWizardSources(list); } }, 'Remove source'));
      list.appendChild(row);
    });
    if (!wizardState.data.sources.length) list.appendChild(el('div', { class: 'text-small text-secondary' }, 'No sources added yet.'));
  }

  function fieldInput(label, id, value, onChange) {
    const wrap = el('div', { class: 'field' });
    wrap.appendChild(el('label', { class: 'field__label', for: id }, label));
    const input = el('input', { class: 'input', id });
    input.value = value || '';
    input.addEventListener('input', () => onChange(input.value));
    wrap.appendChild(input);
    return wrap;
  }
  function fieldTextarea(label, id, value, onChange) {
    const wrap = el('div', { class: 'field' });
    wrap.appendChild(el('label', { class: 'field__label', for: id }, label));
    const ta = el('textarea', { class: 'textarea', id });
    ta.value = value || '';
    ta.addEventListener('input', () => onChange(ta.value));
    wrap.appendChild(ta);
    return wrap;
  }
  function selectEl(labels, values, current, onChange) {
    const sel = el('select', { class: 'select' });
    labels.forEach((l, i) => {
      const opt = el('option', { value: values[i] }, l);
      if (values[i] === current) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', () => onChange(sel.value));
    return sel;
  }

  // ================================================================== WATCHLISTS
  function renderWatchlists() {
    if (!state.activeWatchlist && DB.watchlists.length) state.activeWatchlist = DB.watchlists[0].id;
    const nav = $('#watchlistNav');
    nav.innerHTML = '';
    nav.appendChild(el('div', { class: 'section-title' }, 'Lists'));
    DB.watchlists.forEach(wl => {
      const btn = el('button', { class: 'nav-item' + (wl.id === state.activeWatchlist ? ' active' : ''), style: 'margin-bottom:4px;' },
        [el('span', { class: 'nav-item__icon' }, '☆'), el('span', { class: 'nav-item__label' }, `${wl.name} (${wl.entries.length})`)]);
      btn.addEventListener('click', () => { state.activeWatchlist = wl.id; renderWatchlists(); });
      nav.appendChild(btn);
    });

    const detail = $('#watchlistDetail');
    detail.innerHTML = '';
    const wl = DB.watchlists.find(w => w.id === state.activeWatchlist);
    if (!wl) { detail.appendChild(stateBlock('☆', 'No watchlists yet', 'Create a watchlist to start tracking subjects of interest.')); return; }

    detail.appendChild(el('div', { class: 'page-title' }, wl.name));
    detail.appendChild(el('div', { class: 'page-subtitle' }, wl.description));

    const meta = el('div', { class: 'panel' }, [
      el('div', { class: 'kv' }, [el('dt', {}, 'Case officer'), el('dd', {}, wl.caseOfficer)]),
      el('div', { class: 'kv' }, [el('dt', {}, 'Department'), el('dd', {}, wl.department)]),
      el('div', { class: 'kv' }, [el('dt', {}, 'Retention / review expiry'), el('dd', {}, wl.retention)]),
    ]);
    detail.appendChild(meta);

    detail.appendChild(el('div', { class: 'section-title', style: 'margin-top:18px;' }, `Entries (${wl.entries.length})`));
    const wrap = el('div', { class: 'table-wrap' });
    const table = el('table');
    table.appendChild(el('thead', {}, el('tr', {}, ['Image', 'Nickname', 'Case reference', 'Notes', 'Added by', 'Added', 'Actions'].map(h => el('th', {}, h)))));
    const tbody = el('tbody');
    wl.entries.forEach(entry => {
      const tr = el('tr');
      tr.appendChild(el('td', {}, el('img', { class: 'thumb', src: placeholderPortrait(entry.imageId) })));
      tr.appendChild(el('td', {}, entry.nickname));
      tr.appendChild(el('td', {}, entry.caseRef));
      tr.appendChild(el('td', {}, entry.notes || '—'));
      tr.appendChild(el('td', {}, (DB.users.find(u => u.id === entry.addedBy) || {}).name || entry.addedBy));
      tr.appendChild(el('td', {}, entry.added));
      const act = el('td');
      act.appendChild(el('button', { class: 'btn btn-sm btn-ghost', onclick: () => { wl.entries = wl.entries.filter(e => e.id !== entry.id); renderWatchlists(); toast('Entry removed from watchlist.'); } }, 'Remove'));
      tr.appendChild(act);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    detail.appendChild(wrap);
    if (!wl.entries.length) detail.appendChild(stateBlock('∅', 'No entries yet', 'Add candidates from Search or Match Review using "Add to watchlist".'));
  }

  function wireWatchlistsPage() {
    $('#newWatchlistBtn').addEventListener('click', () => {
      openModal('New watchlist', (body, close) => {
        body.appendChild(fieldInput('Watchlist name', 'nwName', '', () => {}));
        body.appendChild(fieldTextarea('Description', 'nwDesc', '', () => {}));
        body.appendChild(fieldInput('Case officer', 'nwOfficer', '', () => {}));
        body.appendChild(fieldInput('Department', 'nwDept', '', () => {}));
        return () => {
          const name = $('#nwName').value.trim() || 'Untitled watchlist';
          const wl = DB.createWatchlist({ name, description: $('#nwDesc').value, caseOfficer: $('#nwOfficer').value, department: $('#nwDept').value, retention: '—' });
          state.activeWatchlist = wl.id;
          renderWatchlists();
          toast(`Watchlist "${name}" created.`, 'success');
        };
      }, 'Create watchlist');
    });
  }

  function openWatchlistDrawer(img) {
    const backdrop = $('#drawerBackdrop');
    backdrop.innerHTML = '';
    backdrop.classList.add('open');
    const drawer = el('div', { class: 'drawer' });
    drawer.appendChild(el('div', { class: 'overlay__header' }, [el('h3', {}, 'Add to watchlist'), el('button', { class: 'overlay__close', onclick: () => backdrop.classList.remove('open') }, '×')]));
    drawer.appendChild(el('div', { class: 'text-small text-secondary', style: 'margin-bottom:14px;' }, `Adding ${img.id} with its case context.`));

    const selWrap = el('div', { class: 'field' });
    selWrap.appendChild(el('label', { class: 'field__label' }, 'Target watchlist'));
    const select = selectEl(DB.watchlists.map(w => w.name), DB.watchlists.map(w => w.id), DB.watchlists[0]?.id, () => {});
    selWrap.appendChild(select);
    drawer.appendChild(selWrap);

    drawer.appendChild(fieldInput('Nickname', 'wlNickname', img.id, () => {}));
    drawer.appendChild(fieldInput('Case reference', 'wlCaseRef', img.occurrences[0].caseRef, () => {}));
    drawer.appendChild(fieldTextarea('Entry notes', 'wlNotes', '', () => {}));

    const footer = el('div', { class: 'overlay__footer' });
    footer.appendChild(el('button', { class: 'btn btn-ghost', onclick: () => backdrop.classList.remove('open') }, 'Cancel'));
    footer.appendChild(el('button', { class: 'btn btn-primary', onclick: () => {
      const wl = DB.watchlists.find(w => w.id === select.value);
      DB.addWatchlistEntry(wl.id, { imageId: img.id, nickname: $('#wlNickname').value, caseRef: $('#wlCaseRef').value, notes: $('#wlNotes').value, addedBy: 'u1' });
      backdrop.classList.remove('open');
      toast(`Added to "${wl.name}".`, 'success');
      if (state.page === 'watchlists') renderWatchlists();
    } }, 'Add entry'));
    drawer.appendChild(footer);
    backdrop.appendChild(drawer);
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) backdrop.classList.remove('open'); }, { once: true });
  }

  // ================================================================== HISTORY
  function renderHistory() {
    const area = $('#historyArea');
    area.innerHTML = '';
    area.appendChild(el('div', { class: 'page-title' }, 'Search History'));
    area.appendChild(el('div', { class: 'page-subtitle' }, 'Audit trail of previous searches across the system.'));

    const wrap = el('div', { class: 'table-wrap' });
    const table = el('table');
    table.appendChild(el('thead', {}, el('tr', {}, ['Search ID', 'User', 'Date / time', 'Datasets', 'Query', 'Results', 'Status', 'Actions'].map(h => el('th', {}, h)))));
    const tbody = el('tbody');
    DB.searchHistory.forEach(h => {
      const tr = el('tr');
      tr.appendChild(el('td', {}, h.id));
      tr.appendChild(el('td', {}, h.user));
      tr.appendChild(el('td', {}, h.time));
      tr.appendChild(el('td', {}, h.datasets.join(', ')));
      tr.appendChild(el('td', {}, h.query));
      tr.appendChild(el('td', {}, String(h.results)));
      tr.appendChild(el('td', {}, el('span', { class: `badge ${h.status === 'Completed' ? 'ok' : 'err'}` }, h.status)));
      const act = el('td');
      act.appendChild(el('button', { class: 'btn btn-sm btn-ghost', onclick: () => toast('Opening saved result set — simulated.') }, 'View results'));
      tr.appendChild(act);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    area.appendChild(wrap);
  }

  // ================================================================== IMPORT CENTRE
  const importState = { step: 1, mapping: { 'Report ID': 'Report ID', 'Image Name': 'Image reference', 'Child Name': '', 'Report Date': '' } };

  function renderImport() {
    const area = $('#importArea');
    area.innerHTML = '';
    area.appendChild(el('div', { class: 'page-title' }, 'Metadata Importer'));
    area.appendChild(el('div', { class: 'page-subtitle' }, `Step ${importState.step} of 5 · map arbitrary spreadsheet columns to canonical fields`));

    const stepsBar = el('div', { class: 'wizard-steps' });
    ['Source', 'Sheets', 'Map', 'Validate', 'Confirm'].forEach((label, i) => {
      const n = i + 1;
      const cls = n === importState.step ? 'active' : n < importState.step ? 'done' : '';
      stepsBar.appendChild(el('div', { class: `wizard-step ${cls}` }, [el('span', { class: 'wizard-step__dot' }, n < importState.step ? '✓' : String(n)), el('span', {}, label)]));
      if (n < 5) stepsBar.appendChild(el('div', { class: 'wizard-step__sep' }));
    });
    area.appendChild(stepsBar);

    const body = el('div', { class: 'wizard-body', style: 'max-width:760px;' });
    if (importState.step === 1) {
      body.appendChild(el('div', { class: 'dropzone' }, [
        el('div', { class: 'dropzone__icon' }, '⇧'),
        el('div', {}, 'Drop an Excel workbook or CSV file here'),
        el('div', { class: 'text-small text-secondary' }, 'sample_workbook.xlsx — 2 sheets detected (simulated)'),
        el('button', { class: 'btn btn-sm', style: 'margin-top:10px;' }, 'Browse files'),
      ]));
    } else if (importState.step === 2) {
      body.appendChild(el('div', { class: 'field' }, [el('label', { class: 'field__label' }, 'Selected sheet'), selectEl(['Sheet 1', 'Sheet 2'], ['Sheet 1', 'Sheet 2'], 'Sheet 1', () => {})]));
      body.appendChild(el('div', { class: 'field' }, [el('label', { class: 'field__label' }, 'Header row'), selectEl(['Row 1', 'Row 2'], ['1', '2'], '1', () => {})]));
      body.appendChild(el('div', { class: 'text-small text-secondary' }, 'Detected headers: ID, Report ID, Image Name, Child Name, Report Date'));
    } else if (importState.step === 3) {
      body.appendChild(el('div', { class: 'section-title' }, 'Map columns'));
      const canonicalOptions = ['', 'Report ID', 'Image reference', 'Subject name', 'Report date', 'Case reference'];
      Object.keys(importState.mapping).forEach(src => {
        const row = el('div', { class: 'panel', style: 'display:flex;align-items:center;gap:12px;margin-bottom:8px;' });
        row.appendChild(el('div', { style: 'width:160px;' }, [el('div', { class: 'text-small text-secondary' }, 'Source column'), el('div', {}, src)]));
        row.appendChild(el('div', { class: 'text-secondary' }, '→'));
        const sel = selectEl(canonicalOptions.map(o => o || 'Select…'), canonicalOptions, importState.mapping[src], v => importState.mapping[src] = v);
        row.appendChild(sel);
        body.appendChild(row);
      });
      body.appendChild(el('div', { class: 'section-title', style: 'margin-top:16px;' }, 'Join configuration'));
      body.appendChild(el('div', { class: 'panel text-small' }, 'Sheet 1 · Report ID  ⟷  Sheet 2 · ID — match records across sheets using these key columns.'));
    } else if (importState.step === 4) {
      body.appendChild(el('div', { class: 'stat-row' }, [
        el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value' }, '184'), el('div', { class: 'stat-card__label' }, 'Matched records')]),
        el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value' }, '6'), el('div', { class: 'stat-card__label' }, 'Unmatched rows')]),
        el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value' }, '2'), el('div', { class: 'stat-card__label' }, 'Duplicate keys')]),
      ]));
      body.appendChild(el('div', { class: 'panel' }, [
        el('div', { class: 'kv' }, [el('dt', {}, 'Missing image references'), el('dd', {}, '3 rows')]),
        el('div', { class: 'kv' }, [el('dt', {}, 'Conflicting metadata'), el('dd', {}, '1 row — "Report Date" differs between sheets')]),
      ]));
      body.appendChild(el('div', { class: 'field', style: 'margin-top:14px;' }, [el('label', { class: 'field__label' }, 'On duplicate key'), selectEl(['Skip', 'Update existing', 'Insert new'], ['skip', 'update', 'insert'], 'update', () => {})]));
    } else if (importState.step === 5) {
      body.appendChild(el('div', { class: 'panel' }, [
        el('div', { class: 'kv' }, [el('dt', {}, 'Rows to import'), el('dd', {}, '184')]),
        el('div', { class: 'kv' }, [el('dt', {}, 'Mapping template'), el('dd', {}, 'Unsaved — new mapping')]),
      ]));
      body.appendChild(fieldInput('Save mapping as template', 'impTemplateName', '', () => {}));
    }
    area.appendChild(body);

    const footer = el('div', { class: 'overlay__footer', style: 'justify-content:space-between;max-width:760px;' });
    footer.appendChild(el('button', { class: 'btn btn-ghost', onclick: () => { importState.step = 1; renderImport(); } }, 'Cancel'));
    const rightBtns = el('div', { style: 'display:flex;gap:8px;' });
    if (importState.step > 1) rightBtns.appendChild(el('button', { class: 'btn', onclick: () => { importState.step--; renderImport(); } }, '‹ Back'));
    if (importState.step < 5) rightBtns.appendChild(el('button', { class: 'btn btn-primary', onclick: () => { importState.step++; renderImport(); } }, 'Next ›'));
    else rightBtns.appendChild(el('button', { class: 'btn btn-primary', onclick: () => { toast('Import job started — simulated. 184 records queued.', 'success'); importState.step = 1; renderImport(); } }, 'Start import'));
    footer.appendChild(rightBtns);
    area.appendChild(footer);
  }

  // ================================================================== INDEX MANAGEMENT
  function renderIndexMgmt() {
    const area = $('#indexArea');
    area.innerHTML = '';
    area.appendChild(el('div', { class: 'page-title' }, 'Index Management'));
    area.appendChild(el('div', { class: 'page-subtitle' }, 'Model, detector and per-dataset index health. Admin only.'));

    const cfg = el('div', { class: 'panel' }, [
      el('div', { class: 'section-title' }, 'Active configuration'),
      el('div', { class: 'kv' }, [el('dt', {}, 'Model'), el('dd', {}, DB.indexState.model)]),
      el('div', { class: 'kv' }, [el('dt', {}, 'Detector'), el('dd', {}, DB.indexState.detector)]),
      el('div', { class: 'kv' }, [el('dt', {}, 'Device'), el('dd', {}, DB.indexState.device)]),
      el('div', { class: 'kv' }, [el('dt', {}, 'Worker count'), el('dd', {}, String(DB.indexState.workers))]),
      el('div', { class: 'kv' }, [el('dt', {}, 'Clustering'), el('dd', {}, DB.indexState.clustering)]),
    ]);
    area.appendChild(cfg);

    area.appendChild(el('div', { class: 'section-title', style: 'margin-top:18px;' }, 'Per-dataset index status'));
    const wrap = el('div', { class: 'table-wrap' });
    const table = el('table');
    table.appendChild(el('thead', {}, el('tr', {}, ['Dataset', 'Images', 'Indexed faces', 'Last indexed', 'Status', 'Actions'].map(h => el('th', {}, h)))));
    const tbody = el('tbody');
    DB.indexState.perDataset.forEach(d => {
      const tr = el('tr');
      tr.appendChild(el('td', {}, d.dataset));
      tr.appendChild(el('td', {}, d.images.toLocaleString()));
      tr.appendChild(el('td', {}, d.faces.toLocaleString()));
      tr.appendChild(el('td', {}, d.lastIndexed));
      tr.appendChild(el('td', {}, healthBadge(d.status)));
      const act = el('td');
      act.appendChild(el('button', { class: 'btn btn-sm', onclick: () => toast(`Incremental index started for ${d.dataset} — simulated.`) }, 'Index now'));
      act.appendChild(el('button', { class: 'btn btn-sm btn-danger', style: 'margin-left:6px;', onclick: () => confirmRebuild(d.dataset) }, 'Force rebuild'));
      tr.appendChild(act);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    area.appendChild(wrap);

    area.appendChild(el('div', { class: 'section-title', style: 'margin-top:18px;' }, 'Job history'));
    const wrap2 = el('div', { class: 'table-wrap' });
    const table2 = el('table');
    table2.appendChild(el('thead', {}, el('tr', {}, ['Job', 'Type', 'Dataset', 'Started', 'Status', 'Duration'].map(h => el('th', {}, h)))));
    const tbody2 = el('tbody');
    DB.indexState.jobs.forEach(j => {
      const tr = el('tr');
      tr.appendChild(el('td', {}, j.id));
      tr.appendChild(el('td', {}, j.type));
      tr.appendChild(el('td', {}, j.dataset));
      tr.appendChild(el('td', {}, j.started));
      tr.appendChild(el('td', {}, [healthBadge(j.status === 'Completed' ? 'Healthy' : j.status === 'Failed' ? 'Offline' : 'Index pending'), j.error ? el('div', { class: 'cell-sub' }, j.error) : null]));
      tr.appendChild(el('td', {}, j.duration));
      tbody2.appendChild(tr);
    });
    table2.appendChild(tbody2);
    wrap2.appendChild(table2);
    area.appendChild(wrap2);
  }

  function confirmRebuild(datasetName) {
    openModal('Confirm force rebuild', (body) => {
      body.appendChild(el('div', { class: 'text-small' }, `This will rebuild the entire index for "${datasetName}" from source. This can take a long time and will temporarily reduce search coverage for this dataset. This action cannot be undone.`));
      return () => toast(`Force rebuild started for ${datasetName} — simulated.`, 'success');
    }, 'Force rebuild', true);
  }

  // ================================================================== ADMIN
  function renderAdmin() {
    const area = $('#adminArea');
    area.innerHTML = '';
    area.appendChild(el('div', { class: 'page-title' }, 'Administration & System Health'));
    area.appendChild(el('div', { class: 'page-subtitle' }, 'Users, roles, permissions and system status.'));

    const stats = el('div', { class: 'stat-row' }, [
      statusStat('API', DB.systemHealth.api), statusStat('Database', DB.systemHealth.database),
      el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value', style: 'font-size:16px;' }, DB.systemHealth.storage), el('div', { class: 'stat-card__label' }, 'Storage')]),
      el('div', { class: 'stat-card' }, [el('div', { class: 'stat-card__value', style: 'font-size:16px;' }, DB.systemHealth.workers), el('div', { class: 'stat-card__label' }, 'Background workers')]),
    ]);
    area.appendChild(stats);

    area.appendChild(el('div', { class: 'section-title', style: 'margin-top:18px;' }, 'Users & roles'));
    const wrap = el('div', { class: 'table-wrap' });
    const table = el('table');
    table.appendChild(el('thead', {}, el('tr', {}, ['User', 'Role', 'Actions'].map(h => el('th', {}, h)))));
    const tbody = el('tbody');
    DB.users.forEach(u => {
      const tr = el('tr');
      tr.appendChild(el('td', {}, u.name));
      tr.appendChild(el('td', {}, selectEl(['Normal User', 'Power User', 'Admin'], ['Normal User', 'Power User', 'Admin'], u.role, v => { u.role = v; toast(`${u.name} role set to ${v} — simulated.`, 'success'); })));
      tr.appendChild(el('td', {}, el('button', { class: 'btn btn-sm btn-ghost', onclick: () => toast('Dataset access editor — simulated.') }, 'Edit dataset access')));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    area.appendChild(wrap);

    area.appendChild(el('div', { class: 'section-title', style: 'margin-top:18px;' }, 'Audit log'));
    const wrap2 = el('div', { class: 'table-wrap' });
    const table2 = el('table');
    table2.appendChild(el('thead', {}, el('tr', {}, ['Time', 'User', 'Action', 'Detail'].map(h => el('th', {}, h)))));
    const tbody2 = el('tbody');
    DB.systemHealth.auditLog.forEach(a => {
      const tr = el('tr');
      tr.appendChild(el('td', {}, a.time)); tr.appendChild(el('td', {}, a.user)); tr.appendChild(el('td', {}, a.action)); tr.appendChild(el('td', {}, a.detail));
      tbody2.appendChild(tr);
    });
    table2.appendChild(tbody2);
    wrap2.appendChild(table2);
    area.appendChild(wrap2);
    area.appendChild(el('div', { class: 'simulated-note' }, 'Role switching here is a visual demo only. The eventual Flask backend must enforce all permissions server-side.'));
  }

  function statusStat(label, value) {
    return el('div', { class: 'stat-card' }, [
      el('div', { class: 'stat-card__value', style: 'font-size:16px;display:flex;align-items:center;gap:8px;' }, [el('span', { class: 'dot' }), value]),
      el('div', { class: 'stat-card__label' }, label),
    ]);
  }

  // ================================================================== MODALS
  function openModal(title, buildBody, confirmLabel = 'Save', danger = false) {
    const backdrop = $('#modalBackdrop');
    backdrop.innerHTML = '';
    backdrop.classList.add('open');
    const modal = el('div', { class: 'modal' });
    modal.appendChild(el('div', { class: 'overlay__header' }, [el('h3', {}, title), el('button', { class: 'overlay__close', onclick: () => backdrop.classList.remove('open') }, '×')]));
    const body = el('div');
    const onConfirm = buildBody(body, () => backdrop.classList.remove('open'));
    modal.appendChild(body);
    const footer = el('div', { class: 'overlay__footer' });
    footer.appendChild(el('button', { class: 'btn btn-ghost', onclick: () => backdrop.classList.remove('open') }, 'Cancel'));
    footer.appendChild(el('button', { class: `btn ${danger ? 'btn-danger' : 'btn-primary'}`, onclick: () => { if (onConfirm) onConfirm(); backdrop.classList.remove('open'); } }, confirmLabel));
    modal.appendChild(footer);
    backdrop.appendChild(modal);
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) backdrop.classList.remove('open'); }, { once: true });
  }

  // ================================================================== GLOBAL CHROME
  function wireChrome() {
    $$('.nav-item[data-page]').forEach(btn => btn.addEventListener('click', () => goToPage(btn.dataset.page)));
    $('#railToggle').addEventListener('click', () => $('#app').classList.toggle('rail-expanded'));
    $('#roleSelect').addEventListener('change', (e) => { state.role = e.target.value; applyRoleVisibility(); });
  }

  function init() {
    wireChrome();
    wireSearchPage();
    wireDatasetsPage();
    wireWatchlistsPage();
    applyRoleVisibility();
    goToPage('search');
  }

  return { init };
})();

document.addEventListener('DOMContentLoaded', App.init);
