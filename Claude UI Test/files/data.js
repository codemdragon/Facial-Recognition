/* =========================================================================
   CPLC Face Search — mock data + data-service layer
   All functions here simulate what will eventually be REST calls to the
   Flask / DeepFace / FAISS / PostgreSQL backend. Nothing here is a real
   recognition engine; scores and records are fictitious.
   ========================================================================= */

const DB = (() => {

  // ---- Users -------------------------------------------------------------
  const users = [
    { id: 'u1', name: 'M. Codem', role: 'Admin', initials: 'AF' },
    { id: 'u2', name: 'S. Khan', role: 'Power User', initials: 'SK' },
    { id: 'u3', name: 'M. Raza', role: 'Normal User', initials: 'MR' },
    { id: 'u4', name: 'N. Siddiqui', role: 'Power User', initials: 'NS' },
  ];

  // ---- Datasets & sources --------------------------------------------------
  const datasets = [
    {
      id: 'ds-missing', name: 'Missing Persons', owner: 'u1',
      description: 'Active missing-persons case photographs, CPLC intake.',
      images: 12840, health: 'Healthy', lastScan: '2026-10-27 06:00',
      sources: [
        { id: 'src-1', name: 'Intake Share', type: 'Network share', path: '\\\\\\\\fileserver\\\\missing\\\\intake', status: 'Healthy', images: 9210 },
        { id: 'src-2', name: 'Field Reports', type: 'Local directory', path: 'D:\\\\Images\\\\field_reports', status: 'Healthy', images: 2980 },
        { id: 'src-3', name: 'Archive Scans', type: 'Network share', path: '\\\\\\\\fileserver\\\\missing\\\\archive', status: 'Healthy', images: 650 },
      ],
    },
    {
      id: 'ds-case', name: 'Case Archive', owner: 'u1',
      description: 'Closed and historical case files with photo evidence.',
      images: 8230, health: 'Warning', lastScan: '2026-10-26 22:14',
      sources: [
        { id: 'src-4', name: 'Case Vault', type: 'Network share', path: '\\\\\\\\fileserver\\\\cases\\\\vault', status: 'Healthy', images: 7510 },
        { id: 'src-5', name: 'Legacy Imports', type: 'Local directory', path: 'D:\\\\Images\\\\legacy_cases', status: 'Warning', images: 720, note: 'Share unreachable since 21:40 — last known image count shown.' },
      ],
    },
    {
      id: 'ds-historical', name: 'Historical Reports', owner: 'u4',
      description: 'Digitised paper reports, 2008–2018, indexed by report ID.',
      images: 3102, health: 'Index pending', lastScan: '2026-10-20 09:02',
      sources: [
        { id: 'src-6', name: 'Scanned Reports', type: 'Network share', path: '\\\\\\\\fileserver\\\\historical\\\\scans', status: 'Index pending', images: 3102 },
      ],
    },
  ];

  // ---- Canonical images + occurrences ------------------------------------
  // Each canonical image can appear in multiple sources with distinct metadata.
  const canonicalImages = [
    {
      id: 'IMG-00142', similarity: 87.4,
      occurrences: [
        { dataset: 'Missing Persons', source: 'Intake Share', filename: 'IMG_104.jpg', path: '\\\\\\\\fileserver\\\\missing\\\\intake\\\\IMG_104.jpg', linkedName: 'Report subject — A. Bhatti', caseRef: 'CASE-104', recordId: 'REC-10422', imported: '2026-08-14' },
        { dataset: 'Case Archive', source: 'Case Vault', filename: 'subject_7.png', path: '\\\\\\\\fileserver\\\\cases\\\\vault\\\\subject_7.png', linkedName: 'Witness reference — F. Noor', caseRef: 'REPORT-882', recordId: 'REC-88209', imported: '2026-05-02' },
      ],
    },
    {
      id: 'IMG-00381', similarity: 81.2, occurrences: [
        { dataset: 'Case Archive', source: 'Legacy Imports', filename: 'archive_0381.jpg', path: 'D:\\\\Images\\\\legacy_cases\\\\archive_0381.jpg', linkedName: 'Subject — R. Memon', caseRef: 'REPORT-331', recordId: 'REC-33104', imported: '2024-11-30' },
      ]
    },
    {
      id: 'IMG-00512', similarity: 78.9, occurrences: [
        { dataset: 'Missing Persons', source: 'Field Reports', filename: 'fr_2026_0512.jpg', path: 'D:\\\\Images\\\\field_reports\\\\fr_2026_0512.jpg', linkedName: 'Report subject — H. Jatoi', caseRef: 'CASE-512', recordId: 'REC-51201', imported: '2026-09-01' },
        { dataset: 'Historical Reports', source: 'Scanned Reports', filename: 'scan_1998_0771.tif', path: '\\\\\\\\fileserver\\\\historical\\\\scans\\\\scan_1998_0771.tif', linkedName: 'Report subject — unnamed minor', caseRef: 'HIST-0771', recordId: 'REC-07711', imported: '2022-02-18' },
        { dataset: 'Case Archive', source: 'Case Vault', filename: 'vault_2210.jpg', path: '\\\\\\\\fileserver\\\\cases\\\\vault\\\\vault_2210.jpg', linkedName: 'Witness reference — S. Abbasi', caseRef: 'REPORT-440', recordId: 'REC-44003', imported: '2025-06-09' },
      ]
    },
    {
      id: 'IMG-00076', similarity: 76.5, occurrences: [
        { dataset: 'Missing Persons', source: 'Intake Share', filename: 'IMG_076.jpg', path: '\\\\\\\\fileserver\\\\missing\\\\intake\\\\IMG_076.jpg', linkedName: 'Report subject — K. Dawood', caseRef: 'CASE-076', recordId: 'REC-07601', imported: '2026-10-02' },
      ]
    },
    {
      id: 'IMG-00298', similarity: 74.1, occurrences: [
        { dataset: 'Case Archive', source: 'Case Vault', filename: 'vault_0298.jpg', path: '\\\\\\\\fileserver\\\\cases\\\\vault\\\\vault_0298.jpg', linkedName: 'Subject — T. Qureshi', caseRef: 'REPORT-205', recordId: 'REC-20503', imported: '2023-12-11' },
        { dataset: 'Missing Persons', source: 'Archive Scans', filename: 'arc_0298.jpg', path: '\\\\\\\\fileserver\\\\missing\\\\archive\\\\arc_0298.jpg', linkedName: 'Report subject — T. Qureshi', caseRef: 'CASE-298', recordId: 'REC-29804', imported: '2023-12-20' },
      ]
    },
    {
      id: 'IMG-00611', similarity: 72.8, occurrences: [
        { dataset: 'Historical Reports', source: 'Scanned Reports', filename: 'scan_2004_1190.tif', path: '\\\\\\\\fileserver\\\\historical\\\\scans\\\\scan_2004_1190.tif', linkedName: 'Report subject — unnamed', caseRef: 'HIST-1190', recordId: 'REC-11905', imported: '2021-07-03' },
      ]
    },
    {
      id: 'IMG-00933', similarity: 71.0, occurrences: [
        { dataset: 'Missing Persons', source: 'Field Reports', filename: 'fr_2026_0933.jpg', path: 'D:\\\\Images\\\\field_reports\\\\fr_2026_0933.jpg', linkedName: 'Report subject — Z. Lakhani', caseRef: 'CASE-933', recordId: 'REC-93302', imported: '2026-07-19' },
      ]
    },
  ];

  // generate a few more procedurally to reach a reasonable result-set size
  const extraNames = ['B. Chandio', 'I. Soomro', 'W. Panhwar', 'Y. Baloch', 'R. Sheikh', 'N. Rind', 'U. Memon', 'D. Khoso', 'J. Brohi', 'L. Rajput'];
  let n = 1001;
  extraNames.forEach((name, i) => {
    canonicalImages.push({
      id: `IMG-${n++}`,
      similarity: Math.max(70, 92 - i * 1.6 - Math.random() * 3).toFixed(1) * 1,
      occurrences: [{
        dataset: ['Missing Persons', 'Case Archive', 'Historical Reports'][i % 3],
        source: ['Intake Share', 'Case Vault', 'Scanned Reports'][i % 3],
        filename: `img_${1000 + i}.jpg`,
        path: `\\\\\\\\fileserver\\\\misc\\\\img_${1000 + i}.jpg`,
        linkedName: `Report subject — ${name}`,
        caseRef: `CASE-${1000 + i}`,
        recordId: `REC-${90000 + i}`,
        imported: '2026-0' + ((i % 9) + 1) + '-1' + (i % 9),
      }],
    });
  });

  canonicalImages.forEach(img => {
    img.reviewStatus = 'Unreviewed';
    img.notes = '';
  });

  // ---- Watchlists ----------------------------------------------------------
  const watchlists = [
    {
      id: 'wl-1', name: 'District Central — Active Cases', description: 'Active missing-persons cases under District Central jurisdiction.',
      caseOfficer: 'Insp. R. Abbas', department: 'CPLC District Central', retention: '2027-01-01',
      entries: [
        { id: 'e1', imageId: 'IMG-00142', nickname: 'Subject 104', caseRef: 'CASE-104', notes: 'Flagged for follow-up interview.', addedBy: 'u2', added: '2026-10-20' },
        { id: 'e2', imageId: 'IMG-00512', nickname: 'Subject 512', caseRef: 'CASE-512', notes: 'Family requested priority review.', addedBy: 'u3', added: '2026-10-22' },
      ],
    },
    {
      id: 'wl-2', name: 'Cross-reference — Historical Unresolved', description: 'Historical unresolved reports cross-referenced against recent intake.',
      caseOfficer: 'Insp. F. Qadri', department: 'CPLC Research Unit', retention: '2026-12-31',
      entries: [
        { id: 'e3', imageId: 'IMG-00611', nickname: 'Hist. 1190', caseRef: 'HIST-1190', notes: '', addedBy: 'u4', added: '2026-09-01' },
      ],
    },
  ];

  // ---- Search history --------------------------------------------------
  const searchHistory = [
    { id: 'sh-1001', user: 'S. Khan', time: '2026-10-28 09:14', datasets: ['Missing Persons'], query: 'upload_0142.jpg', results: 24, status: 'Completed' },
    { id: 'sh-1000', user: 'M. Codem', time: '2026-10-27 17:02', datasets: ['Missing Persons', 'Case Archive'], query: 'query_photo.png', results: 41, status: 'Completed' },
    { id: 'sh-0999', user: 'M. Raza', time: '2026-10-27 11:48', datasets: ['Case Archive'], query: 'subject_scan.jpg', results: 8, status: 'Completed' },
    { id: 'sh-0998', user: 'N. Siddiqui', time: '2026-10-26 15:30', datasets: ['Historical Reports'], query: 'hist_candidate.tif', results: 3, status: 'Completed' },
    { id: 'sh-0997', user: 'S. Khan', time: '2026-10-25 10:05', datasets: ['Missing Persons', 'Case Archive', 'Historical Reports'], query: 'case_440_ref.jpg', results: 57, status: 'Failed' },
  ];

  // ---- Index management --------------------------------------------------
  const indexState = {
    model: 'ArcFace (DeepFace)', detector: 'RetinaFace', device: 'GPU 0 — RTX A4000',
    workers: 4, clustering: 'Off',
    perDataset: datasets.map(d => ({ dataset: d.name, images: d.images, faces: Math.round(d.images * 1.02), lastIndexed: d.lastScan, status: d.health })),
    jobs: [
      { id: 'job-55', type: 'Incremental index', dataset: 'Missing Persons', started: '2026-10-27 06:00', status: 'Completed', duration: '14m 02s' },
      { id: 'job-54', type: 'Full rebuild', dataset: 'Case Archive', started: '2026-10-21 02:00', status: 'Completed', duration: '1h 12m' },
      { id: 'job-53', type: 'Incremental index', dataset: 'Historical Reports', started: '2026-10-20 09:02', status: 'Failed', duration: '—', error: 'Source share unreachable (src-6)' },
    ],
  };

  // ---- Admin / system health ---------------------------------------------
  const systemHealth = {
    api: 'Online', database: 'Online', storage: '62% used (1.24 TB / 2 TB)',
    workers: '4 / 4 active',
    auditLog: [
      { time: '2026-10-28 09:15', user: 'S. Khan', action: 'Ran search', detail: 'sh-1001' },
      { time: '2026-10-27 18:40', user: 'M. Codem', action: 'Edited user role', detail: 'N. Siddiqui → Power User' },
      { time: '2026-10-27 17:03', user: 'M. Codem', action: 'Ran search', detail: 'sh-1000' },
      { time: '2026-10-26 22:14', user: 'System', action: 'Source scan', detail: 'Case Archive — Legacy Imports flagged Warning' },
    ],
  };

  // ---- "API" layer (simulated latency) -----------------------------------
  const delay = (ms) => new Promise(res => setTimeout(res, ms));

  return {
    users, datasets, canonicalImages, watchlists, searchHistory, indexState, systemHealth,

    getDataset: (id) => datasets.find(d => d.id === id),

    async runSearch({ datasetIds, maxResults, minSimilarity }) {
      await delay(1100 + Math.random() * 600);
      let pool = canonicalImages.filter(img => img.similarity >= minSimilarity);
      // simulate dataset scoping: only keep images with at least one occurrence in a selected dataset
      if (datasetIds && datasetIds.length) {
        const names = datasetIds.map(id => datasets.find(d => d.id === id)?.name);
        pool = pool.filter(img => img.occurrences.some(o => names.includes(o.dataset)));
      }
      pool = pool.sort((a, b) => b.similarity - a.similarity);
      if (maxResults !== 'all') pool = pool.slice(0, Number(maxResults));
      return pool;
    },

    addWatchlistEntry(watchlistId, entry) {
      const wl = watchlists.find(w => w.id === watchlistId);
      if (!wl) return null;
      const e = { id: 'e' + Math.random().toString(36).slice(2, 8), ...entry, added: new Date().toISOString().slice(0, 10) };
      wl.entries.push(e);
      return e;
    },

    createWatchlist(payload) {
      const wl = { id: 'wl-' + Math.random().toString(36).slice(2, 7), entries: [], ...payload };
      watchlists.push(wl);
      return wl;
    },
  };
})();
