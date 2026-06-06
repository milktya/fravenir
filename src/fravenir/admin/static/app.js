/* fravenir admin UI — vanilla JS */

// ─── Theme constants ────────────────────────────────────────────────────────
const THEMES = {
  dark: {
    nodeText: '#e6edf3',
    mentions: '#8b949e',
    relation: '#c9d1d9',
  },
  light: {
    nodeText: '#1f2328',
    mentions: '#C0C0C0',
    relation: '#333',
  },
};

// ─── Util ───────────────────────────────────────────────────────────────────
// SEC-1 HIGH-5-1: LLM 出力 / DB 文字列を innerHTML 経路に流す前に必ず通す。
// バッチ③ MEDIUM-2 の表示側責務をフロント側で担保する。
function escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ─── State ──────────────────────────────────────────────────────────────────
const SPREAD_DEFAULT = 3.0;
const state = {
  scope: 'active',
  view: localStorage.getItem('admin_view') || 'panel',
  theme: localStorage.getItem('admin_theme') || 'dark',
  spread: parseFloat(localStorage.getItem('admin_spread')) || SPREAD_DEFAULT,
  filters: {
    kind: { facts: true, state: true, emo: true },
    type: { episode: true, entity: true, relation: true, mentions: true },
    importance: 1,
    degree: 0,
  },
  search: { query: '', hops: 1 },
  focus: { nodeId: null, hops: 2 },
  selected: null,
  graphData: null,
  stats: null,
  cy: null,
};

// ─── API ────────────────────────────────────────────────────────────────────
async function api(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${body ? ' — ' + body : ''}`);
  }
  return res.json();
}

// ─── Cytoscape ──────────────────────────────────────────────────────────────
function initCytoscape(elements) {
  const t = THEMES[state.theme];
  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: elements,
    minZoom: 0.1,
    maxZoom: 3,
    wheelSensitivity: 0.3,
    style: [
      {
        selector: 'node',
        style: {
          label: 'data(label)',
          color: t.nodeText,
          'font-size': '11px',
          'text-wrap': 'ellipsis',
          'text-max-width': '80px',
          width: 32,
          height: 32,
          'border-width': 1,
          'border-color': '#555',
          'background-color': '#BDBDBD',
        },
      },
      {
        selector: 'node[type="episode"]',
        style: { shape: 'ellipse' },
      },
      {
        selector: 'node[type="episode"][kind="facts"]',
        style: { 'background-color': '#4A90E2' },
      },
      {
        selector: 'node[type="episode"][kind="state"]',
        style: { 'background-color': '#7ED321' },
      },
      {
        selector: 'node[type="episode"][kind="emo"]',
        style: { 'background-color': '#D0021B' },
      },
      {
        selector: 'node[type="episode"][kind!="facts"][kind!="state"][kind!="emo"]',
        style: { 'background-color': '#9B9B9B' },
      },
      {
        selector: 'node[type="entity"]',
        style: { shape: 'rectangle' },
      },
      {
        selector: 'node[type="entity"][?is_self]',
        style: {
          shape: 'pentagon',
          'background-color': '#F5A623',
          width: 48,
          height: 48,
          'border-width': 2,
          'border-color': '#B8860B',
        },
      },
      {
        selector: 'node[type="entity"][entity_type="person"][!is_self]',
        style: { 'background-color': '#9013FE' },
      },
      {
        selector: 'node[type="entity"][entity_type="place"][!is_self]',
        style: { 'background-color': '#8B572A' },
      },
      {
        selector: 'node[type="entity"][entity_type="concept"][!is_self]',
        style: { 'background-color': '#50E3C2' },
      },
      {
        selector: 'node[type="entity"][entity_type=""][!is_self]',
        style: { 'background-color': '#BDBDBD' },
      },
      {
        selector: 'node[!is_active]',
        style: { opacity: 0.5, 'border-style': 'dashed' },
      },
      {
        selector: 'node[?is_suppressed]',
        style: { opacity: 0.3, 'border-style': 'dotted' },
      },
      {
        selector: 'edge',
        style: {
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'arrow-scale': 0.8,
        },
      },
      {
        selector: 'edge[type="mentions"]',
        style: {
          'line-color': t.mentions,
          'target-arrow-color': t.mentions,
          width: 1,
        },
      },
      {
        selector: 'edge[type="relation"]',
        style: {
          'line-color': t.relation,
          'target-arrow-color': t.relation,
          width: 'mapData(strength, 0, 1, 1, 4)',
        },
      },
      {
        selector: 'edge[!is_active]',
        style: { 'line-style': 'dashed', opacity: 0.4 },
      },
      {
        selector: 'node:selected',
        style: { 'border-width': 3, 'border-color': '#FF4081' },
      },
      {
        selector: '.hidden',
        style: { visibility: 'hidden' },
      },
      {
        selector: '.dimmed',
        style: { opacity: 0.12 },
      },
      {
        selector: 'node.match',
        style: {
          'border-width': 4,
          'border-color': '#FF4081',
          'border-opacity': 1,
        },
      },
    ],
  });

  cy.on('tap', 'node', (evt) => {
    const node = evt.target;
    state.focus.nodeId = node.id();
    applyFocus();
    showDetail(node.id(), node.data('type'));
  });

  cy.on('tap', 'edge', (evt) => {
    const edge = evt.target;
    showDetail(edge.id(), 'relation');
  });

  cy.on('tap', (evt) => {
    if (evt.target === cy) {
      hidePopover();
      state.selected = null;
      state.focus.nodeId = null;
      cy.$(':selected').unselect();
      applyFocus();
    }
  });

  return cy;
}

function runLayout() {
  if (!state.cy) return;
  const k = state.spread;
  state.cy.layout({
    name: 'fcose',
    quality: 'default',
    animate: false,
    randomize: true,
    idealEdgeLength: 80 * k,
    nodeSeparation: 75 * k,
    nodeRepulsion: 4500 * k,
    gravity: 0.25 / k,
  }).run();
}

let searchTimer = null;
function setSearchQuery(q) {
  state.search.query = q;
  state.focus.nodeId = null;
  const clearBtn = document.getElementById('search-clear');
  if (clearBtn) clearBtn.hidden = !q;
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => applyHighlight(), 150);
}

function applyHighlight() {
  if (!state.cy) return;
  const q = state.search.query.trim().toLowerCase();
  state.cy.batch(() => {
    state.cy.elements().removeClass('dimmed match');
    if (!q) return;
    const matched = state.cy.nodes().filter((n) => {
      const label = (n.data('label') || '').toLowerCase();
      return label.includes(q);
    });
    if (matched.length === 0) {
      state.cy.elements().addClass('dimmed');
      return;
    }
    matched.addClass('match');
    let neighborhood = matched;
    for (let i = 0; i < state.search.hops; i++) {
      neighborhood = neighborhood.union(neighborhood.openNeighborhood());
    }
    state.cy.elements().difference(neighborhood).addClass('dimmed');
  });
}

// クリックで選択したノードを起点に hops 近傍以外を dim する。
// focus が解除されたら検索ハイライト（あれば）を復元する。
function applyFocus() {
  if (!state.cy) return;
  const focusId = state.focus.nodeId;
  if (!focusId) {
    state.cy.batch(() => {
      state.cy.elements().removeClass('dimmed match');
    });
    applyHighlight();
    return;
  }
  state.cy.batch(() => {
    state.cy.elements().removeClass('dimmed match');
    const target = state.cy.getElementById(focusId);
    if (target.length === 0) return;
    let neighborhood = target;
    for (let i = 0; i < state.focus.hops; i++) {
      neighborhood = neighborhood.union(neighborhood.openNeighborhood());
    }
    state.cy.elements().difference(neighborhood).addClass('dimmed');
  });
}

let spreadTimer = null;
function setSpread(v) {
  state.spread = v;
  localStorage.setItem('admin_spread', String(v));
  const valueEl = document.getElementById('spread-value');
  if (valueEl) valueEl.textContent = v.toFixed(1);
  clearTimeout(spreadTimer);
  spreadTimer = setTimeout(() => runLayout(), 300);
}

function applyGraphTheme() {
  if (!state.cy) return;
  const t = THEMES[state.theme];
  state.cy.style()
    .selector('node')
      .style('color', t.nodeText)
    .selector('edge[type="mentions"]')
      .style('line-color', t.mentions)
      .style('target-arrow-color', t.mentions)
    .selector('edge[type="relation"]')
      .style('line-color', t.relation)
      .style('target-arrow-color', t.relation)
    .update();
}

// ─── Filters ────────────────────────────────────────────────────────────────
function applyFilters() {
  if (!state.cy) return;
  state.cy.batch(() => {
    state.cy.nodes().forEach((n) => {
      const t = n.data('type');
      let visible = true;
      if (t === 'episode') {
        const importance = n.data('importance') ?? 1;
        visible = state.filters.type.episode
          && !!state.filters.kind[n.data('kind')]
          && importance >= state.filters.importance;
      } else if (t === 'entity') {
        visible = state.filters.type.entity;
      }
      // degree threshold: 元グラフの接続数で判定 (hidden 状態は無視)
      if (visible && state.filters.degree > 0) {
        if (n.degree(false) < state.filters.degree) visible = false;
      }
      n.toggleClass('hidden', !visible);
    });
    state.cy.edges().forEach((e) => {
      const t = e.data('type');
      let visible = true;
      if (t === 'mentions') {
        visible = state.filters.type.mentions;
      } else if (t === 'relation') {
        visible = state.filters.type.relation;
      }
      if (visible && (e.source().hasClass('hidden') || e.target().hasClass('hidden'))) {
        visible = false;
      }
      e.toggleClass('hidden', !visible);
    });
  });
}

// ─── Stats Bar ──────────────────────────────────────────────────────────────
function renderStatsBar() {
  const s = state.stats;
  if (!s) return;
  const parts = [];

  const makeBadge = (name, value, warn, modal) => {
    const cls = warn ? 'stat-item stat-warning' : 'stat-item';
    const click = modal ? ' clickable' : '';
    const data = modal ? ` data-modal="${modal}"` : '';
    return `<span class="${cls}${click}"${data}><span class="stat-name">${name}</span><span class="stat-value">${value}</span></span>`;
  };

  parts.push(makeBadge('Episodes', s.episodes.active));
  parts.push(makeBadge('Entities', s.entities.active));
  parts.push(makeBadge('Relations', s.relations.active));

  const mergeWarn = s.merge_candidates.pending > 0;
  parts.push(makeBadge('Merge', s.merge_candidates.pending, mergeWarn, 'merge'));

  if (s.doc_status_failed > 0) {
    parts.push(makeBadge('Failed', '⚠ ' + s.doc_status_failed, true, 'failed'));
  }

  const orphanTotal = (s.orphans.episodes || 0) + (s.orphans.entities || 0);
  if (orphanTotal > 0) {
    parts.push(makeBadge('Orphans', orphanTotal, false, 'orphan'));
  }

  const target = document.getElementById('stats-items') || document.getElementById('stats-bar');
  target.innerHTML = parts.join('');
}

// ─── Detail rendering ───────────────────────────────────────────────────────
function renderDetail(data) {
  if (!data) return '<div class="empty-state">データがありません</div>';

  const rows = [];
  const pushRow = (label, value, wrap) => {
    const v = value === null || value === undefined ? '—' : String(value);
    rows.push(`<div class="detail-row"><span class="detail-label">${label}</span><span class="detail-value ${wrap ? 'wrap' : ''}">${escapeHtml(v)}</span></div>`);
  };

  if (data.content !== undefined) {
    // episode
    pushRow('ID', data.id);
    pushRow('Type', 'episode');
    pushRow('Kind', data.kind);
    pushRow('Importance', data.importance);
    pushRow('Content', data.content, true);
    pushRow('Session', data.session_id);
    pushRow('Valid from', data.valid_from);
    pushRow('Valid to', data.valid_to);
    pushRow('Supersedes', data.supersedes);
    pushRow('Suppressed', data.is_suppressed);
    pushRow('Activated', data.activation_count);
    pushRow('Last activated', data.last_activated_at);
    pushRow('Created', data.created_at);
    if (data.doc_status) {
      pushRow('Doc stage', data.doc_status.stage);
      pushRow('Doc error', data.doc_status.error);
      pushRow('Doc updated', data.doc_status.updated_at);
    }
    if (data.mentions && data.mentions.length) {
      rows.push('<div class="detail-section"><h4>Mentions</h4><ul class="detail-list">' +
        data.mentions.map(m => `<li>${escapeHtml(m.canonical_name)}${m.is_self ? ' (self)' : ''}</li>`).join('') +
        '</ul></div>');
    }
  } else if (data.canonical_name !== undefined) {
    // entity
    pushRow('ID', data.id);
    pushRow('Type', 'entity');
    pushRow('Name', data.canonical_name);
    pushRow('Entity type', data.entity_type);
    // Description: edit ボタン付き。archived (valid_to が立った) entity は編集不可。
    const editable = data.valid_to == null;
    const descVal = data.description === null || data.description === undefined ? '—' : String(data.description);
    const descEditBtn = editable ? `<button type="button" class="edit-btn" data-edit-field="description" data-entity-id="${data.id}" aria-label="Description を編集">✎</button>` : '';
    rows.push(`<div class="detail-row" data-field-row="description"><span class="detail-label">Description${descEditBtn}</span><span class="detail-value wrap">${escapeHtml(descVal)}</span></div>`);
    if (data.curated_at) {
      pushRow('Curated at', data.curated_at);
    }
    pushRow('Self', data.is_self);
    pushRow('Self weight', data.self_weight);
    pushRow('Decay rate', data.decay_rate);
    pushRow('Valid from', data.valid_from);
    pushRow('Valid to', data.valid_to);
    pushRow('Supersedes', data.supersedes);
    pushRow('Activated', data.activation_count);
    pushRow('Last activated', data.last_activated_at);
    pushRow('Created', data.created_at);
    // Aliases: edit ボタン付き。常に行を出す (空でも編集できるように)。
    const aliasesVal = data.aliases && data.aliases.length ? data.aliases.join(', ') : '—';
    const aliasesEditBtn = editable ? `<button type="button" class="edit-btn" data-edit-field="aliases" data-entity-id="${data.id}" aria-label="Aliases を編集">✎</button>` : '';
    rows.push(`<div class="detail-row" data-field-row="aliases"><span class="detail-label">Aliases${aliasesEditBtn}</span><span class="detail-value wrap">${escapeHtml(aliasesVal)}</span></div>`);
    if (data.in_relations && data.in_relations.length) {
      rows.push('<div class="detail-section"><h4>In relations</h4><ul class="detail-list">' +
        data.in_relations.map(r => `<li>${escapeHtml(r.predicate)} from ${escapeHtml(r.src_type)}:${r.src_id}</li>`).join('') +
        '</ul></div>');
    }
    if (data.out_relations && data.out_relations.length) {
      rows.push('<div class="detail-section"><h4>Out relations</h4><ul class="detail-list">' +
        data.out_relations.map(r => `<li>${escapeHtml(r.predicate)} to ${escapeHtml(r.dst_type)}:${r.dst_id}${r.strength != null ? ` (strength ${r.strength})` : ''}</li>`).join('') +
        '</ul></div>');
    }
  } else if (data.predicate !== undefined) {
    // relation
    pushRow('ID', data.id);
    pushRow('Type', 'relation');
    pushRow('Predicate', data.predicate);
    pushRow('Source', `${data.src_type}:${data.src_id} (${data.src_label || '?'})`);
    pushRow('Target', `${data.dst_type}:${data.dst_id} (${data.dst_label || '?'})`);
    pushRow('Strength', data.strength);
    pushRow('Fan out', data.fan_out);
    pushRow('Description', data.description, true);
    pushRow('Valid from', data.valid_from);
    pushRow('Valid to', data.valid_to);
    pushRow('Supersedes', data.supersedes);
    pushRow('Created', data.created_at);
  }

  return rows.join('');
}

function showDetail(cyId, typeHint) {
  state.selected = cyId;
  const prefix = cyId.split('_')[0];
  const dbId = parseInt(cyId.split('_')[1], 10);
  let path = null;

  if (prefix === 'ep') {
    path = `/api/episodes/${dbId}`;
  } else if (prefix === 'en') {
    path = `/api/entities/${dbId}`;
  } else if (prefix === 'men' || prefix === 'rel' || typeHint === 'relation') {
    path = `/api/relations/${dbId}`;
  }

  if (!path) return;

  api(path)
    .then((data) => {
      const html = renderDetail(data);
      if (state.view === 'panel') {
        document.getElementById('detail-panel').innerHTML = html;
      } else {
        const popover = document.getElementById('popover');
        popover.innerHTML = '<button class="popover-close">&times;</button>' + html;
        popover.querySelector('.popover-close').addEventListener('click', hidePopover);
        popover.hidden = false;
        if (state.cy) {
          const ele = state.cy.getElementById(cyId);
          if (ele.length) {
            const pos = ele.isEdge() ? ele.renderedMidpoint() : ele.renderedPosition();
            const container = document.getElementById('cy');
            const rect = container.getBoundingClientRect();
            let left = pos.x + 12;
            let top = pos.y + 12;
            if (left + 320 > rect.width) left = pos.x - 332;
            if (top + 220 > rect.height) top = pos.y - 232;
            popover.style.left = Math.max(4, left) + 'px';
            popover.style.top = Math.max(4, top) + 'px';
          }
        }
      }
    })
    .catch((err) => {
      console.error('detail fetch failed', err);
      const msg = '<div class="empty-state">詳細の取得に失敗しました</div>';
      if (state.view === 'panel') {
        document.getElementById('detail-panel').innerHTML = msg;
      } else {
        document.getElementById('popover').innerHTML = msg;
        document.getElementById('popover').hidden = false;
      }
    });
}

function hidePopover() {
  const popover = document.getElementById('popover');
  popover.hidden = true;
  popover.innerHTML = '';
}

// ─── Modal ──────────────────────────────────────────────────────────────────
function openModal(title, bodyHtml) {
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-body').innerHTML = bodyHtml;
  document.getElementById('modal-root').hidden = false;
}

function closeModal() {
  document.getElementById('modal-root').hidden = true;
  document.getElementById('modal-body').innerHTML = '';
}

async function openMergeModal() {
  const data = await api('/api/merge_candidates?status=pending');
  if (!data.candidates || !data.candidates.length) {
    openModal('Merge candidates', '<div class="empty-state">pending はありません</div>');
    return;
  }
  const rows = data.candidates.map((c) =>
    `<tr class="clickable" data-node-id="en_${c.entity_a.id}">` +
    `<td>${escapeHtml(c.entity_a.canonical_name)}</td>` +
    `<td>${escapeHtml(c.entity_b.canonical_name)}</td>` +
    `<td>${(c.similarity * 100).toFixed(1)}%</td>` +
    `</tr>`
  ).join('');
  const html =
    '<table><thead><tr><th>Entity A</th><th>Entity B</th><th>Similarity</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table>';
  openModal('Merge candidates', html);
  document.querySelectorAll('#modal-body tr[data-node-id]').forEach((tr) => {
    tr.addEventListener('click', () => {
      const id = tr.dataset.nodeId;
      closeModal();
      if (state.cy) {
        state.cy.$(`#${id}`).select();
        showDetail(id, 'entity');
      }
    });
  });
}

async function openFailedModal() {
  const data = await api('/api/doc_status?status=failed');
  if (!data.items || !data.items.length) {
    openModal('Failed doc_status', '<div class="empty-state">失敗はありません</div>');
    return;
  }
  const rows = data.items.map((item) =>
    `<tr class="clickable" data-node-id="ep_${item.episode_id}">` +
    `<td>${escapeHtml(item.episode_label || '—')}</td>` +
    `<td>${escapeHtml(item.error || '—')}</td>` +
    `<td>${escapeHtml(item.stage)}</td>` +
    `</tr>`
  ).join('');
  const html =
    '<table><thead><tr><th>Episode</th><th>Error</th><th>Stage</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table>';
  openModal('Failed doc_status', html);
  document.querySelectorAll('#modal-body tr[data-node-id]').forEach((tr) => {
    tr.addEventListener('click', () => {
      const id = tr.dataset.nodeId;
      closeModal();
      if (state.cy) {
        state.cy.$(`#${id}`).select();
        showDetail(id, 'episode');
      }
    });
  });
}

async function openOrphanModal() {
  const data = await api('/api/orphans?scope=active');
  let html = '';
  if (data.episodes && data.episodes.length) {
    html += '<h4>Orphan episodes</h4>';
    const rows = data.episodes.map((e) =>
      `<tr class="clickable" data-node-id="ep_${e.id}">` +
      `<td>${e.id}</td><td>${escapeHtml(e.label || '—')}</td><td>${escapeHtml(e.kind || '—')}</td><td>${escapeHtml(e.created_at || '—')}</td>` +
      `</tr>`
    ).join('');
    html += '<table><thead><tr><th>ID</th><th>Label</th><th>Kind</th><th>Created</th></tr></thead><tbody>' + rows + '</tbody></table>';
  }
  if (data.entities && data.entities.length) {
    html += '<h4>Orphan entities</h4>';
    const rows = data.entities.map((e) =>
      `<tr class="clickable" data-node-id="en_${e.id}">` +
      `<td>${e.id}</td><td>${escapeHtml(e.canonical_name || '—')}</td><td>${e.is_self ? 'self' : '—'}</td><td>${escapeHtml(e.created_at || '—')}</td>` +
      `</tr>`
    ).join('');
    html += '<table><thead><tr><th>ID</th><th>Name</th><th>Self</th><th>Created</th></tr></thead><tbody>' + rows + '</tbody></table>';
  }
  if (!html) {
    html = '<div class="empty-state">orphan はありません</div>';
  }
  openModal('Orphans', html);
  document.querySelectorAll('#modal-body tr[data-node-id]').forEach((tr) => {
    tr.addEventListener('click', () => {
      const id = tr.dataset.nodeId;
      closeModal();
      if (state.cy) {
        state.cy.$(`#${id}`).select();
        showDetail(id, id.startsWith('ep_') ? 'episode' : 'entity');
      }
    });
  });
}

// ─── View / Scope / Theme ───────────────────────────────────────────────────
function setView(mode) {
  state.view = mode;
  localStorage.setItem('admin_view', mode);
  document.body.classList.toggle('view-panel', mode === 'panel');
  document.body.classList.toggle('view-popover', mode === 'popover');

  const toggle = document.getElementById('view-toggle');
  if (toggle) toggle.checked = mode === 'popover';

  if (mode === 'panel') {
    hidePopover();
  }
  if (state.selected) {
    showDetail(state.selected);
  } else {
    hidePopover();
    if (mode === 'panel') {
      document.getElementById('detail-panel').innerHTML = '';
    }
  }
}

function setTheme(theme) {
  state.theme = theme;
  localStorage.setItem('admin_theme', theme);
  document.body.setAttribute('data-theme', theme);

  const toggle = document.getElementById('theme-toggle');
  if (toggle) toggle.checked = theme === 'light';

  applyGraphTheme();
}

async function setScope(scope) {
  if (scope === 'all') {
    const total = ((state.stats?.episodes?.total) || 0) + ((state.stats?.entities?.total) || 0);
    if (!confirm(`現役以外も含めて ${total} 件のノードが描画されます。重くなる可能性があります。続行しますか？`)) {
      const radio = document.querySelector(`input[name="scope"][value="${state.scope}"]`);
      if (radio) radio.checked = true;
      return;
    }
  }
  state.scope = scope;
  await reloadGraph();
}

async function reloadGraph() {
  try {
    state.graphData = await api(`/api/graph?scope=${state.scope}`);
    if (state.cy) {
      state.cy.elements().remove();
      state.cy.add(state.graphData.elements);
      applyFilters();
      runLayout();
    }
  } catch (err) {
    console.error('reloadGraph failed', err);
  }
}

async function reloadAll() {
  try {
    state.stats = await api('/api/stats');
    renderStatsBar();
  } catch (err) {
    console.error('stats fetch failed', err);
  }
  await reloadGraph();
}

// ─── Events ─────────────────────────────────────────────────────────────────
function setupEventListeners() {
  // Scope
  document.querySelectorAll('input[name="scope"]').forEach((el) => {
    el.addEventListener('change', (e) => {
      if (e.target.checked) setScope(e.target.value);
    });
  });

  // View toggle
  const viewToggle = document.getElementById('view-toggle');
  if (viewToggle) {
    viewToggle.addEventListener('change', (e) => {
      setView(e.target.checked ? 'popover' : 'panel');
    });
  }

  // Theme toggle
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('change', (e) => {
      setTheme(e.target.checked ? 'light' : 'dark');
    });
  }

  // Kind filters
  document.querySelectorAll('input[name="kind"]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.filters.kind[e.target.value] = e.target.checked;
      applyFilters();
    });
  });

  // Type filters
  document.querySelectorAll('input[name="type"]').forEach((el) => {
    el.addEventListener('change', (e) => {
      state.filters.type[e.target.value] = e.target.checked;
      applyFilters();
    });
  });

  // Spread slider
  const spreadSlider = document.getElementById('spread-slider');
  if (spreadSlider) {
    spreadSlider.value = String(state.spread);
    const valueEl = document.getElementById('spread-value');
    if (valueEl) valueEl.textContent = state.spread.toFixed(1);
    spreadSlider.addEventListener('input', (e) => {
      setSpread(parseFloat(e.target.value));
    });
  }

  // Importance filter
  const impSlider = document.getElementById('importance-slider');
  if (impSlider) {
    impSlider.addEventListener('input', (e) => {
      const v = parseInt(e.target.value, 10);
      state.filters.importance = v;
      const label = document.getElementById('importance-value');
      if (label) label.textContent = v === 1 ? 'all' : `≥ ${v}`;
      applyFilters();
    });
  }

  // Degree filter
  const degSlider = document.getElementById('degree-slider');
  if (degSlider) {
    degSlider.addEventListener('input', (e) => {
      const v = parseInt(e.target.value, 10);
      state.filters.degree = v;
      const label = document.getElementById('degree-value');
      if (label) label.textContent = v === 0 ? 'all' : `≥ ${v}`;
      applyFilters();
    });
  }

  // Reload
  document.getElementById('reload').addEventListener('click', reloadAll);

  // Stats bar clicks (delegate)
  document.getElementById('stats-bar').addEventListener('click', (e) => {
    const item = e.target.closest('[data-modal]');
    if (!item) return;
    const modal = item.dataset.modal;
    if (modal === 'merge') openMergeModal();
    else if (modal === 'failed') openFailedModal();
    else if (modal === 'orphan') openOrphanModal();
  });

  // Search box
  const searchInput = document.getElementById('search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => setSearchQuery(e.target.value));
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        searchInput.value = '';
        setSearchQuery('');
      }
    });
  }
  const searchClear = document.getElementById('search-clear');
  if (searchClear) {
    searchClear.addEventListener('click', () => {
      if (searchInput) searchInput.value = '';
      setSearchQuery('');
      if (searchInput) searchInput.focus();
    });
  }

  // Modal close
  document.getElementById('modal-close').addEventListener('click', closeModal);
  document.querySelector('.modal-overlay').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeModal();
  });

  // Entity inline edit (delegation on document.body — covers both panel and popover)
  document.body.addEventListener('click', (e) => {
    const editBtn = e.target.closest('.edit-btn[data-edit-field]');
    if (editBtn) {
      e.preventDefault();
      enterEntityEditMode(editBtn);
      return;
    }
    const saveBtn = e.target.closest('.edit-save');
    if (saveBtn) {
      e.preventDefault();
      submitEntityEdit(saveBtn);
      return;
    }
    const cancelBtn = e.target.closest('.edit-cancel');
    if (cancelBtn) {
      e.preventDefault();
      // 取り直しで素の表示に戻す (ローカル復元より確実)
      if (state.selected) showDetail(state.selected, 'entity');
      return;
    }
  });
}

function enterEntityEditMode(btn) {
  const field = btn.dataset.editField;
  const entityId = btn.dataset.entityId;
  const row = btn.closest('[data-field-row]');
  if (!row || !field || !entityId) return;
  if (row.dataset.editing === 'true') return;
  row.dataset.editing = 'true';

  const currentValueEl = row.querySelector('.detail-value');
  const currentText = currentValueEl ? currentValueEl.textContent : '';
  const isDescription = field === 'description';
  const isPlaceholder = currentText === '—';
  const initial = isPlaceholder ? '' : currentText;

  const inputHtml = isDescription
    ? `<textarea class="edit-input" rows="4" maxlength="4000">${escapeHtml(initial)}</textarea>`
    : `<input class="edit-input" type="text" value="${escapeHtml(initial)}" placeholder="comma-separated">`;
  const helperHtml = isDescription
    ? '<div class="edit-helper">最大 4000 文字</div>'
    : '<div class="edit-helper">カンマ区切り (空にすると alias を全削除)</div>';

  currentValueEl.innerHTML =
    inputHtml +
    helperHtml +
    `<div class="edit-actions">` +
    `<button type="button" class="edit-save" data-edit-field="${field}" data-entity-id="${entityId}">保存</button>` +
    `<button type="button" class="edit-cancel">キャンセル</button>` +
    `</div>`;

  const input = currentValueEl.querySelector('.edit-input');
  if (input) input.focus();
}

async function submitEntityEdit(btn) {
  const field = btn.dataset.editField;
  const entityId = btn.dataset.entityId;
  const row = btn.closest('[data-field-row]');
  if (!row || !field || !entityId) return;
  const input = row.querySelector('.edit-input');
  if (!input) return;

  let payload;
  if (field === 'description') {
    payload = { description: input.value };
  } else if (field === 'aliases') {
    const aliases = input.value
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    payload = { aliases };
  } else {
    return;
  }

  btn.disabled = true;
  const cancelBtn = row.querySelector('.edit-cancel');
  if (cancelBtn) cancelBtn.disabled = true;

  try {
    const res = await fetch(`/api/entities/${entityId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      alert(`更新に失敗しました (${res.status}): ${text}`);
      btn.disabled = false;
      if (cancelBtn) cancelBtn.disabled = false;
      return;
    }
    // 成功 → detail を再取得して再描画 (curated_at の更新も反映される)
    if (state.selected) showDetail(state.selected, 'entity');
  } catch (err) {
    console.error('entity update failed', err);
    alert('更新に失敗しました (network error)');
    btn.disabled = false;
    if (cancelBtn) cancelBtn.disabled = false;
  }
}

// ─── Boot ───────────────────────────────────────────────────────────────────
async function main() {
  try {
    state.stats = await api('/api/stats');
    renderStatsBar();
  } catch (err) {
    console.error('initial stats fetch failed', err);
  }

  try {
    state.graphData = await api(`/api/graph?scope=${state.scope}`);
    state.cy = initCytoscape(state.graphData.elements);
    runLayout();
    applyFilters();
  } catch (err) {
    console.error('initial graph fetch failed', err);
    document.getElementById('cy').innerHTML = '<div class="empty-state" style="padding:2rem">グラフの読み込みに失敗しました</div>';
  }

  setView(state.view);
  setTheme(state.theme);
  setupEventListeners();
}

main();
