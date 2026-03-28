'use strict';

/**
 * AOP-Wiki RDF Dashboard - Network Graph Interactivity
 *
 * Client-side logic for the /network page. Initializes Cytoscape.js with
 * preset layout (pre-computed positions), handles node interaction (click, search, filter), metric-based
 * sizing, sortable metrics table, community summary, tab switching, and exports.
 *
 * Dependencies (loaded via CDN in network.html):
 *   - Cytoscape.js 3.33.x
 *   - cytoscape-fcose 2.2.x
 *
 * API endpoints consumed:
 *   - GET /api/network/graph       -> { elements, version, stats }
 *   - GET /api/network/metrics     -> { metrics, version }
 *   - GET /api/network/communities -> { communities, version }
 *
 * Export endpoints:
 *   - GET /download/network/metrics -> CSV file
 *   - GET /download/network/graph   -> JSON file
 */

// ============================================================
// Module-scoped state
// ============================================================

/** @type {cytoscape.Core|null} */
let cy = null;

/** @type {Object|null} - Stats from the graph API response */
let graphStats = null;

/** @type {Array|null} - Metrics array from the metrics API */
let metricsData = null;

/** @type {Array|null} - Communities array from the communities API */
let communitiesData = null;

/** @type {string} - Currently selected sizing metric */
let currentMetric = 'pagerank';

/** @type {string} - Current sort column for metrics table */
let sortColumn = 'pagerank';

/** @type {boolean} - Current sort direction for metrics table */
let sortAscending = false;


// ============================================================
// Section 1: Initialization and Graph Loading
// ============================================================

/**
 * Main entry point. Called on DOMContentLoaded.
 * Loads graph data from API, initializes Cytoscape.js, then fetches
 * metrics and communities for the Metrics tab.
 */
async function initNetworkGraph() {
    const overlay = document.getElementById('loading-overlay');
    const container = document.getElementById('cy-container');

    try {
        // Show loading overlay
        if (overlay) overlay.classList.remove('hidden');

        // Fetch graph data
        const response = await fetch('/api/network/graph');
        if (!response.ok) {
            throw new Error(`Graph API returned ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        graphStats = data.stats || {};

        // Register fcose layout extension
        if (typeof cytoscapeFcose !== 'undefined') {
            cytoscape.use(cytoscapeFcose);
        }

        // Initialize Cytoscape.js
        cy = cytoscape({
            container: container,
            elements: data.elements,
            boxSelectionEnabled: true,
            style: [
                {
                    selector: 'node',
                    style: {
                        'shape': 'ellipse',
                        'background-color': 'data(color)',
                        'label': 'data(label)',
                        'width': 'mapData(pagerank, 0, 0.01, 15, 70)',
                        'height': 'mapData(pagerank, 0, 0.01, 15, 70)',
                        'font-size': '7px',
                        'text-wrap': 'ellipsis',
                        'text-max-width': '80px',
                        'min-zoomed-font-size': 12,
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'color': '#29235C'
                    }
                },
                {
                    selector: 'node:selected',
                    style: {
                        'border-width': 3,
                        'border-color': '#E6007E'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'line-color': '#93D5F6',
                        'opacity': 0.5,
                        'width': 1.5,
                        'target-arrow-shape': 'triangle',
                        'target-arrow-color': '#93D5F6',
                        'curve-style': 'bezier'
                    }
                }
            ],
            layout: {
                name: 'preset'
            }
        });

        // Hide loading overlay
        if (overlay) overlay.classList.add('hidden');

        // Update stats bar with initial counts
        updateStatsBar(graphStats);

        // Set up all interaction handlers
        setupNodeEvents(cy);
        setupSearch(cy);
        setupFilters(cy);
        setupMetricSelector(cy);

        // Apply initial metric-based sizing with actual data range
        updateNodeSizing(cy, currentMetric);

        // Fetch metrics and communities for the Metrics tab
        fetchMetricsAndCommunities();

    } catch (err) {
        console.error('Network graph initialization failed:', err);
        if (overlay) overlay.classList.add('hidden');
        if (container) {
            container.innerHTML =
                '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:#333;">' +
                '<p style="font-size:16px;font-weight:600;color:#29235C;">Failed to load network graph</p>' +
                '<p style="font-size:13px;color:#666;margin:8px 0;">' + escapeHtml(err.message) + '</p>' +
                '<button onclick="initNetworkGraph()" style="padding:8px 20px;background:#E6007E;color:white;border:none;border-radius:4px;cursor:pointer;font-size:14px;">Retry</button>' +
                '</div>';
        }
    }
}

/**
 * Fetch metrics and communities data for the Metrics & Communities tab.
 */
async function fetchMetricsAndCommunities() {
    try {
        const [metricsResp, commResp] = await Promise.all([
            fetch('/api/network/metrics'),
            fetch('/api/network/communities')
        ]);

        if (metricsResp.ok) {
            const metricsJson = await metricsResp.json();
            metricsData = metricsJson.metrics || [];
            populateMetricsTable(metricsData);
        }

        if (commResp.ok) {
            const commJson = await commResp.json();
            communitiesData = commJson.communities || [];
            populateCommunities(communitiesData);
        }
    } catch (err) {
        console.error('Failed to fetch metrics/communities:', err);
    }
}


// ============================================================
// Section 2: Node Click and Info Panel
// ============================================================

/**
 * Set up node click and background click event handlers.
 * @param {cytoscape.Core} cyInstance
 */
function setupNodeEvents(cyInstance) {
    // Node click -> open info panel
    cyInstance.on('tap', 'node', function (evt) {
        const node = evt.target;
        showInfoPanel(node.data(), node.id());
    });

    // Background click -> close info panel
    cyInstance.on('tap', function (evt) {
        if (evt.target === cyInstance) {
            hideInfoPanel();
        }
    });

    // Close button
    const closeBtn = document.getElementById('info-panel-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', function () {
            hideInfoPanel();
        });
    }
}

/**
 * Populate and show the info panel with node details.
 * @param {Object} nodeData - Cytoscape node data object
 * @param {string} nodeId - The node ID in the Cytoscape graph
 */
function showInfoPanel(nodeData, nodeId) {
    const panel = document.getElementById('info-panel');
    if (!panel) return;

    // Title
    const titleEl = document.getElementById('info-panel-title');
    if (titleEl) titleEl.textContent = nodeData.label || nodeData.id;

    // Type badge
    const typeBadge = document.getElementById('info-type-badge');
    if (typeBadge) {
        var roleLabels = {MIE: 'Molecular Initiating Event', KE: 'Key Event', AO: 'Adverse Outcome'};
        var nodeType = nodeData.type || 'KE';
        typeBadge.textContent = roleLabels[nodeType] || nodeType;
        typeBadge.className = 'type-badge ' + nodeType.toLowerCase();
    }

    // Centrality metrics
    setMetricValue('info-degree', nodeData.degree, 4);
    setMetricValue('info-betweenness', nodeData.betweenness, 4);
    setMetricValue('info-closeness', nodeData.closeness, 4);
    setMetricValue('info-pagerank', nodeData.pagerank, 6);

    // Community
    var communityEl = document.getElementById('info-community');
    if (communityEl) {
        communityEl.textContent = 'Community ' + nodeData.community;
    }

    // Neighbors
    const neighborsList = document.getElementById('info-neighbors');
    if (neighborsList && cy) {
        const node = cy.getElementById(nodeId);
        const neighbors = node.neighborhood('node');
        neighborsList.innerHTML = '';

        if (neighbors.length === 0) {
            neighborsList.innerHTML = '<li style="color:#999;cursor:default;">No connected nodes</li>';
        } else {
            neighbors.forEach(function (n) {
                const li = document.createElement('li');
                li.appendChild(document.createTextNode(n.data('label') || n.id()));
                li.dataset.nodeId = n.id();
                li.addEventListener('click', function () {
                    const targetNode = cy.getElementById(this.dataset.nodeId);
                    cy.animate({
                        center: { eles: targetNode },
                        zoom: 2
                    }, { duration: 500 });
                    targetNode.select();
                    showInfoPanel(targetNode.data(), targetNode.id());
                });
                neighborsList.appendChild(li);
            });
        }
    }

    // AOP-Wiki link
    const wikiLink = document.getElementById('info-wiki-link');
    if (wikiLink) {
        let url = nodeData.wiki_url || '';
        if (!url && nodeData.uri) {
            // Fallback: construct URL from URI
            // e.g., http://aopwiki.org/aops/1 -> https://aopwiki.org/aops/1
            url = nodeData.uri.replace('http://', 'https://');
        }
        if (url) {
            wikiLink.href = url;
            wikiLink.style.display = 'inline-block';
        } else {
            wikiLink.style.display = 'none';
        }
    }

    // Open panel with animation
    panel.classList.add('open');
}

/**
 * Close the info panel and restore graph container width.
 */
function hideInfoPanel() {
    const panel = document.getElementById('info-panel');
    if (panel) panel.classList.remove('open');

    // Deselect all nodes
    if (cy) cy.elements().unselect();
}

/**
 * Set a metric value in the info panel.
 * @param {string} elementId - DOM element ID
 * @param {number} value - Metric value
 * @param {number} decimals - Number of decimal places
 */
function setMetricValue(elementId, value, decimals) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = (value !== undefined && value !== null)
            ? Number(value).toFixed(decimals)
            : '-';
    }
}


// ============================================================
// Section 3: Search with Type-Ahead
// ============================================================

/**
 * Set up search input with debounced type-ahead filtering.
 * @param {cytoscape.Core} cyInstance
 */
function setupSearch(cyInstance) {
    const searchInput = document.getElementById('node-search');
    const resultsDiv = document.getElementById('search-results');
    if (!searchInput || !resultsDiv) return;

    let debounceTimer = null;

    searchInput.addEventListener('input', function () {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(function () {
            const query = searchInput.value.toLowerCase().trim();

            if (query.length < 2) {
                resultsDiv.style.display = 'none';
                resultsDiv.innerHTML = '';
                return;
            }

            const matches = cyInstance.nodes().filter(function (n) {
                const label = (n.data('label') || '').toLowerCase();
                const id = (n.data('id') || '').toLowerCase();
                return label.includes(query) || id.includes(query);
            });

            // Limit to first 20 results
            const limited = matches.slice(0, 20);

            if (limited.length === 0) {
                resultsDiv.innerHTML = '<div class="search-result" style="color:#999;cursor:default;">No matches found</div>';
                resultsDiv.style.display = 'block';
                return;
            }

            let html = '';
            limited.forEach(function (n) {
                const typeClass = (n.data('type') || 'KE').toLowerCase();
                html += '<div class="search-result" data-node-id="' + escapeHtml(n.id()) + '">' +
                    '<span class="type-badge ' + typeClass + '">' + escapeHtml(n.data('type')) + '</span> ' +
                    escapeHtml(n.data('label') || n.id()) +
                    '</div>';
            });
            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }, 200);
    });

    // Click on a search result
    resultsDiv.addEventListener('click', function (e) {
        const result = e.target.closest('.search-result');
        if (!result || !result.dataset.nodeId) return;

        const nodeId = result.dataset.nodeId;
        const node = cyInstance.getElementById(nodeId);

        if (node && node.length > 0) {
            cyInstance.animate({
                center: { eles: node },
                zoom: 2
            }, { duration: 500 });
            node.select();
            showInfoPanel(node.data(), node.id());
        }

        // Clear search
        searchInput.value = '';
        resultsDiv.style.display = 'none';
        resultsDiv.innerHTML = '';
    });

    // Close results when clicking outside
    document.addEventListener('click', function (e) {
        if (!e.target.closest('.search-container')) {
            resultsDiv.style.display = 'none';
        }
    });
}


// ============================================================
// Section 4: Filter Panel
// ============================================================

/**
 * Set up filter panel checkboxes, dropdowns, and reset button.
 * @param {cytoscape.Core} cyInstance
 */
function setupFilters(cyInstance) {
    const roleSelect = document.getElementById('filter-role');
    const resetBtn = document.getElementById('reset-filters');

    const applyCurrentFilters = function () {
        applyFilters(cyInstance);
    };

    if (roleSelect) roleSelect.addEventListener('change', applyCurrentFilters);

    // Reset filters
    if (resetBtn) {
        resetBtn.addEventListener('click', function () {
            if (roleSelect) roleSelect.value = 'all';
            applyFilters(cyInstance);
        });
    }
}

/**
 * Apply current filter state to graph nodes and edges.
 * @param {cytoscape.Core} cyInstance
 */
function applyFilters(cyInstance) {
    const roleSelect = document.getElementById('filter-role');
    const roleValue = roleSelect ? roleSelect.value : 'all';

    // Start by showing all elements
    cyInstance.elements().show();

    // Role filter
    if (roleValue !== 'all') {
        cyInstance.nodes().forEach(function (n) {
            if (n.data('type') !== roleValue) {
                n.hide();
            }
        });
    }

    // Hide edges where either endpoint is hidden
    cyInstance.edges().forEach(function (e) {
        if (!e.source().visible() || !e.target().visible()) {
            e.hide();
        }
    });

    // Update stats with filtered counts
    updateFilteredStats(cyInstance);
}

/**
 * Update stats bar with filtered node/edge counts.
 * @param {cytoscape.Core} cyInstance
 */
function updateFilteredStats(cyInstance) {
    const visibleNodes = cyInstance.nodes().filter(function (n) { return n.visible(); });
    const visibleEdges = cyInstance.edges().filter(function (e) { return e.visible(); });

    // Count unique communities among visible nodes
    const visibleCommunities = new Set();
    visibleNodes.forEach(function (n) {
        visibleCommunities.add(n.data('community'));
    });

    updateStatsBar({
        nodes: visibleNodes.length,
        edges: visibleEdges.length,
        ke_count: visibleNodes.length,
        ker_count: visibleEdges.length,
        communities: visibleCommunities.size
    });
}


// ============================================================
// Section 5: Metric-Based Node Sizing
// ============================================================

/**
 * Set up the metric selector dropdown handler.
 * @param {cytoscape.Core} cyInstance
 */
function setupMetricSelector(cyInstance) {
    const selector = document.getElementById('metric-selector');
    if (!selector) return;

    selector.addEventListener('change', function () {
        currentMetric = this.value;
        updateNodeSizing(cyInstance, currentMetric);
    });
}

/**
 * Update node sizing based on the selected centrality metric.
 * Computes min/max values from the graph data and updates Cytoscape styles
 * with mapData expressions for proportional sizing.
 * @param {cytoscape.Core} cyInstance
 * @param {string} metric - One of 'degree', 'betweenness', 'closeness', 'pagerank'
 */
function updateNodeSizing(cyInstance, metric) {
    const values = cyInstance.nodes().map(function (n) { return n.data(metric) || 0; });
    if (values.length === 0) return;

    const minVal = Math.min.apply(null, values);
    let maxVal = Math.max.apply(null, values);

    // Prevent zero-range (all same values)
    if (maxVal === minVal) {
        maxVal = minVal + 0.001;
    }

    cyInstance.style()
        .selector('node')
        .style({
            'width': 'mapData(' + metric + ', ' + minVal + ', ' + maxVal + ', 15, 70)',
            'height': 'mapData(' + metric + ', ' + minVal + ', ' + maxVal + ', 15, 70)'
        })
        .update();
}


// ============================================================
// Section 6: Metrics Table (Sortable) and Community Summary
// ============================================================

/**
 * Populate the metrics table in the Metrics tab with sortable columns.
 * @param {Array} metrics - Array of metric objects from the API
 */
function populateMetricsTable(metrics) {
    const tbody = document.getElementById('metrics-tbody');
    if (!tbody || !metrics) return;

    // Store metrics for sorting
    metricsData = metrics;

    // Set up sortable headers
    const headers = document.querySelectorAll('.metrics-table th.sortable');
    headers.forEach(function (th) {
        th.addEventListener('click', function () {
            const col = this.dataset.sort;
            if (sortColumn === col) {
                sortAscending = !sortAscending;
            } else {
                sortColumn = col;
                sortAscending = true;
            }
            renderMetricsTable();
            updateSortIndicators();
        });
    });

    // Initial render
    renderMetricsTable();
    updateSortIndicators();
}

/**
 * Render the metrics table body based on current sort state.
 */
function renderMetricsTable() {
    const tbody = document.getElementById('metrics-tbody');
    if (!tbody || !metricsData) return;

    // Sort the data
    const sorted = metricsData.slice().sort(function (a, b) {
        let valA = a[sortColumn];
        let valB = b[sortColumn];

        // Numeric comparison for metric columns
        if (typeof valA === 'number' && typeof valB === 'number') {
            return sortAscending ? valA - valB : valB - valA;
        }

        // String comparison for text columns
        valA = String(valA || '').toLowerCase();
        valB = String(valB || '').toLowerCase();
        if (valA < valB) return sortAscending ? -1 : 1;
        if (valA > valB) return sortAscending ? 1 : -1;
        return 0;
    });

    let html = '';
    sorted.forEach(function (row) {
        html += '<tr data-node-id="' + escapeHtml(row.id) + '" style="cursor:pointer;">' +
            '<td>' + escapeHtml(row.label || row.id) + '</td>' +
            '<td><span class="type-badge ' + (row.type || 'KE').toLowerCase() + '">' + escapeHtml(row.type || 'KE') + '</span></td>' +
            '<td>' + Number(row.degree).toFixed(4) + '</td>' +
            '<td>' + Number(row.betweenness).toFixed(4) + '</td>' +
            '<td>' + Number(row.closeness).toFixed(4) + '</td>' +
            '<td>' + Number(row.pagerank).toFixed(6) + '</td>' +
            '<td>' + row.community + '</td>' +
            '</tr>';
    });

    tbody.innerHTML = html;

    // Click a row to center the graph on that node
    tbody.querySelectorAll('tr').forEach(function (tr) {
        tr.addEventListener('click', function () {
            const nodeId = this.dataset.nodeId;
            if (!cy || !nodeId) return;

            const node = cy.getElementById(nodeId);
            if (node && node.length > 0) {
                // Switch to Graph tab
                activateTab('graph-view');
                // Center on node with slight delay for tab switch
                setTimeout(function () {
                    cy.resize();
                    cy.animate({
                        center: { eles: node },
                        zoom: 2
                    }, { duration: 500 });
                    node.select();
                    showInfoPanel(node.data(), node.id());
                }, 100);
            }
        });
    });
}

/**
 * Update sort indicator arrows on table headers.
 */
function updateSortIndicators() {
    const headers = document.querySelectorAll('.metrics-table th.sortable');
    headers.forEach(function (th) {
        const indicator = th.querySelector('.sort-indicator');
        if (!indicator) return;

        if (th.dataset.sort === sortColumn) {
            indicator.textContent = sortAscending ? '\u25B2' : '\u25BC';
        } else {
            indicator.textContent = '';
        }
    });
}

/**
 * Populate the community summary section with cards.
 * @param {Array} communities - Array of community objects from the API
 */
function populateCommunities(communities) {
    const countEl = document.getElementById('community-count');
    const cardsEl = document.getElementById('community-cards');
    if (!countEl || !cardsEl) return;

    countEl.textContent = communities.length + ' communities detected in the AOP-KE network';

    // Get the VHP4Safety palette for community colors
    const palette = [
        '#29235C', '#E6007E', '#307BBF', '#93D5F6', '#F39200',
        '#009FE3', '#951B81', '#00A59B', '#6E4C9E', '#007B77', '#8B70B8'
    ];

    let html = '';
    communities.forEach(function (comm) {
        const color = palette[comm.id % palette.length];
        const top5 = comm.members.slice(0, 5);
        const memberHtml = top5.map(function (m) {
            return '<strong>' + escapeHtml(m.type) + ':</strong> ' + escapeHtml(m.label);
        }).join('<br>');
        const moreText = comm.members.length > 5
            ? '<br><em>... and ' + (comm.members.length - 5) + ' more</em>'
            : '';

        html += '<div class="community-card" data-community-id="' + comm.id + '" style="border-left:4px solid ' + color + ';cursor:pointer;">' +
            '<h3>' +
            '<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:' + color + ';margin-right:8px;vertical-align:middle;"></span>' +
            'Community ' + comm.id +
            '</h3>' +
            '<div class="community-size">' + comm.size + ' members</div>' +
            '<div class="community-members">' + memberHtml + moreText + '</div>' +
            '</div>';
    });

    cardsEl.innerHTML = html;

    // Click a community card to filter the graph to that community
    cardsEl.querySelectorAll('.community-card').forEach(function (card) {
        card.addEventListener('click', function () {
            const commId = this.dataset.communityId;
            const communitySelect = document.getElementById('filter-community');
            if (communitySelect) {
                communitySelect.value = commId;
            }
            // Switch to Graph tab and apply filter
            activateTab('graph-view');
            setTimeout(function () {
                if (cy) {
                    cy.resize();
                    applyFilters(cy);
                }
            }, 100);
        });
    });
}


// ============================================================
// Section 7: Tab Switching
// ============================================================

/**
 * Set up tab navigation between Graph View and Metrics & Communities.
 */
function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');

    tabBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            const tabId = this.dataset.tab;
            activateTab(tabId);
        });
    });
}

/**
 * Activate a specific tab by ID.
 * @param {string} tabId - Tab content element ID ('graph-view' or 'metrics-view')
 */
function activateTab(tabId) {
    // Update button active states
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(function (btn) {
        if (btn.dataset.tab === tabId) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Update content visibility
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(function (content) {
        if (content.id === tabId) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });

    // Resize Cytoscape when switching to graph view
    if (tabId === 'graph-view' && cy) {
        setTimeout(function () {
            cy.resize();
        }, 50);
    }
}


// ============================================================
// Section 8: Export Handlers
// ============================================================

// Export buttons are plain <a> links in the HTML template pointing to
// /download/network/metrics and /download/network/graph respectively.
// No additional JavaScript needed since they trigger direct file downloads.
// This section is kept as a documented placeholder for any future
// export enhancements (e.g., client-side PNG export of the graph).


// ============================================================
// Section 9: Stats Bar Update
// ============================================================

/**
 * Update the stats bar with node, edge, and community counts.
 * @param {Object} stats - Stats object with nodes, edges, aop_count, ke_count, communities
 */
function updateStatsBar(stats) {
    if (!stats) return;

    const setVal = function (id, val) {
        const el = document.getElementById(id);
        if (el) el.textContent = val !== undefined ? val.toLocaleString() : '-';
    };

    setVal('stat-nodes', stats.nodes);
    setVal('stat-edges', stats.edges);
    setVal('stat-kes', stats.ke_count);
    setVal('stat-kers', stats.ker_count);
    setVal('stat-communities', stats.communities);
}


// ============================================================
// Utility Functions
// ============================================================

/**
 * Escape HTML special characters to prevent XSS.
 * @param {string} str - Raw string
 * @returns {string} HTML-safe string
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(String(str)));
    return div.innerHTML;
}

/**
 * Simple debounce utility.
 * @param {Function} fn - Function to debounce
 * @param {number} delay - Delay in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(fn, delay) {
    let timer = null;
    return function () {
        const context = this;
        const args = arguments;
        clearTimeout(timer);
        timer = setTimeout(function () {
            fn.apply(context, args);
        }, delay);
    };
}


// ============================================================
// Entry Point
// ============================================================

document.addEventListener('DOMContentLoaded', function () {
    setupTabs();
    initNetworkGraph();
});
