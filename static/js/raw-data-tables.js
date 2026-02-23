/**
 * Raw Data Tables for AOP-Wiki RDF Dashboard
 *
 * Provides toggle-able raw data table views beneath plots.
 * Fetches data on-demand from /api/plot-data/<plot_name> and renders
 * static HTML tables. Truncates at 100 rows with a message.
 */

async function toggleDataTable(plotName) {
    const container = document.querySelector('[data-table-for="' + plotName + '"]');
    if (!container) return;

    const content = container.querySelector('.data-table-content');
    const button = container.querySelector('.data-table-toggle');

    if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'block';
        button.textContent = 'Hide Raw Data';

        if (!content.dataset.loaded) {
            content.innerHTML = '<div class="data-table-loading">Loading data...</div>';
            try {
                // Get version from URL params or version selector
                var params = new URLSearchParams(window.location.search);
                var version = params.get('version') || '';
                var url = '/api/plot-data/' + plotName + (version ? '?version=' + encodeURIComponent(version) : '');
                var response = await fetch(url);
                var data = await response.json();

                if (!data.success) {
                    content.innerHTML = '<div class="data-table-empty">Data not available for this plot.</div>';
                    return;
                }

                var html = buildDataTable(data.columns, data.rows);
                if (data.truncated) {
                    html += '<div class="data-table-truncated">Showing first ' + data.rows.length + ' of ' + data.total_rows + ' rows. Download CSV for full data.</div>';
                }
                content.innerHTML = html;
                content.dataset.loaded = 'true';
            } catch (err) {
                content.innerHTML = '<div class="data-table-empty">Failed to load data.</div>';
            }
        }
    } else {
        content.style.display = 'none';
        button.textContent = 'Show Raw Data';
    }
}

function buildDataTable(columns, rows) {
    var html = '<div class="data-table-wrapper"><table class="raw-data-table"><thead><tr>';
    columns.forEach(function(col) {
        html += '<th>' + escapeHtml(String(col)) + '</th>';
    });
    html += '</tr></thead><tbody>';
    rows.forEach(function(row) {
        html += '<tr>';
        columns.forEach(function(col) {
            var val = row[col] != null ? String(row[col]) : '';
            html += '<td>' + escapeHtml(val) + '</td>';
        });
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    return html;
}

function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Reset loaded state when version changes (so data tables refetch)
document.addEventListener('version-changed', function() {
    document.querySelectorAll('.data-table-content').forEach(function(el) {
        el.dataset.loaded = '';
        el.style.display = 'none';
    });
    document.querySelectorAll('.data-table-toggle').forEach(function(btn) {
        btn.textContent = 'Show Raw Data';
    });
});
