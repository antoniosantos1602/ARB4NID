(function () {
    let execChart = null;
    let asrChart  = null;
    let mahChart  = null;
    let asrMode   = 'classic';
    let mahSelectedModel  = null;
    let nkSelectedModel   = null;
    let cbrSelectedModel  = null;
    let execSelectedModel = null;
    let asrSelectedModel  = null;
    let cleanAccIndex     = 0;
    let robScoreIndex     = 0;
    let activeModels = new Set();
    const COLORS = ['#60a5fa', '#fbbf24', '#c084fc', '#34d399', '#fb923c'];
    const NK_COLORS = ['#334155', '#64748b', '#ca8a04', '#c2410c', '#7f1d1d'];
    const MAH_COLORS = { inside: '#475569', mild: '#ca8a04', severe: '#9f1239' };
    const CBR_COLORS = { binary: '#60a5fa', integer: '#fbbf24', dependency: '#c084fc', mahalanobis: '#34d399' };

    function allSections() {
        return ['step-1', 'results-section', 'dashboard-section'];
    }

    window.showDashboard = function () {
        allSections().forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        document.getElementById('dashboard-section').style.display = 'block';
        if (!execChart) initExecChart();
        if (!asrChart)  initAsrChart();
        if (!mahChart)  initMahChart();
        renderDashboard();
    };

    window.showResults = function () {
        allSections().forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        document.getElementById('results-section').style.display = 'block';
    };

    function initExecChart() {
        const ctx = document.getElementById('execTimeChart').getContext('2d');
        execChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#ccc', font: { family: 'Inter', size: 11 } }
                    },
                    tooltip: {
                        callbacks: { label: ctx => ` ${ctx.parsed.x.toFixed(2)}s` }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#888', callback: v => v + 's' },
                        grid: { color: '#2a2a4a' },
                        title: { display: true, text: 'Execution Time (s)', color: '#888', font: { size: 11 } }
                    },
                    y: {
                        ticks: { color: '#ccc', font: { size: 11 } },
                        grid: { color: '#1a1a2e' }
                    }
                }
            }
        });
    }

    function getResults() {
        return (window.allResults || []).filter(r =>
            activeModels.size === 0 || activeModels.has(r.model_name)
        );
    }

    function renderPills() {
        const allModels = [...new Set((window.allResults || []).map(r => r.model_name))];
        if (activeModels.size === 0) allModels.forEach(m => activeModels.add(m));

        const container = document.getElementById('dash-model-pills');
        container.innerHTML = allModels.map((m, i) => {
            const active = activeModels.has(m);
            const color = COLORS[i % COLORS.length];
            return `<button
                class="btn btn-sm"
                style="border:1px solid ${color};color:${active ? '#0d0d0d' : color};background:${active ? color : 'transparent'};font-size:0.72rem;border-radius:20px;"
                onclick="toggleModel('${m}')"
            >${m}</button>`;
        }).join('');
    }

    window.toggleModel = function (model) {
        if (activeModels.has(model)) {
            if (activeModels.size > 1) activeModels.delete(model);
        } else {
            activeModels.add(model);
        }
        renderDashboard();
    };

    function renderCleanAcc() {
        const byModel = {};
        (window.allResults || []).forEach(r => {
            if (!byModel[r.model_name]) byModel[r.model_name] = r.clean_accuracy;
        });

        const container = document.getElementById('dash-clean-acc');
        const entries = Object.entries(byModel);

        if (entries.length === 0) {
            container.innerHTML = '<div class="text-muted" style="font-size:0.8rem;">Waiting for results...</div>';
            return;
        }

        if (cleanAccIndex >= entries.length) cleanAccIndex = 0;
        const [model, acc] = entries[cleanAccIndex];
        const color = COLORS[cleanAccIndex % COLORS.length];
        const isMulti = entries.length > 1;

        const dotsHtml = isMulti ? `
            <div style="display:flex;gap:4px;justify-content:center;margin-top:10px;">
                ${entries.map((_, i) => `<span style="width:6px;height:6px;border-radius:50%;background:${i === cleanAccIndex ? color : '#444'};"></span>`).join('')}
            </div>
        ` : '';

        container.innerHTML = `
            <div style="width:100%;">
                <div style="display:flex;align-items:center;justify-content:center;gap:6px;">
                    ${isMulti ? `<button onclick="cleanAccPrev()" style="background:transparent;border:none;color:#888;font-size:1.8rem;cursor:pointer;padding:0 6px;line-height:1;">‹</button>` : ''}
                    <div>
                        <div style="font-family:'Orbitron',sans-serif;font-size:2.4rem;color:${color};">
                            ${typeof acc === 'number' ? acc.toFixed(5) : acc}%
                        </div>
                        <div style="font-size:0.68rem;color:#888;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;">Clean Accuracy</div>
                        <span style="font-size:0.68rem;background:#2a2a4a;color:#a0a0c0;border-radius:6px;padding:2px 8px;margin-top:8px;display:inline-block;">${model}</span>
                    </div>
                    ${isMulti ? `<button onclick="cleanAccNext()" style="background:transparent;border:none;color:#888;font-size:1.8rem;cursor:pointer;padding:0 6px;line-height:1;">›</button>` : ''}
                </div>
                ${dotsHtml}
            </div>
        `;
    }

    window.cleanAccPrev = function() {
        const models = [...new Set((window.allResults || []).map(r => r.model_name))];
        if (models.length <= 1) return;
        cleanAccIndex = (cleanAccIndex - 1 + models.length) % models.length;
        renderCleanAcc();
    };

    window.cleanAccNext = function() {
        const models = [...new Set((window.allResults || []).map(r => r.model_name))];
        if (models.length <= 1) return;
        cleanAccIndex = (cleanAccIndex + 1) % models.length;
        renderCleanAcc();
    };

    function renderRobustnessScore() {
        const raaEl = document.getElementById('dash-raa');
        const wcEl  = document.getElementById('dash-worst-case');
        if (!raaEl || !wcEl) return;

        const waiting = '<div class="text-muted" style="font-size:0.8rem;">Waiting for results...</div>';

        const byModel = {};
        (window.allResults || []).forEach(r => {
            if (!byModel[r.model_name]) byModel[r.model_name] = [];
            if (r.realistic_adversarial_accuracy != null && !isNaN(parseFloat(r.realistic_adversarial_accuracy))) {
                byModel[r.model_name].push(r);
            }
        });

        const entries = Object.entries(byModel).filter(([, arr]) => arr.length > 0);
        if (entries.length === 0) { raaEl.innerHTML = waiting; wcEl.innerHTML = waiting; return; }

        if (robScoreIndex >= entries.length) robScoreIndex = 0;
        const [model, modelResults] = entries[robScoreIndex];
        const color   = COLORS[robScoreIndex % COLORS.length];
        const isMulti = entries.length > 1;

        const accs   = modelResults.map(r => parseFloat(r.realistic_adversarial_accuracy));
        const raa    = accs.reduce((a, b) => a + b, 0) / accs.length;
        const minAcc = Math.min(...accs);
        const worst  = modelResults[accs.indexOf(minAcc)];

        const dotsHtml = isMulti ? `
            <div style="display:flex;gap:4px;justify-content:center;margin-top:10px;">
                ${entries.map((_, i) => `<span style="width:6px;height:6px;border-radius:50%;background:${i === robScoreIndex ? color : '#444'};"></span>`).join('')}
            </div>` : '';

        raaEl.innerHTML = `
            <div style="width:100%;">
                <div style="display:flex;align-items:center;justify-content:center;gap:6px;">
                    ${isMulti ? `<button onclick="robScorePrev()" style="background:transparent;border:none;color:#888;font-size:1.8rem;cursor:pointer;padding:0 6px;line-height:1;">‹</button>` : ''}
                    <div>
                        <div style="font-family:'Orbitron',sans-serif;font-size:2.2rem;color:#60a5fa;">${raa.toFixed(5)}%</div>
                        <div style="font-size:0.68rem;color:#888;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;">across ${accs.length} attack${accs.length > 1 ? 's' : ''}</div>
                        <span style="font-size:0.68rem;background:#2a2a4a;color:#a0a0c0;border-radius:6px;padding:2px 8px;margin-top:8px;display:inline-block;">${model}</span>
                    </div>
                    ${isMulti ? `<button onclick="robScoreNext()" style="background:transparent;border:none;color:#888;font-size:1.8rem;cursor:pointer;padding:0 6px;line-height:1;">›</button>` : ''}
                </div>
                ${dotsHtml}
            </div>`;

        wcEl.innerHTML = `
            <div style="width:100%;">
                <div style="display:flex;align-items:center;justify-content:center;gap:6px;">
                    ${isMulti ? `<button onclick="robScorePrev()" style="background:transparent;border:none;color:#888;font-size:1.8rem;cursor:pointer;padding:0 6px;line-height:1;">‹</button>` : ''}
                    <div>
                        <div style="font-family:'Orbitron',sans-serif;font-size:2.2rem;color:#f87171;">${minAcc.toFixed(5)}%</div>
                        <div style="font-size:0.68rem;color:#888;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;">Worst-Case — ${worst.attack_name}</div>
                        <span style="font-size:0.68rem;background:#2a2a4a;color:#a0a0c0;border-radius:6px;padding:2px 8px;margin-top:8px;display:inline-block;">${model}</span>
                    </div>
                    ${isMulti ? `<button onclick="robScoreNext()" style="background:transparent;border:none;color:#888;font-size:1.8rem;cursor:pointer;padding:0 6px;line-height:1;">›</button>` : ''}
                </div>
                ${dotsHtml}
            </div>`;
    }

    window.robScorePrev = function() {
        const models = [...new Set((window.allResults || []).map(r => r.model_name))];
        if (models.length <= 1) return;
        robScoreIndex = (robScoreIndex - 1 + models.length) % models.length;
        renderRobustnessScore();
    };

    window.robScoreNext = function() {
        const models = [...new Set((window.allResults || []).map(r => r.model_name))];
        if (models.length <= 1) return;
        robScoreIndex = (robScoreIndex + 1) % models.length;
        renderRobustnessScore();
    };

    function renderExecModelPills() {
        const models = [...new Set(getResults().map(r => r.model_name))];
        const wrapper = document.getElementById('exec-model-pills');

        if (models.length <= 1) {
            wrapper.style.display = 'none';
            execSelectedModel = models[0] || null;
            return;
        }

        wrapper.style.display = 'flex';
        if (!execSelectedModel || !models.includes(execSelectedModel)) {
            execSelectedModel = models[0];
        }

        wrapper.innerHTML = models.map((m, i) => {
            const color  = COLORS[i % COLORS.length];
            const active = m === execSelectedModel;
            return `<button
                class="btn btn-sm"
                style="border:1px solid ${color};color:${active ? '#0d0d0d' : color};background:${active ? color : 'transparent'};font-size:0.72rem;border-radius:20px;"
                onclick="setExecModel('${m}')"
            >${m}</button>`;
        }).join('');
    }

    window.setExecModel = function(model) {
        execSelectedModel = model;
        renderExecModelPills();
        renderExecChart();
    };

    function renderExecChart() {
        if (!execChart) return;
        let results = getResults();
        if (results.length === 0) return;

        const allModels = [...new Set(results.map(r => r.model_name))];
        if (allModels.length > 1 && execSelectedModel) {
            results = results.filter(r => r.model_name === execSelectedModel);
        }

        const attacks = [...new Set(results.map(r => r.attack_name))];
        const colorIdx = Math.max(0, allModels.indexOf(execSelectedModel));
        const color = COLORS[colorIdx % COLORS.length];

        const chartHeight = Math.max(300, attacks.length * 44);
        document.getElementById('exec-chart-wrapper').style.height = chartHeight + 'px';

        execChart.data.labels = attacks;
        execChart.data.datasets = [{
            label: execSelectedModel || 'Execution Time',
            data: attacks.map(atk => {
                const r = results.find(x => x.attack_name === atk);
                return r ? r.execution_time : null;
            }),
            backgroundColor: color + '99',
            borderColor: color,
            borderWidth: 1,
            borderRadius: 4,
        }];
        execChart.update('none');
    }

    function initAsrChart() {
        const ctx = document.getElementById('asrChart').getContext('2d');
        asrChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#ccc', font: { family: 'Inter', size: 11 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.x.toFixed(5)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        min: 0,
                        max: 100,
                        ticks: { color: '#888', callback: v => v + '%' },
                        grid: { color: '#2a2a4a' },
                        title: { display: true, text: 'Percentage (%)', color: '#888', font: { size: 11 } }
                    },
                    y: {
                        stacked: true,
                        ticks: { color: '#ccc', font: { size: 11 } },
                        grid: { color: '#1a1a2e' }
                    }
                }
            }
        });
    }

    window.setAsrMode = function(mode) {
        asrMode = mode;
        document.querySelectorAll('#asr-toggle button').forEach(btn => {
            btn.classList.toggle('active', btn.textContent.toLowerCase() === mode);
            btn.classList.toggle('btn-outline-info', btn.textContent.toLowerCase() === mode);
            btn.classList.toggle('btn-outline-secondary', btn.textContent.toLowerCase() !== mode);
        });
        renderAsrChart();
    };

    function renderAsrModelPills() {
        const models = [...new Set(getResults().map(r => r.model_name))];
        const wrapper = document.getElementById('asr-model-pills');

        if (models.length <= 1) {
            wrapper.style.display = 'none';
            asrSelectedModel = models[0] || null;
            return;
        }

        wrapper.style.display = 'flex';
        if (!asrSelectedModel || !models.includes(asrSelectedModel)) {
            asrSelectedModel = models[0];
        }

        wrapper.innerHTML = models.map((m, i) => {
            const color  = COLORS[i % COLORS.length];
            const active = m === asrSelectedModel;
            return `<button
                class="btn btn-sm"
                style="border:1px solid ${color};color:${active ? '#0d0d0d' : color};background:${active ? color : 'transparent'};font-size:0.72rem;border-radius:20px;"
                onclick="setAsrModel('${m}')"
            >${m}</button>`;
        }).join('');
    }

    window.setAsrModel = function(model) {
        asrSelectedModel = model;
        renderAsrModelPills();
        renderAsrChart();
    };

    function renderAsrChart() {
        if (!asrChart) return;
        let results = getResults();
        if (results.length === 0) return;

        const models = [...new Set(results.map(r => r.model_name))];
        if (models.length > 1 && asrSelectedModel) {
            results = results.filter(r => r.model_name === asrSelectedModel);
        }

        const labels   = results.map(r => r.attack_name);
        const datasets = [];

        if (asrMode === 'classic') {
            datasets.push({
                label: 'ASR',
                data: results.map(r => parseFloat(r.asr) || 0),
                backgroundColor: '#ef444499', borderColor: '#ef4444', borderWidth: 1, borderSkipped: false,
            });
            datasets.push({
                label: 'Adv. Accuracy',
                data: results.map(r => parseFloat(r.adversarial_accuracy) || parseFloat((100 - (r.asr||0)).toFixed(5))),
                backgroundColor: '#6b728033', borderColor: '#6b7280', borderWidth: 1, borderSkipped: false,
            });
        } else {
            datasets.push({
                label: 'ARS',
                data: results.map(r => parseFloat(((r.wasr_v2||0)*100).toFixed(5))),
                backgroundColor: '#f59e0b99', borderColor: '#f59e0b', borderWidth: 1, borderSkipped: false,
            });
            datasets.push({
                label: 'Real. Adv. Accuracy',
                data: results.map(r => parseFloat(r.realistic_adversarial_accuracy) || parseFloat((100-(r.wasr_v2||0)*100).toFixed(5))),
                backgroundColor: '#2dd4bf33', borderColor: '#2dd4bf', borderWidth: 1, borderSkipped: false,
            });
        }

        document.getElementById('asr-chart-wrapper').style.height = Math.max(300, labels.length * 36) + 'px';
        asrChart.data.labels   = labels;
        asrChart.data.datasets = datasets;
        asrChart.update('none');
    }

    function initMahChart() {
        const ctx = document.getElementById('mahChart').getContext('2d');
        mahChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#ccc', font: { family: 'Inter', size: 11 } } },
                    tooltip: {
                        callbacks: {
                            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(5)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#ccc', font: { size: 10 }, maxRotation: 45, minRotation: 45 },
                        grid: { color: '#1a1a2e' }
                    },
                    y: {
                        min: 0, max: 100,
                        ticks: { color: '#888', callback: v => v + '%' },
                        grid: { color: '#2a2a4a' },
                        title: { display: true, text: '% of successes', color: '#888', font: { size: 11 } }
                    }
                }
            }
        });
    }

    function renderNkModelPills() {
        const models = [...new Set(getResults().map(r => r.model_name))];
        const wrapper = document.getElementById('nk-model-pills');

        if (models.length <= 1) {
            wrapper.style.display = 'none';
            nkSelectedModel = models[0] || null;
            return;
        }

        wrapper.style.display = 'flex';
        if (!nkSelectedModel || !models.includes(nkSelectedModel)) {
            nkSelectedModel = models[0];
        }

        wrapper.innerHTML = models.map((m, i) => {
            const color  = COLORS[i % COLORS.length];
            const active = m === nkSelectedModel;
            return `<button
                class="btn btn-sm"
                style="border:1px solid ${color};color:${active ? '#0d0d0d' : color};background:${active ? color : 'transparent'};font-size:0.72rem;border-radius:20px;"
                onclick="setNkModel('${m}')"
            >${m}</button>`;
        }).join('');
    }

    window.setNkModel = function(model) {
        nkSelectedModel = model;
        renderNkModelPills();
        renderNkHeatmap();
    };

    function renderNkHeatmap() {
        let results = getResults();
        const container = document.getElementById('nk-heatmap');
        if (!container) return;

        if (results.length === 0) {
            container.innerHTML = '<div class="text-muted" style="font-size:0.8rem;">Waiting for results...</div>';
            return;
        }

        const models = [...new Set(results.map(r => r.model_name))];
        if (models.length > 1 && nkSelectedModel) {
            results = results.filter(r => r.model_name === nkSelectedModel);
        }

        const headerCell = label => `<div style="padding:8px 12px;color:#888;font-size:0.7rem;text-transform:uppercase;letter-spacing:.08em;text-align:center;">${label}</div>`;
        const rowLabelCell = label => `<div style="padding:8px 12px;color:#ccc;font-size:0.78rem;white-space:nowrap;display:flex;align-items:center;">${label}</div>`;

        const cell = (pct, baseColor) => {
            return `<div style="
                padding:10px 8px;
                background:${baseColor};
                color:#0d0d0d;
                text-align:center;
                font-size:0.82rem;
                font-weight:600;
                border-radius:4px;
            ">${pct.toFixed(5)}%</div>`;
        };

        const rowHeaderLabel = models.length > 1 ? 'Attack' : 'Attack — Model';
        let html = `<div style="display:grid;grid-template-columns:minmax(180px,auto) repeat(5, minmax(90px,1fr));gap:6px;min-width:600px;">`;
        html += headerCell(rowHeaderLabel);
        [0,1,2,3,4].forEach(k => html += headerCell(`N${k}`));

        results.forEach(r => {
            const nk = r.nk || {};
            const total = [0,1,2,3,4].reduce((acc, kk) => acc + (nk[kk] ?? nk[String(kk)] ?? 0), 0);
            const label = models.length > 1 ? r.attack_name : `${r.attack_name} — ${r.model_name}`;
            html += rowLabelCell(label);
            [0,1,2,3,4].forEach(k => {
                const count = nk[k] ?? nk[String(k)] ?? 0;
                const pct = total > 0 ? (count / total * 100) : 0;
                html += cell(pct, NK_COLORS[k]);
            });
        });
        html += `</div>`;

        html += `<div class="d-flex flex-wrap gap-3 mt-3" style="font-size:0.7rem;color:#888;">
            <span><span style="display:inline-block;width:10px;height:10px;background:${NK_COLORS[0]};border-radius:2px;margin-right:4px;"></span>N₀ clean</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:${NK_COLORS[1]};border-radius:2px;margin-right:4px;"></span>N₁</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:${NK_COLORS[2]};border-radius:2px;margin-right:4px;"></span>N₂</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:${NK_COLORS[3]};border-radius:2px;margin-right:4px;"></span>N₃</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:${NK_COLORS[4]};border-radius:2px;margin-right:4px;"></span>N₄ full violation</span>
        </div>`;

        container.innerHTML = html;
    }

    function renderCbrModelPills() {
        const models = [...new Set(getResults().map(r => r.model_name))];
        const wrapper = document.getElementById('cbr-model-pills');

        if (models.length <= 1) {
            wrapper.style.display = 'none';
            cbrSelectedModel = models[0] || null;
            return;
        }

        wrapper.style.display = 'flex';
        if (!cbrSelectedModel || !models.includes(cbrSelectedModel)) {
            cbrSelectedModel = models[0];
        }

        wrapper.innerHTML = models.map((m, i) => {
            const color  = COLORS[i % COLORS.length];
            const active = m === cbrSelectedModel;
            return `<button
                class="btn btn-sm"
                style="border:1px solid ${color};color:${active ? '#0d0d0d' : color};background:${active ? color : 'transparent'};font-size:0.72rem;border-radius:20px;"
                onclick="setCbrModel('${m}')"
            >${m}</button>`;
        }).join('');
    }

    window.setCbrModel = function(model) {
        cbrSelectedModel = model;
        renderCbrModelPills();
        renderCbrHeatmap();
    };

    function renderCbrHeatmap() {
        let results = getResults();
        const container = document.getElementById('cbr-heatmap');
        if (!container) return;

        if (results.length === 0) {
            container.innerHTML = '<div class="text-muted" style="font-size:0.8rem;">Waiting for results...</div>';
            return;
        }

        const models = [...new Set(results.map(r => r.model_name))];
        if (models.length > 1 && cbrSelectedModel) {
            results = results.filter(r => r.model_name === cbrSelectedModel);
        }

        const headerCell = label => `<div style="padding:8px 12px;color:#888;font-size:0.7rem;text-transform:uppercase;letter-spacing:.08em;text-align:center;">${label}</div>`;
        const rowLabelCell = label => `<div style="padding:8px 12px;color:#ccc;font-size:0.78rem;white-space:nowrap;display:flex;align-items:center;">${label}</div>`;

        const cell = (pct, baseColor) => `<div style="
            padding:10px 8px;
            background:${baseColor};
            color:#0d0d0d;
            text-align:center;
            font-size:0.82rem;
            font-weight:600;
            border-radius:4px;
        ">${pct.toFixed(5)}%</div>`;

        const types = ['binary', 'integer', 'dependency', 'mahalanobis'];
        const typeLabels = { binary: 'Binary', integer: 'Integer', dependency: 'Dependency', mahalanobis: 'Mahalanobis' };

        const rowHeaderLabel = models.length > 1 ? 'Attack' : 'Attack — Model';
        let html = `<div style="display:grid;grid-template-columns:minmax(180px,auto) repeat(4, minmax(110px,1fr));gap:6px;min-width:600px;">`;
        html += headerCell(rowHeaderLabel);
        types.forEach(t => html += headerCell(typeLabels[t]));

        results.forEach(r => {
            const cbr = r.constraint_breach_rate || {};
            const label = models.length > 1 ? r.attack_name : `${r.attack_name} — ${r.model_name}`;
            html += rowLabelCell(label);
            types.forEach(t => {
                const rate = cbr[t]?.rate;
                const pct = rate != null ? rate * 100 : 0;
                html += cell(pct, CBR_COLORS[t]);
            });
        });
        html += `</div>`;

        html += `<div class="d-flex flex-wrap gap-3 mt-3" style="font-size:0.7rem;color:#888;">
            <span><span style="display:inline-block;width:10px;height:10px;background:${CBR_COLORS.binary};border-radius:2px;margin-right:4px;"></span>Binary</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:${CBR_COLORS.integer};border-radius:2px;margin-right:4px;"></span>Integer</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:${CBR_COLORS.dependency};border-radius:2px;margin-right:4px;"></span>Dependency</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:${CBR_COLORS.mahalanobis};border-radius:2px;margin-right:4px;"></span>Mahalanobis</span>
        </div>`;

        container.innerHTML = html;
    }

    function renderMahModelPills() {
        const models = [...new Set(getResults().map(r => r.model_name))];
        const wrapper = document.getElementById('mah-model-pills');

        if (models.length <= 1) {
            wrapper.style.display = 'none';
            mahSelectedModel = models[0] || null;
            return;
        }

        wrapper.style.display = 'flex';
        if (!mahSelectedModel || !models.includes(mahSelectedModel)) {
            mahSelectedModel = models[0];
        }

        wrapper.innerHTML = models.map((m, i) => {
            const color  = COLORS[i % COLORS.length];
            const active = m === mahSelectedModel;
            return `<button
                class="btn btn-sm"
                style="border:1px solid ${color};color:${active ? '#0d0d0d' : color};background:${active ? color : 'transparent'};font-size:0.72rem;border-radius:20px;"
                onclick="setMahModel('${m}')"
            >${m}</button>`;
        }).join('');
    }

    window.setMahModel = function(model) {
        mahSelectedModel = model;
        renderMahModelPills();
        renderMahChart();
        renderMahPenTable();
    };

    function renderRawPenTable(containerId, dim, selectedModel) {
        const container = document.getElementById(containerId);
        if (!container) return;
        let results = getResults();
        if (results.length === 0) {
            container.innerHTML = '<div class="text-muted" style="font-size:0.8rem;">Waiting for results...</div>';
            return;
        }

        const models = [...new Set(results.map(r => r.model_name))];
        if (models.length > 1 && selectedModel) {
            results = results.filter(r => r.model_name === selectedModel);
        }

        const headerCell = label => `<div style="padding:6px 10px;color:#888;font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;text-align:center;">${label}</div>`;
        const rowLabel   = label => `<div style="padding:8px 10px;color:#ccc;font-size:0.75rem;white-space:nowrap;display:flex;align-items:center;">${label}</div>`;

        let html = `<div style="display:grid;grid-template-columns:minmax(150px,auto) repeat(4,minmax(80px,1fr));gap:5px;min-width:420px;">`;
        html += headerCell('Attack');
        html += headerCell('p25');
        html += headerCell('Median');
        html += headerCell('Mean');
        html += headerCell('p75');

        results.forEach((r, i) => {
            const rowColor = COLORS[i % COLORS.length];
            const pp = (r.penalty_percentiles || {})[dim] || {};
            const lbl = models.length > 1 ? r.attack_name : `${r.attack_name} — ${r.model_name}`;

            const valueCell = (val) => {
                const v = 1 - (val ?? 0);
                const intensity = Math.min(v, 1);
                const bg = hexWithAlpha(rowColor, 0.15 + intensity * 0.75);
                const textColor = intensity > 0.5 ? '#0d0d0d' : '#e0e0e0';
                return `<div style="padding:8px 6px;background:${bg};color:${textColor};text-align:center;font-size:0.78rem;font-weight:600;border-radius:4px;">${v.toFixed(4)}</div>`;
            };

            html += rowLabel(lbl);
            html += valueCell(pp.p25);
            html += valueCell(pp.p50);
            html += valueCell(pp.mean);
            html += valueCell(pp.p75);
        });

        html += `</div>`;
        container.innerHTML = html;
    }

    function renderMahPenTable() {
        renderRawPenTable('mahpen-table', 'mahalanobis', mahSelectedModel);
    }

    function renderMahChart() {
        if (!mahChart) return;
        let results = getResults();
        if (results.length === 0) return;

        const models = [...new Set(results.map(r => r.model_name))];
        if (models.length > 1 && mahSelectedModel) {
            results = results.filter(r => r.model_name === mahSelectedModel);
        }

        const labels = results.map(r => r.attack_name);
        const zones  = ['inside', 'mild', 'severe'];
        const labelMap = { inside: 'Inside (r≤1)', mild: 'Mild (1<r≤2)', severe: 'Severe (r>2)' };

        const datasets = zones.map(z => ({
            label: labelMap[z],
            data: results.map(r => {
                const mz = r.mahalanobis_zones || {};
                const rate = mz[z]?.rate;
                return rate != null ? parseFloat((rate * 100).toFixed(5)) : 0;
            }),
            backgroundColor: MAH_COLORS[z] + 'cc',
            borderColor: MAH_COLORS[z],
            borderWidth: 1,
            borderRadius: 4,
            borderSkipped: false,
        }));

        const minWidth = Math.max(500, labels.length * 130);
        const canvas = document.getElementById('mahChart');
        canvas.parentElement.style.width = minWidth + 'px';

        mahChart.data.labels = labels;
        mahChart.data.datasets = datasets;
        mahChart.update('none');
    }

    let binpenSelectedModel = null;
    let intpenSelectedModel = null;

    function makePenaltyModelPills(wrapperId, currentModel, setFn) {
        const models = [...new Set(getResults().map(r => r.model_name))];
        const wrapper = document.getElementById(wrapperId);
        if (models.length <= 1) {
            wrapper.style.display = 'none';
            return models[0] || null;
        }
        wrapper.style.display = 'flex';
        let selected = currentModel;
        if (!selected || !models.includes(selected)) selected = models[0];
        wrapper.innerHTML = models.map((m, i) => {
            const color  = COLORS[i % COLORS.length];
            const active = m === selected;
            return `<button class="btn btn-sm"
                style="border:1px solid ${color};color:${active ? '#0d0d0d' : color};background:${active ? color : 'transparent'};font-size:0.72rem;border-radius:20px;"
                onclick="${setFn}('${m}')">${m}</button>`;
        }).join('');
        return selected;
    }

    function renderPenaltyTable(containerId, dim, totalKey) {
        const container = document.getElementById(containerId);
        if (!container) return;
        let results = getResults();
        if (results.length === 0) {
            container.innerHTML = '<div class="text-muted" style="font-size:0.8rem;">Waiting for results...</div>';
            return;
        }

        const models = [...new Set(results.map(r => r.model_name))];
        const selectedModel = dim === 'binary' ? binpenSelectedModel : intpenSelectedModel;
        if (models.length > 1 && selectedModel) {
            results = results.filter(r => r.model_name === selectedModel);
        }

        const nTotal = results[0]?.[totalKey] || 0;

        const headerCell = label => `<div style="padding:6px 10px;color:#888;font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;text-align:center;">${label}</div>`;
        const rowLabel   = label => `<div style="padding:8px 10px;color:#ccc;font-size:0.75rem;white-space:nowrap;display:flex;align-items:center;">${label}</div>`;

        let html = `<div style="display:grid;grid-template-columns:minmax(150px,auto) repeat(4,minmax(80px,1fr));gap:5px;min-width:420px;">`;
        html += headerCell('Attack');
        html += headerCell('p25');
        html += headerCell('Median');
        html += headerCell('Mean');
        html += headerCell('p75');

        results.forEach((r, i) => {
            const rowColor = COLORS[i % COLORS.length];
            const valueCell = (frac) => {
                const rep   = 1 - frac;
                const pct   = rep * 100;
                const count = nTotal > 0 ? (rep * nTotal) : 0;
                const intensity = Math.min(rep, 1);
                const bg = hexWithAlpha(rowColor, 0.15 + intensity * 0.75);
                const textColor = intensity > 0.5 ? '#0d0d0d' : '#e0e0e0';
                const tooltip = nTotal > 0 ? `≈${count.toFixed(2)} of ${nTotal} features respected` : '';
                return `<div title="${tooltip}" style="padding:8px 6px;background:${bg};color:${textColor};text-align:center;font-size:0.78rem;font-weight:600;border-radius:4px;cursor:default;">
                    ${rep.toFixed(4)}
                </div>`;
            };
            const pp  = (r.penalty_percentiles || {})[dim] || {};
            const lbl = models.length > 1 ? r.attack_name : `${r.attack_name} — ${r.model_name}`;
            html += rowLabel(lbl);
            html += valueCell(pp.p25  ?? 0);
            html += valueCell(pp.p50  ?? 0);
            html += valueCell(pp.mean ?? 0);
            html += valueCell(pp.p75  ?? 0);
        });
        html += `</div>`;
        container.innerHTML = html;
    }

    function hexWithAlpha(hex, alpha) {
        const r = parseInt(hex.slice(1,3),16);
        const g = parseInt(hex.slice(3,5),16);
        const b = parseInt(hex.slice(5,7),16);
        return `rgba(${r},${g},${b},${alpha.toFixed(2)})`;
    }

    window.setBinpenModel = function(model) {
        binpenSelectedModel = model;
        binpenSelectedModel = makePenaltyModelPills('binpen-model-pills', binpenSelectedModel, 'setBinpenModel');
        renderPenaltyTable('binpen-table', 'binary', 'n_binary_cols');
    };

    window.setIntpenModel = function(model) {
        intpenSelectedModel = model;
        intpenSelectedModel = makePenaltyModelPills('intpen-model-pills', intpenSelectedModel, 'setIntpenModel');
        renderPenaltyTable('intpen-table', 'integer', 'n_integer_cols');
    };

    function renderBinpenTable() {
        binpenSelectedModel = makePenaltyModelPills('binpen-model-pills', binpenSelectedModel, 'setBinpenModel');
        renderPenaltyTable('binpen-table', 'binary', 'n_binary_cols');
    }

    function renderIntpenTable() {
        intpenSelectedModel = makePenaltyModelPills('intpen-model-pills', intpenSelectedModel, 'setIntpenModel');
        renderPenaltyTable('intpen-table', 'integer', 'n_integer_cols');
    }

    let deppenSelectedModel = null;

    window.setDeppenModel = function(model) {
        deppenSelectedModel = model;
        deppenSelectedModel = makePenaltyModelPills('deppen-model-pills', deppenSelectedModel, 'setDeppenModel');
        renderDeppenTable();
    };

    function renderDeppenTable() {
        const container = document.getElementById('deppen-table');
        const deppenCol = document.getElementById('deppen-col');
        const mahCol    = document.getElementById('mah-col');
        if (!container) return;

        let results = getResults();
        if (results.length === 0) {
            if (deppenCol) deppenCol.style.display = 'none';
            return;
        }

        const hasDep = results.some(r => (r.penalty_percentiles || {})['dependency']?.p50 != null);
        if (!hasDep) {
            if (deppenCol) deppenCol.style.display = 'none';
            if (mahCol)    mahCol.className = 'col-lg-12';
            return;
        }
        if (deppenCol) deppenCol.style.display = 'block';
        if (mahCol)    mahCol.className = 'col-lg-6';

        deppenSelectedModel = makePenaltyModelPills('deppen-model-pills', deppenSelectedModel, 'setDeppenModel');
        const models = [...new Set(results.map(r => r.model_name))];
        if (models.length > 1 && deppenSelectedModel) {
            results = results.filter(r => r.model_name === deppenSelectedModel);
        }

        const headerCell = label => `<div style="padding:6px 10px;color:#888;font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;text-align:center;">${label}</div>`;
        const rowLabel   = label => `<div style="padding:8px 10px;color:#ccc;font-size:0.75rem;white-space:nowrap;display:flex;align-items:center;">${label}</div>`;
        const valueCell  = (val, rowColor) => {
            const rep = 1 - (val ?? 0);
            const pct = rep * 100;
            const intensity = Math.min(rep, 1);
            const bg = hexWithAlpha(rowColor, 0.15 + intensity * 0.75);
            const textColor = intensity > 0.5 ? '#0d0d0d' : '#e0e0e0';
            return `<div style="padding:8px 6px;background:${bg};color:${textColor};text-align:center;font-size:0.78rem;font-weight:600;border-radius:4px;">${pct.toFixed(5)}%</div>`;
        };

        let html = `<div style="display:grid;grid-template-columns:minmax(150px,auto) repeat(4,minmax(80px,1fr));gap:5px;min-width:420px;">`;
        html += headerCell('Attack');
        html += headerCell('p25');
        html += headerCell('Median');
        html += headerCell('Mean');
        html += headerCell('p75');

        results.forEach((r, i) => {
            const rowColor = COLORS[i % COLORS.length];
            const pp  = (r.penalty_percentiles || {})['dependency'] || {};
            const lbl = models.length > 1 ? r.attack_name : `${r.attack_name} — ${r.model_name}`;
            html += rowLabel(lbl);
            html += valueCell(pp.p25,  rowColor);
            html += valueCell(pp.p50,  rowColor);
            html += valueCell(pp.mean, rowColor);
            html += valueCell(pp.p75,  rowColor);
        });

        html += `</div>`;
        container.innerHTML = html;
    }

    function renderDashboard() {
        renderPills();
        renderCleanAcc();
        renderRobustnessScore();
        renderExecModelPills();
        renderExecChart();
        renderNkModelPills();
        renderNkHeatmap();
        renderCbrModelPills();
        renderCbrHeatmap();
        renderBinpenTable();
        renderIntpenTable();
        renderDeppenTable();
        renderMahModelPills();
        renderMahChart();
        renderMahPenTable();
        renderAsrModelPills();
        renderAsrChart();
    }

    window.updateDashboard = function () {
        if (document.getElementById('dashboard-section').style.display !== 'none') {
            renderDashboard();
        }
    };
})();
