document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('attackForm');
  const btn = document.getElementById('executeAttackBtn');
  const resultsContainer = document.getElementById('attack-results');

  window.deleteAllResults = async function(event) {
      if (event) event.preventDefault();
      if (!confirm("Remove all results?")) return;

      try {
          const response = await fetch('/api/results/clear-all', { method: 'DELETE' });
          if (response.ok) {
              const ul = document.getElementById('resultsList');
              if (ul) ul.innerHTML = '';
              const header = document.getElementById('resultsHeader');
              if (header) header.remove();

              document.getElementById('clear-all-container').style.display = 'none';
              showNotification("All results clear", "success");
          }
      } catch (error) { console.error("Erro:", error); }
  };

  window.deleteResult = async function(attackId, button) {
      try {
          const response = await fetch(`/api/results/${attackId}`, { method: 'DELETE' });
          if (response.ok) {
              button.closest('.result-card').remove();
              // Se não houver mais cards, esconde o botão de apagar tudo
              if (document.querySelectorAll('.result-card').length === 0) {
                  document.getElementById('clear-all-container').style.display = 'none';
              }
          }
      } catch (error) { console.error(error); }
  };

  function showNotification(message, type = "danger", timeout = 4000) {
    const container = document.getElementById("toast-container");

    const toast = document.createElement("div");
    toast.className = `alert alert-${type} alert-dismissible fade show shadow`;
    toast.role = "alert";

    let prefix = "Error";
    if (type === "info") prefix = "Info";
    if (type === "success") prefix = "Success";

    toast.innerHTML = `
      <strong>${prefix}:</strong> ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.remove("show");
      toast.classList.add("hide");
      setTimeout(() => toast.remove(), 500);
    }, timeout);
  }

  const jsonInput = document.getElementById('attack_config_file');
  if (jsonInput) {
    jsonInput.addEventListener('change', function(e) {
      if (this.files.length > 0) {
        showNotification("JSON Config detected. Manual parameters will be ignored.", "info", 3000);
        if (typeof updateFileNameDisplay === "function") {
            updateFileNameDisplay(this);
        }
      }
    });
  }


  function validateAttackForm(form, resultsContainer) {
    //const trainInput = document.getElementById("trainFile");
    const testInput  = document.getElementById("testFile");
    const modelInput = document.getElementById("modelFile");
    const selectedAttacks = document.querySelectorAll(
      'input[name="selected_attacks"]:checked'
    );

    const jsonConfigInput = document.getElementById("attack_config_file");
    const hasJsonConfig = jsonConfigInput && jsonConfigInput.files.length > 0;


    let errors = [];


    if (
      //!trainInput || trainInput.files.length === 0 ||
      !testInput  || testInput.files.length === 0 ||
      !modelInput || modelInput.files.length === 0
    ) {
      errors.push("Please provide the test dataset and at least one model.");
    }

    if (!hasJsonConfig && selectedAttacks.length === 0) {
      errors.push("Please select at least one adversarial attack.");
    }

    if (errors.length > 0) {
      errors.forEach(err => showNotification(err, "danger"));
      return false;
    }

    return true;
  }


  window.showConfig = function() {
    document.getElementById('attackForm').style.display = 'block';
    document.getElementById('results-section').style.display = 'none';
  };

  function showResults() {
    document.getElementById('attackForm').style.display = 'none';
    document.getElementById('results-section').style.display = 'block';
  }

  function getRowId(attack, model) {
    return 'row-' + (attack + '-' + model).replace(/[^a-zA-Z0-9]/g, '_');
  }

  function getOrCreateModelGroup(modelName, container) {
    const groupId = 'group-' + modelName.replace(/[^a-zA-Z0-9]/g, '_');
    let cardsEl = document.getElementById(groupId + '-cards');
    if (!cardsEl) {
      const wrapper = document.createElement('div');
      wrapper.id = groupId;
      wrapper.className = 'model-group';
      wrapper.innerHTML = `
        <div class="model-group-title d-flex align-items-center justify-content-between">
          <span>${modelName}</span>
        </div>
        <div class="model-group-cards" id="${groupId}-cards"></div>
      `;
      container.appendChild(wrapper);
      cardsEl = document.getElementById(groupId + '-cards');
    }
    return cardsEl;
  }

  function addModelReportButtons() {
    document.querySelectorAll('.model-group').forEach(group => {
      const header = group.querySelector('.model-group-title');
      if (!header || header.querySelector('.model-report-btn')) return;
      const modelName = header.querySelector('span').textContent.trim();
      const btn = document.createElement('button');
      btn.className = 'btn btn-sm btn-outline-info model-report-btn';
      btn.style.cssText = 'font-size:0.7rem;padding:2px 10px;white-space:nowrap;';
      btn.textContent = 'Model Report';
      btn.onclick = () => downloadModelReport(modelName);
      header.appendChild(btn);
    });
  }

  async function executeAttack() {

    if(!validateAttackForm(form, resultsContainer)){
      return;
    }

    window.allResults = [];

    showResults();

    const container = document.getElementById('attack-results');
    container.innerHTML = '<div id="resultsList"></div>';

    const formElement = document.getElementById('attackForm');
    const patternsInDOM = document.querySelectorAll('.combination-pattern').length;
    console.log(`📊 Patterns visible in DOM: ${patternsInDOM}`);

    const formData = new FormData(formElement);
    console.log("📤 FormData entries:");
    for (let [key, value] of formData.entries()) {
        if (key.includes('combination')) {
            console.log(`${key}: ${value}`);
        }
    }

    const combFeats = formData.getAll("combination_features[]");
    combFeats.forEach((feat, i) => {
        console.log(`   [${i}] ${feat.substring(0, 50)}...`);
    });
    // Disable button
    btn.disabled = true;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Processing...';

    let spinnerBox = document.getElementById("attack-spinner");
    if (!spinnerBox) {
        spinnerBox = document.createElement("div");
        spinnerBox.id = "attack-spinner";
        resultsContainer.prepend(spinnerBox);
    }


    formData.set("submit_type", "attack");

    console.log("=== A2PM Form Data ===");
    for (let [key, value] of formData.entries()) {
        if (key.includes('combination_features_') ||
            key.includes('locked_features_') ||
            key.includes('combination_probability_')) {
            console.log(`${key}: ${value}`);
        }
    }

    const jsonInput = document.getElementById('attack_config_file');
    if (!jsonInput || !jsonInput.value) {
        formData.delete('attack_config_file');
    }

    try {
      const resp = await fetch('/api/upload', { method: 'POST', body: formData });
      if (!resp.ok) {
        const errTxt = await resp.text();
        resultsContainer.insertAdjacentHTML(
          "beforeend",
          `<div class="alert alert-danger mt-3">Erro: ${errTxt}</div>`
        );
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const ul = document.getElementById('resultsList');

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const chunk of parts) {
          const lines = chunk.split("\n").map(l => l.trim());
          const eventLine = lines.find(l => l.startsWith("event:"));
          const dataLine  = lines.find(l => l.startsWith("data:"));
          if (!dataLine) continue;

          const eventType = eventLine ? eventLine.replace(/^event:\s*/, '') : 'result';
          const dataStr = dataLine.replace(/^data:\s*/, '');
          if (!dataStr || dataStr === "{}" || dataStr === "END") continue;

          let data;
          try { data = JSON.parse(dataStr); }
          catch (e) {
            console.error("Falha ao fazer parse de um evento SSE (ignorado):", e, dataStr.slice(0, 300));
            continue;
          }

          // --- Evento skipped: mostra aviso sem spinner ---
          if (eventType === 'skipped') {
            showNotification(`[${data.attack}] ${data.model}: ${data.reason}`, 'warning', 6000);
            continue;
          }

          // --- Evento running: cria linha com spinner agrupada por modelo ---
          if (eventType === 'running') {
            const rowId = getRowId(data.attack, data.model);
            const card = document.createElement('div');
            card.id = rowId;
            card.className = 'result-card d-flex align-items-center gap-3';
            card.innerHTML = `
              <span class="spinner-border spinner-border-sm text-info"></span>
              <span style="font-size:1rem;">[${data.attack}]</span>
            `;
            const group = getOrCreateModelGroup(data.model, ul);
            group.appendChild(card);
            continue;
          }

          if (data.message) {
              console.log(data.message);
              continue;
          }

          const clearContainer = document.getElementById('clear-all-container');
          if (clearContainer) clearContainer.style.display = 'block';

          if (data.is_notification) {
              const msg = `Attack ${data.attack} skipped on ${data.model}: ${data.error}`;
              showNotification(msg, data.type || "warning", 6000);
              const spinnerEl = document.getElementById(getRowId(data.attack, data.model));
              if (spinnerEl) spinnerEl.remove();
              continue;
          }

          if (data.error) {
            const card = document.createElement('li');
            card.className = 'result-card';
            card.textContent = `Error: ${data.error}`;
            ul.appendChild(card);
            continue;
          }



          if (!window.allResults) window.allResults = [];
          window.allResults.push(data);
          if (window.updateDashboard) window.updateDashboard(data);

          const name = data.attack_name || '(sem nome)';
          const model = data.model_name || '';
          const params = data.params || {};
          const cleanAccuracy = data.clean_accuracy;
          const cleanDetection = data.clean_detection || { tp: 0, total: 0 };
          const detectionRate = cleanDetection.rate;
          const executionTime = data.execution_time;
          const adv = data.adv_acc_by_k_cumulativo || {};
          const framework = data.framework || '';

          const isTargeted = data.targeted;
          const asr = data.asr;
          const advAccGlobal = data.adversarial_accuracy;


          let advHtml = '';
          const cumul = {};
          for (let k = 0; k <= 4; k++) {
            const obj = adv[String(k)] || null;
            cumul[k] = {
              acc: obj && obj.acc != null ? obj.acc : null,
              asr: obj && obj.asr != null ? obj.asr : null,
              total: obj && obj.total != null ? Number(obj.total) : 0,
              corretos: obj && obj.corretos != null ? Number(obj.corretos) : 0,
              enganados: obj && obj.enganados != null ? Number(obj.enganados) : 0,
              desvios: obj && obj.desvios != null ? Number(obj.desvios) : 0,
              desvios_classes: obj && obj.desvios_classes ? obj.desvios_classes : {}
            };
          }
          const exact = {};
          let prev = { total: 0, corretos: 0, enganados: 0, desvios: 0 };
          for (let k = 0; k <= 4; k++) {
            exact[k] = {
              total: Math.max(0, cumul[k].total - prev.total),
              corretos: Math.max(0, cumul[k].corretos - prev.corretos),
              enganados: Math.max(0, cumul[k].enganados - prev.enganados),
              desvios: Math.max(0, (cumul[k].desvios || 0) - (prev.desvios || 0)),
              desvios_classes: cumul[k].desvios_classes || {},
              acc_cumul: cumul[k].acc,
              asr_cumul: cumul[k].asr
            };
            prev = cumul[k];
          }

          // Serializa payload completo para o botão Confusion Matrix (modal global)
          const fullDataAttr = JSON.stringify(data).replace(/"/g, "&quot;");

          for (let k = 0; k <= 4; k++) {
            const c = cumul[k];
            const accDisplay = (c.acc == null || c.total === 0) ? 'N/A' : (Number(c.acc).toFixed(4) + '%');
            advHtml += `
              <li style="font-size: 1rem; line-height: 2;">
                <span class="text-info">Adversarial Accuracy (k ≤ ${k})</span>: <strong>${accDisplay}</strong>
              </li>
            `;
          }



          let paramsHtml = "";
          if (Object.keys(params).length > 0) {
              paramsHtml = Object.entries(params)
                  .map(([key, val]) => `<span class="badge bg-dark text-info border border-info me-1" style="font-size: 0.7rem;">${key}: ${val}</span>`)
                  .join("");
          }

          // --- ASR display ---
          const asrDisplay = asr != null
              ? (isTargeted ? `${asr.toFixed(5)}% (→ class ${data.target_class})` : `${asr.toFixed(5)}%`)
              : 'N/A';

          // --- ARS (WASR V2) ---
          const wasrV2     = data.wasr_v2;
          const wasrV2Display = wasrV2 != null ? `${(wasrV2 * 100).toFixed(5)}%` : 'N/A';
          const advAccDisplay  = data.adversarial_accuracy != null ? `${data.adversarial_accuracy.toFixed(5)}%` : 'N/A';
          const realAdvAcc     = data.realistic_adversarial_accuracy;
          const realAdvAccDisplay = realAdvAcc != null ? `${realAdvAcc.toFixed(5)}%` : 'N/A';
          const gap = (asr != null && wasrV2 != null) ? (asr - wasrV2 * 100).toFixed(5) : null;
          const gapDisplay = gap != null ? `${gap}%` : 'N/A';

          // --- Median Reputation (overall_realism.p50) ---
          const repRealism = (data.penalty_percentiles || {}).overall_realism || {};
          const medRep = repRealism.p50;
          const medRepPct = medRep != null ? Math.round(medRep * 100) : null;

          // --- N0–N4 ---
          const n0       = data.n0;
          const nK0      = data.n_k0_success;
          const nSuccess = data.n_success;
          const n0Display = n0 != null
              ? `${(n0 * 100).toFixed(5)}% <span class="text-muted small">(${nK0 ?? 0} / ${nSuccess ?? 0})</span>`
              : 'N/A';

          // --- Nk table — realism médio e contribuição ao WASR ---
          const nkDetail = data.nk_detail || {};
          const wasr_v2_val = data.wasr_v2 ?? 0;
          const nkRows = [0,1,2,3,4].map(k => {
              const d = nkDetail[String(k)] || nkDetail[k] || {};
              const n_k = d.n ?? 0;
              if (n_k === 0) return '';
              const pctSuccess = nSuccess > 0 ? ((n_k / nSuccess) * 100).toFixed(5) : '0.0';
              const realism = d.realism_medio != null ? d.realism_medio.toFixed(4) : '—';
              const contrib = d.contribution != null ? (d.contribution * 100).toFixed(5) : '0.00';
              return `<tr>
                  <td class="text-muted pe-2">N<sub>${k}</sub> <small>(${k} violação${k !== 1 ? 'ões' : ''})</small></td>
                  <td class="text-center">${n_k} <span class="text-muted small">(${pctSuccess}%)</span></td>
                  <td class="text-center">${realism}</td>
                  <td class="text-center text-info"><strong>${contrib}%</strong></td>
              </tr>`;
          }).join('');
          const nkHtml = nkRows ? `
              <div class="mt-3 pt-2 border-top border-secondary">
                  <small class="text-white d-block mb-2">WASR Decomposition by Constraint Group</small>
                  <table class="table table-sm table-borderless text-white mb-0" style="font-size:0.82rem;">
                      <thead>
                          <tr>
                              <th class="text-muted fw-normal">Group</th>
                              <th class="text-muted fw-normal text-center">Attacks</th>
                              <th class="text-muted fw-normal text-center">Avg Realism</th>
                              <th class="text-muted fw-normal text-center">WASR contrib.</th>
                          </tr>
                      </thead>
                      <tbody>${nkRows}</tbody>
                      <tfoot>
                          <tr style="border-top:1px solid #334155;">
                              <td colspan="3" class="text-muted pt-1">Total WASR</td>
                              <td class="text-center text-warning pt-1"><strong>${(wasr_v2_val * 100).toFixed(5)}%</strong></td>
                          </tr>
                      </tfoot>
                  </table>
              </div>` : '';

          // --- Constraint Breach Rate ---
          const cbr = data.constraint_breach_rate || {};
          const cbrDimensions = [
              { key: 'binary',      label: 'Binary' },
              { key: 'integer',     label: 'Integer' },
              { key: 'dependency',  label: 'Dependency' },
              { key: 'mahalanobis', label: 'Mahalanobis' },
          ];
          const cbrRows = cbrDimensions.map(({ key, label }) => {
              const d = cbr[key];
              if (!d) return '';
              const pct = (d.rate * 100).toFixed(5);
              const barWidth = Math.round(d.rate * 100);
              return `<tr>
                  <td class="text-muted pe-2" style="white-space:nowrap;">${label}</td>
                  <td style="width:100px;">
                      <div style="background:#334155;border-radius:3px;height:8px;">
                          <div style="width:${barWidth}%;background:#38bdf8;border-radius:3px;height:8px;"></div>
                      </div>
                  </td>
                  <td class="ps-2"><strong>${pct}%</strong> <span class="text-muted small">(${d.count} / ${nSuccess ?? 0})</span></td>
              </tr>`;
          }).join('');
          const cbrHtml = cbrRows ? `
              <div class="mt-3 pt-2 border-top border-secondary">
                  <small class="text-white d-block mb-1">Constraint Breach Rate — of successful attacks</small>
                  <table class="table table-sm table-borderless text-white mb-0" style="font-size:0.85rem;">
                      <tbody>${cbrRows}</tbody>
                  </table>
              </div>` : '';

          // --- Percentis P25/P50/P75 por dimensão — sobre ataques bem-sucedidos ---
          const penPerc = data.penalty_percentiles || {};
          const percDimensions = [
              { key: 'binary',      label: 'Binary' },
              { key: 'integer',     label: 'Integer' },
              { key: 'dependency',  label: 'Dependency' },
              { key: 'mahalanobis', label: 'Mahalanobis' },
          ];
          const percRows = percDimensions.map(({ key, label }) => {
              const d = penPerc[key];
              if (!d) return '';
              const fmt = v => v != null ? (v * 100).toFixed(5) + '%' : 'N/A';
              return `<tr>
                  <td class="text-muted pe-2">${label}</td>
                  <td class="text-center">${fmt(d.p25)}</td>
                  <td class="text-center"><strong>${fmt(d.p50)}</strong></td>
                  <td class="text-center">${fmt(d.p75)}</td>
              </tr>`;
          }).join('');
          const overallRealism = penPerc["overall_realism"];
          const overallRow = overallRealism ? (() => {
              const fmt = v => v != null ? (v * 100).toFixed(5) + '%' : 'N/A';
              return `<tr style="border-top:1px solid #334155;">
                  <td class="text-white fw-semibold pe-2">Overall Realism</td>
                  <td class="text-center text-info">${fmt(overallRealism.p25)}</td>
                  <td class="text-center text-info"><strong>${fmt(overallRealism.p50)}</strong></td>
                  <td class="text-center text-info">${fmt(overallRealism.p75)}</td>
              </tr>`;
          })() : '';
          const percTableHtml = percRows ? `
              <div class="mt-3 pt-2 border-top border-secondary">
                  <small class="text-white d-block mb-2">Penalty Distribution — Successful Attacks</small>
                  <table class="table table-sm table-borderless text-white mb-0" style="font-size:0.82rem;">
                      <thead>
                          <tr>
                              <th class="text-muted fw-normal">Constraint</th>
                              <th class="text-muted fw-normal text-center">P25</th>
                              <th class="text-muted fw-normal text-center">P50</th>
                              <th class="text-muted fw-normal text-center">P75</th>
                          </tr>
                      </thead>
                      <tbody>${percRows}${overallRow}</tbody>
                  </table>
              </div>` : '';

          // --- Mahalanobis Zone Distribution ---
          const mahZones = data.mahalanobis_zones || {};
          const mahZoneHtml = Object.keys(mahZones).length ? (() => {
              const zones = [
                  { key: 'inside', label: 'Inside distribution', desc: 'r ≤ 1', color: '#2ecc71' },
                  { key: 'mild',   label: 'Mild outlier',        desc: '1 < r ≤ 2', color: '#f39c12' },
                  { key: 'severe', label: 'Severe outlier',      desc: 'r > 2', color: '#e74c3c' },
              ];
              const rows = zones.map(z => {
                  const d = mahZones[z.key] || {};
                  const pct = ((d.rate || 0) * 100).toFixed(5);
                  const barWidth = Math.round((d.rate || 0) * 100);
                  return `<tr>
                      <td style="white-space:nowrap;">
                          <span style="color:${z.color};">●</span>
                          <span class="text-muted ms-1">${z.label}</span>
                          <small class="text-muted ms-1">(${z.desc})</small>
                      </td>
                      <td style="width:90px;">
                          <div style="background:#334155;border-radius:3px;height:8px;">
                              <div style="width:${barWidth}%;background:${z.color};border-radius:3px;height:8px;"></div>
                          </div>
                      </td>
                      <td class="ps-2"><strong>${pct}%</strong> <span class="text-muted small">(${d.count ?? 0})</span></td>
                  </tr>`;
              }).join('');
              return `<div class="mt-3 pt-2 border-top border-secondary">
                  <small class="text-white d-block mb-1">Mahalanobis Zone Distribution — Successful Attacks</small>
                  <table class="table table-sm table-borderless text-white mb-0" style="font-size:0.82rem;">
                      <tbody>${rows}</tbody>
                  </table>
              </div>`;
          })() : '';

          const li = document.createElement('li');
          li.className = 'result-card';

          li.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div style="flex:1;">
                    <div class="result-card-title">
                        ${name}
                        ${model ? `<small class="text-light">— ${model}</small>` : ''}
                    </div>
                    <div class="mb-3">${paramsHtml}</div>
                    <div style="font-size:0.82rem;line-height:1.75;margin-top:0.35rem;">
                        Clean Accuracy: <strong>${cleanAccuracy ?? 'N/A'}%</strong><br>
                        Execution Time: <strong>${executionTime ?? 'N/A'}s</strong><br>
                        Attack Success Rate (ASR): <strong>${asrDisplay}</strong><br>
                        Adversarial Accuracy (AA): <strong>${advAccDisplay}</strong><br>
                        Attack Reputation Score (ARS): <strong>${wasrV2Display}</strong><br>
                        Realistic Adversarial Accuracy: <strong>${realAdvAccDisplay}</strong>
                    </div>
                    <div class="mt-3">
                        <button class="btn btn-sm btn-outline-info"
                            type="button"
                            data-full="${fullDataAttr}"
                            onclick="openGlobalCMModal(this)">
                            Confusion Matrix
                        </button>
                    </div>
                </div>
                <div class="d-flex flex-column gap-2 ms-3">
                  <button type="button"
                      class="btn btn-sm btn-success shadow-sm"
                      onclick="downloadFullReport('${data.attack_id}')">
                    Full Report
                  </button>
                  <button type="button"
                      class="btn btn-sm btn-outline-success shadow-sm"
                      onclick="window.triggerDownload('/api/export/csv/${encodeURIComponent(data.attack_id)}')">
                    Download CSV
                  </button>
                  <button type="button"
                          class="btn btn-sm btn-danger"
                          onclick="deleteResult('${data.attack_id}', this)">
                      Delete
                  </button>
                </div>
            </div>
          `;


          // substituir spinner pelo card, ou append no grupo correto
          const rowId = getRowId(data.attack_name || '', data.model_name || '');
          const existing = document.getElementById(rowId);
          if (existing) {
            existing.replaceWith(li);
          } else {
            const group = getOrCreateModelGroup(data.model_name || '', ul);
            group.appendChild(li);
          }
        }
      }


    } catch (error) {
      resultsContainer.insertAdjacentHTML(
        "beforeend",
        `<div class="alert alert-danger mt-3">Erro inesperado: ${error.message}</div>`
      );
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalText;
      const spinnerBox = document.getElementById("attack-spinner");
      if (spinnerBox) spinnerBox.remove();
      addModelReportButtons();
    }
  }

  btn.addEventListener('click', e => {
    e.preventDefault();
    executeAttack();
  });

  form.addEventListener('submit', e => {
    e.preventDefault();
    executeAttack();
  });

});

  async function deleteAllResults() {
      if (!confirm("Remove all results?")) return;

      try {
          const response = await fetch('/api/results/clear-all', {
              method: 'DELETE'
          });

          if (response.ok) {
              // Remove todos os cards da interface imediatamente
              const ul = document.getElementById('resultsList');
              if (ul) ul.innerHTML = '';

              const clearContainer = document.getElementById('clear-all-container');
              if (clearContainer) clearContainer.style.display = 'none';

          } else {
              alert("Error");
          }
      } catch (error) {
          alert("Error Server");
      }
  }