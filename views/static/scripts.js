function updateFileName(inputId, spanId) {
  const input = document.getElementById(inputId);
  const span = document.getElementById(spanId);
  const files = input.files;

  if (files.length === 0) {
    span.textContent = "Nenhum ficheiro selecionado";
  } else if (files.length === 1) {
    span.textContent = files[0].name;
  } else {
    const fileNames = Array.from(files).map(file => file.name);
    span.innerHTML = fileNames.map(name => `<div>${name}</div>`).join('');
  }
}

setTimeout(() => {
  const alerts = document.querySelectorAll('.alert');
  alerts.forEach(alert => alert.classList.remove('show'));
}, 5000);

const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
const tooltipList = [...tooltipTriggerList].map(el => new bootstrap.Tooltip(el));

document.addEventListener('DOMContentLoaded', () => {
  const wrapper = document.getElementById('combination-patterns-wrapper');
  const addBtn = document.getElementById('add-combination-pattern');

  if (!wrapper || !addBtn) {
    console.error("A2PM wrapper/button not found");
    return;
  }

  let patternCounter = 0;

  addBtn.addEventListener('click', () => {
    const patternId = patternCounter++;
    const div = document.createElement('div');
    div.classList.add('combination-pattern', 'border', 'border-secondary', 'rounded', 'p-3', 'mb-3', 'bg-dark');
    div.dataset.patternId = patternId;

    div.innerHTML = `
      <label class="form-label text-info">Combination Features</label>
      <div class="input-group mb-2">
        <input type="text" class="form-control" id="comb_feat_${patternId}"
              name="combination_features_${patternId}" placeholder="Select binary features...">
        <button class="btn btn-outline-info dropdown-toggle" type="button"
                data-bs-toggle="dropdown" data-bs-auto-close="outside" aria-expanded="false">
          Choose
        </button>
        <div class="dropdown-menu dropdown-menu-dark p-3 shadow-lg" style="width: 300px; max-height: 300px; overflow-y: auto;">
          <input type="text" class="form-control form-control-sm mb-2" placeholder="Search..." onkeyup="filterFeatures(this)">
          <div class="binary-items-container"></div>
        </div>
      </div>

      <label class="form-label text-warning">Locked Features</label>
      <div class="input-group mb-2">
        <input type="text" class="form-control" id="locked_feat_${patternId}"
              name="locked_features_${patternId}" placeholder="Select binary features to lock...">
        <button class="btn btn-outline-info dropdown-toggle" type="button"
                data-bs-toggle="dropdown" data-bs-auto-close="outside" aria-expanded="false">
          Choose
        </button>
        <div class="dropdown-menu dropdown-menu-dark p-3 shadow-lg" style="width: 300px; max-height: 300px; overflow-y: auto;">
          <input type="text" class="form-control form-control-sm mb-2" placeholder="Search..." onkeyup="filterFeatures(this)">
          <div class="locked-binary-container"></div>
        </div>
      </div>

      <div class="row">
        <div class="col-md-6">
          <label class="form-label">Probability</label>
          <input type="number" class="form-control mb-2" name="combination_probability_${patternId}"
                min="0.01" max="1" step="0.01" value="0.5">
        </div>
      </div>

      <button type="button" class="btn btn-sm btn-danger remove-pattern mt-2">
        <i class="bi bi-trash"></i> Remove Pattern
      </button>
    `;

    wrapper.appendChild(div);

    const container = div.querySelector('.binary-items-container');
    const binCols = currentDatasetInfo.binary_columns || [];
    const lockedContainer = div.querySelector('.locked-binary-container');

    renderFeatureButtons(lockedContainer, binCols, `locked_feat_${patternId}`);
    renderFeatureButtons(container, binCols, `comb_feat_${patternId}`);
  });

  wrapper.addEventListener('click', e => {
    if (e.target.closest('.remove-pattern')) {
      e.target.closest('.combination-pattern').remove();
    }
  });
});

function toggleArtParamGroup() {
  const attackInput = document.getElementById("attack_type_hidden");
  if (!attackInput || !attackInput.value) return;

  const attack = attackInput.value.toLowerCase();
  const allGroups = document.querySelectorAll(".attack-param-group");

  allGroups.forEach(group => group.style.display = "none");

  const target = document.getElementById(`params-${attack}`);
  if (target) {
    target.style.display = "block";
  }
}

document.addEventListener("DOMContentLoaded", function () {
  toggleArtParamGroup();
});


document.addEventListener('DOMContentLoaded', function () {
  const attackTypeInput = document.getElementById('attack_type_hidden');
  if (!attackTypeInput) return;

  const artTab = document.getElementById('tab-art');
  if (artTab) {
    artTab.addEventListener('shown.bs.tab', function () {
      attackTypeInput.value = 'ART';
      toggleArtParamGroup();
    });
  }

  // Default seguro
  attackTypeInput.value = 'ART';
  toggleArtParamGroup();
});



function getJsmaParam(paramName) {
  const preset = document.querySelector(`input[name="${paramName}"]:checked`);
  const manual = document.querySelector(`input[name="${paramName}_manual"]`);
  return manual && manual.value !== "" ? manual.value : (preset ? preset.value : null);
}

const theta = getJsmaParam("jsma_theta");
const gamma = getJsmaParam("jsma_gamma");
const batchSize = getJsmaParam("jsma_batch_size");

function toggleCollapseAndParams(checkbox, collapseId) {
  const collapseEl = document.getElementById(collapseId);
  const bsCollapse = new bootstrap.Collapse(collapseEl, { toggle: false });

  if (checkbox.checked) {
    bsCollapse.show();
  } else {
    bsCollapse.hide();
  }
}

document.addEventListener("DOMContentLoaded", () => {

  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {

      const targetId = btn.dataset.target;
      const value = btn.dataset.value;
      const input = document.getElementById(targetId);

      const group = btn.parentElement.querySelectorAll(".preset-btn");

      if (btn.classList.contains("active")) {
        btn.classList.remove("active");
        input.value = "";
        return;
      }

      group.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      input.value = value;
    });
  });

  document.querySelectorAll("input[type='number']").forEach(input => {
    input.addEventListener("input", () => {
      const btns = document.querySelectorAll(
        `.preset-btn[data-target="${input.id}"]`
      );
      btns.forEach(b => b.classList.remove("active"));
    });
  });

});

// Variável para guardar colunas detetadas (útil para expansões futuras)
let datasetColumns = [];



function StepBack() {
  document.getElementById('step-1').style.display = 'block';
  document.getElementById('step-2').style.display = 'none';
}


function populateTargetClasses(classList) {
    // Procura todos os seletores com esta classe específica
    const selectors = document.querySelectorAll('.target-class-selector');

    selectors.forEach(select => {
        select.innerHTML = '';
        classList.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c;
            opt.textContent = (c == "0") ? `Class ${c} (Benign)` : `Class ${c}`;
            select.appendChild(opt);
        });
    });
}

function toggleTargetClassView(selectElement, wrapperId) {
    const wrapper = document.getElementById(wrapperId);
    if (!wrapper) return;

    if (selectElement.value === "True") {
        wrapper.style.display = 'block';
        wrapper.classList.add('animated', 'fadeIn');
    } else {
        wrapper.style.display = 'none';
    }
}

// Variável para evitar re-analisar o mesmo ficheiro desnecessariamente
let currentDatasetInfo = { name: "", size: 0, classes: [] };

async function nextStep() {
    const testFileInput = document.getElementById('testFile');
    const testFile = testFileInput?.files?.[0];


    if (!testFile) {
        alert("Please select a Test Dataset first.");
        return;
    }

    // 1. Otimização: Se for o mesmo ficheiro, não pergunta ao servidor de novo
    if (currentDatasetInfo.name === testFile.name && currentDatasetInfo.size === testFile.size) {
        document.getElementById('step-1').style.display = 'none';
        document.getElementById('step-2').style.display = 'block';
        return;
    }

    // 2. Estado de Carregamento (Feedback Visual)
    const nextBtn = document.querySelector("button[onclick='nextStep()']");
    const originalBtnText = nextBtn.innerHTML;
    nextBtn.disabled = true;
    nextBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Analyzing...';

    // 3. Preparar o envio assíncrono
    const formData = new FormData();
    formData.append('file', testFile);

    try {
        const response = await fetch('/api/analyze-dataset', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.classes) {

            // 1. Mostrar o painel do contrato (estava escondido)
            document.getElementById('domain-constraints-contract').style.display = 'block';

            // 2. Injetar os valores recebidos do servidor nos IDs do HTML
            document.getElementById('modal-bin-count').innerText = data.bin_count || 0;
            document.getElementById('modal-int-count').innerText = data.integer_count || 0;

            // 3. Preencher lista de Binárias
            const binContainer = document.getElementById('modal-bin-list');
            binContainer.innerHTML = ''; // Limpar anterior
            if (data.binary_columns && data.binary_columns.length > 0) {
                data.binary_columns.forEach(col => {
                    binContainer.innerHTML += `<span class="badge bg-outline-primary border text-primary fw-normal" style="font-family: 'Inter', sans-serif;">${col}</span>`;
                });
            } else {
                binContainer.innerHTML = '<span class="text-muted small italic">No binary features detected.</span>';
            }

            // 4. Preencher lista de Inteiras
            const intContainer = document.getElementById('modal-int-list');
            intContainer.innerHTML = ''; // Limpar anterior
            if (data.integer_columns && data.integer_columns.length > 0) {
                data.integer_columns.forEach(col => {
                    intContainer.innerHTML += `<span class="badge bg-outline-secondary border text-dark fw-normal">${col}</span>`;
                });
            } else {
                intContainer.innerHTML = '<span class="text-muted small italic">No integer features detected.</span>';
            }

            // 1. Unidirecionais (A -> B)
            const uniList = document.getElementById('list-unidirecional');
            const uniDeps = data.unidirectional_dependencies || [];
            uniList.innerHTML = uniDeps.length === 0 ? '<div class="text-muted small p-2 italic">No rules detected.</div>' : '';
            uniDeps.forEach(dep => {
                uniList.innerHTML += `
                    <div class="d-flex align-items-center gap-2 mb-1 pb-1 border-bottom border-light">
                        <span class="text-primary fw-bold small">${dep[0]}</span>
                        <i class="bi bi-arrow-right text-muted mx-1"></i>
                        <span class="text-dark small">${dep[1]}</span>
                    </div>`;
            });

            // 3. EXCLUSIVIDADE (A != B)
            const excList = document.getElementById('list-exclusividade');
            const excDeps = data.mutual_exclusivity_dependencies || [];
            excList.innerHTML = excDeps.length === 0 ? '<div class="text-muted small p-2 italic">No rules detected.</div>' : '';
            excDeps.forEach(dep => {
                excList.innerHTML += `
                    <div class="d-flex align-items-center gap-2 mb-1 pb-1 border-bottom border-light">
                        <span class="text-primary fw-bold small">${dep[0]}</span>
                        <span class="text-muted mx-1">≠</span>
                        <span class="text-dark small">${dep[1]}</span>
                    </div>`;
            });

            // 4. LINEARES (Target = A + B)
            const linList = document.getElementById('list-linear');
            const linDeps = data.linear_dependencies || [];
            linList.innerHTML = linDeps.length === 0 ? '<div class="text-muted small p-2 italic">No rules detected.</div>' : '';
            linDeps.forEach(dep => {
                linList.innerHTML += `
                    <div class="d-flex align-items-center gap-2 mb-1 pb-1 border-bottom border-light">
                        <span class="text-primary fw-bold small">${dep[0]}</span>
                        <span class="text-muted mx-1">=</span>
                        <span class="text-dark small">${dep[1].join(' + ')}</span>
                    </div>`;
            });
            //classes
            populateTargetClasses(data.classes);

            //A2PM
            populateA2PMFeatureLists(data);

            // Guardamos info para evitar repetições
            currentDatasetInfo = { name: testFile.name, size: testFile.size, classes: data.classes, binary_columns: data.binary_columns};

            // Avançar na interface
            document.getElementById('step-1').style.display = 'none';
            document.getElementById('step-2').style.display = 'block';
            window.scrollTo(0, 0);
        } else {
            alert("Error: " + (data.error || "Could not analyze dataset labels."));
        }
    } catch (error) {
        console.error("Analysis failed:", error);
        alert("Network error while analyzing dataset.");
    } finally {
        // Restaurar o botão
        nextBtn.disabled = false;
        nextBtn.innerHTML = originalBtnText;
    }
}

/////////////////////// A2PM BUTTONS

function populateA2PMFeatureLists(data) {
    // 1. Preencher a lista Geral (Interval Features)
    renderFeatureButtons(
        document.getElementById('container_features_int'),
        data.continuous_cols || [],
        'features_int_input'
    );

    // 2. Preencher a lista Numérica (Integer Features)
    renderFeatureButtons(
        document.getElementById('container_integer_features'),
        data.integer_columns || [],
        'integer_features_input'
    );
}

function renderFeatureButtons(container, columns, targetInputId) {
    if (!container) return;
    container.innerHTML = '';
    const input = document.getElementById(targetInputId);

    // --- NOVO: Adicionar botões "Select All" e "Clear All" ---
    const controls = document.createElement('div');
    controls.className = 'd-flex justify-content-between mb-2 pb-2 border-bottom border-secondary';
    controls.innerHTML = `
        <button type="button" class="btn btn-xxs btn-outline-info w-50 me-1 py-0" style="font-size: 0.7rem;">Select All</button>
        <button type="button" class="btn btn-xxs btn-outline-secondary w-50 ms-1 py-0" style="font-size: 0.7rem;">Clear All</button>
    `;

    // Lógica do Select All
    controls.querySelector('.btn-outline-info').onclick = (e) => {
        e.stopPropagation();
        // Seleciona apenas as que não estão escondidas pelo filtro
        const visibleItems = container.querySelectorAll('.feature-item:not([style*="display: none"])');
        let currentVal = input.value.trim();
        let currentItems = currentVal ? currentVal.split(',').map(x => x.trim()) : [];

        visibleItems.forEach(item => {
            const colName = item.dataset.name;
            if (!currentItems.includes(colName)) {
                currentItems.push(colName);
                item.classList.add('bg-info', 'text-dark');
                item.querySelector('i').classList.replace('d-none', 'd-block');
            }
        });
        input.value = currentItems.join(',');
    };

    // Lógica do Clear All
    controls.querySelector('.btn-outline-secondary').onclick = (e) => {
        e.stopPropagation();
        const visibleItems = container.querySelectorAll('.feature-item:not([style*="display: none"])');
        let currentVal = input.value.trim();
        let currentItems = currentVal ? currentVal.split(',').map(x => x.trim()) : [];

        visibleItems.forEach(item => {
            const colName = item.dataset.name;
            currentItems = currentItems.filter(i => i !== colName);
            item.classList.remove('bg-info', 'text-dark');
            item.querySelector('i').classList.replace('d-block', 'd-none');
        });
        input.value = currentItems.join(',');
    };

    container.appendChild(controls);

    // --- Renderização das colunas ---
    columns.forEach((colName, index) => {
        const item = document.createElement('div');
        item.className = 'dropdown-item d-flex align-items-center justify-content-between py-2 px-3 rounded feature-item mb-1';
        item.style.cursor = 'pointer';
        item.dataset.name = colName; // Guardamos o nome num data-attribute para facilitar o acesso

        let currentVal = input.value.trim();
        let items = currentVal ? currentVal.split(',').map(x => x.trim()) : [];
        let isSelected = items.includes(colName);

        if (isSelected) item.classList.add('bg-info', 'text-dark');

        item.innerHTML = `
            <span><strong>${colName}</strong> <small class="${isSelected ? 'text-dark' : 'text-muted'}">(${index})</small></span>
            <i class="bi bi-check-circle-fill ${isSelected ? 'd-block' : 'd-none'}"></i>
        `;

        item.onclick = function(e) {
            e.stopPropagation();
            let val = input.value.trim();
            let currentItems = val ? val.split(',').map(x => x.trim()) : [];
            const icon = item.querySelector('i');

            if (currentItems.includes(colName)) {
                currentItems = currentItems.filter(i => i !== colName);
                item.classList.remove('bg-info', 'text-dark');
                icon.classList.replace('d-block', 'd-none');
            } else {
                currentItems.push(colName);
                item.classList.add('bg-info', 'text-dark');
                icon.classList.replace('d-none', 'd-block');
            }
            input.value = currentItems.join(',');
        };
        container.appendChild(item);
    });
}

// Função para filtrar a lista no Dropdown (Barra de Procura)
function filterFeatures(inputEl) {
    const filter = inputEl.value.toLowerCase();
    const container = inputEl.nextElementSibling; // O div que contém os itens
    const items = container.getElementsByClassName('feature-item');

    for (let i = 0; i < items.length; i++) {
        const text = items[i].textContent || items[i].innerText;
        items[i].style.setProperty('display', text.toLowerCase().includes(filter) ? 'flex' : 'none', 'important');
    }
}

// Função chamada quando o ficheiro JSON muda (onchange)
function updateFileNameDisplay(input) {
    const badge = document.getElementById('file_name_badge');
    const clearBtn = document.getElementById('btn_clear_json');
    const uploadBtnIcon = document.querySelector('.bi-upload').parentElement;

    if (input.files && input.files.length > 0) {
        const fileName = input.files[0].name;

        // 1. Mostrar o nome do ficheiro no badge azul
        badge.innerText = fileName;
        badge.style.display = 'inline-block';

        // 2. Mostrar o botão de remover (X vermelho)
        clearBtn.style.display = 'inline-block';

        // 3. Mudar a cor do botão de upload para indicar estado ativo
        uploadBtnIcon.classList.remove('btn-outline-info');
        uploadBtnIcon.classList.add('btn-info');

        // Criar um Toast informativo (opcional, se tiveres a função showToast)
        if (typeof showToast === "function") {
            showToast(`Configuração carregada: ${fileName}`, "info");
        }
    }
}


function clearJsonInput() {
    const input = document.getElementById('attack_config_file');
    const badge = document.getElementById('file_name_badge');
    const clearBtn = document.getElementById('btn_clear_json');
    const uploadBtnIcon = document.querySelector('.bi-upload').parentElement;

    // 1. Reset Total do valor
    input.value = "";

    // 2. Reset Visual
    badge.style.display = 'none';
    badge.innerText = "";
    clearBtn.style.display = 'none';

    // 3. Reset do botão de upload
    uploadBtnIcon.classList.remove('btn-info');
    uploadBtnIcon.classList.add('btn-outline-info');

    // 4. Notificação de sucesso (Usa a função do results.stream se estiver disponível)
    // Se showNotification não for global, você pode usar um alert simples ou console
    console.log("JSON Cleared");
}

// Dispara um download da forma MENOS disruptiva possível para não abortar
// o stream SSE dos ataques em curso. Nem window.location (navegação) nem
// <a download href="/api/..."> (carregamento de recurso do servidor) são
// seguros — ambos podem cancelar o fetch do SSE no browser. Aqui buscamos
// o ficheiro com um fetch() normal (que não interfere com outros fetch) e
// só depois o entregamos via um URL local blob: — sem qualquer navegação
// nem pedido de recurso do servidor no elemento <a>.
window.triggerDownload = async function(url) {
    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error('HTTP ' + resp.status);

        // Nome do ficheiro a partir do Content-Disposition (fallback: fim do URL)
        const cd = resp.headers.get('Content-Disposition') || '';
        let filename = decodeURIComponent(url.split('/').pop().split('?')[0]) || 'download';
        const m = cd.match(/filename\*?=(?:UTF-8'')?"?([^;"]+)"?/i);
        if (m) filename = decodeURIComponent(m[1].replace(/"/g, ''));

        const blob = await resp.blob();
        const objUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = objUrl;
        a.download = filename;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(objUrl), 2000);
    } catch (err) {
        console.error('Download falhou:', err);
        if (typeof showNotification === 'function') {
            showNotification('Download falhou: ' + err.message, 'danger', 6000);
        }
    }
};

function downloadResultsJSON() {
    window.triggerDownload('/api/results/export-json');
}


window.downloadFullReport = function(attackId) {
    if (!attackId) {
        alert("Attack ID is required for export.");
        return;
    }

    const url = `/api/export/full-report/${encodeURIComponent(attackId)}`;

    if (typeof showNotification === "function") {
        showNotification("Generating consolidated ZIP report... Please wait.", "info");
    }

    window.triggerDownload(url);
};

window.downloadModelReport = function(modelName) {
    if (!modelName) {
        alert("Model name is required for export.");
        return;
    }

    const url = `/api/export/model-report/${encodeURIComponent(modelName)}`;

    if (typeof showNotification === "function") {
        showNotification("Generating model report... Please wait.", "info");
    }

    window.triggerDownload(url);
};


function showResults() {
    document.getElementById('step-2').style.display = 'none';
    document.getElementById('step-3').style.display = 'block';
}

function backToConfig() {
    document.getElementById('step-3').style.display = 'none';
    document.getElementById('step-2').style.display = 'block';
}

function toggleAllAttacks(isChecked, containerId) {
    const scope = containerId ? document.getElementById(containerId) : document;
    if (!scope) return;
    const checkboxes = scope.querySelectorAll('input[name="selected_attacks"], input[name="selected_attacks_targeted"]');

    checkboxes.forEach(checkbox => {
        checkbox.checked = isChecked;
    });

    const status = isChecked ? "selected" : "deselected";
    console.log(`All attacks ${status}${containerId ? ' in ' + containerId : ''}`);
}