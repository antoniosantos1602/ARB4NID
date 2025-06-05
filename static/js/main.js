let lastResults = [];
let currentModelName = ''; // Podes mais tarde preencher com o nome real do modelo
let currentMode = '';


document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const attackList = document.getElementById('attack-list');
    const attackSelection = document.getElementById('attack-selection');
    const resultsDiv = document.getElementById('results');

    // Para manter os ataques em ordem com os parâmetros
    window.orderedAttacks = [];

    window.handleUpload = async (mode) => {
        currentMode = mode;
        const trainFile = document.querySelector("input[name='train']").files[0];
        const testFile = document.querySelector("input[name='test']").files[0];
        const modelFile = document.querySelector("input[name='model']").files[0];

        if (!trainFile || !testFile || !modelFile) {
            alert("Por favor seleciona todos os ficheiros.");
            return;
        }

        const formData = new FormData();
        formData.append("train", trainFile);
        formData.append("test", testFile);
        formData.append("model", modelFile);

        const endpoint = mode === 'evasion' ? '/upload_evasion' : '/upload_poisoning';

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            alert("Erro ao carregar ficheiros!");
            return;
        }

        const result = await response.text();
        alert(result);

        // Agora só busca os ataques se o upload correu bem
        const responseAttacks = await fetch(`/get_attacks?mode=${mode}`);
        if (!responseAttacks.ok) {
            alert("Erro ao buscar os ataques!");
            return;
        }
        const attacksAvailable = await responseAttacks.json();

        renderAttackSelection(attacksAvailable);

    };

    function renderAttackSelection(attacks) {
        attackList.innerHTML = '';
        window.orderedAttacks = [];

        attackSelection.style.display = 'block';
        document.getElementById("run-button").style.display = "inline-block";

        // Botão para adicionar ataques
        const addBtn = document.createElement("button");
        addBtn.textContent = "➕ Adicionar ataque";
        addBtn.type = "button";
        addBtn.onclick = () => addAttackBlock(attacks);
        attackList.appendChild(addBtn);
    }

    function addAttackBlock(attacks) {
    const idx = window.orderedAttacks.length;
    window.orderedAttacks.push(null); // reservamos o slot

    const div = document.createElement('div');
    div.className = 'attack-block';

    div.innerHTML = `
        <label>${idx + 1}.
            <select onchange="updateOrderedAttack(${idx}, this)">
                <option value="">-- Escolhe um ataque --</option>
                ${attacks.map(a => `<option value="${a}">${a}</option>`).join('')}
            </select>
        </label>
        <div id="params-${idx}" class="params"></div>
    `;

    attackList.appendChild(div);
    }



    window.updateOrderedAttack = (idx, selectElem) => {
        const attackName = selectElem.value;
        window.orderedAttacks[idx] = {
            name: attackName,
            params: {}
        };

        const paramsDiv = document.getElementById(`params-${idx}`);
        paramsDiv.innerHTML = '';

        if (attackName === 'FGSM') {
            paramsDiv.innerHTML = `
                <label>Epsilon: <input type="number" step="0.01" onchange="updateParam(${idx}, 'epsilon', this.value)" /></label>
                <label>Eps Step: <input type="number" step="0.01" onchange="updateParam(${idx}, 'eps_step', this.value)" /></label>
                <label>Batch Size:</label>
                <div id="batch-options-${idx}">
                    ${[64, 128].map(size => `
                        <button type="button" onclick="setBatchSize(${idx}, ${size}, this)">${size}</button>
                    `).join(' ')}
                </div>
                <label>Norma:</label>
                <div id="norm-options-${idx}" class="param-buttons">
                    ${[1, 2, 'inf'].map(norm => `
                        <button type="button" onclick="setButtonParam(${idx}, 'norm', '${norm}', this)">${norm}</button>
                    `).join(' ')}
                </div>
                `;
        } else if (attackName === 'JSMA') {
            paramsDiv.innerHTML = `
                <label>Theta: <input type="number" step="0.01" onchange="updateParam(${idx}, 'theta', this.value)" /></label>
                <label>Gamma: <input type="number" step="0.01" onchange="updateParam(${idx}, 'gamma', this.value)" /></label>
                <label>Batch Size:</label>
                <div id="batch-options-${idx}">
                    ${[64, 128].map(size => `
                        <button type="button" onclick="setBatchSize(${idx}, ${size}, this)">${size}</button>
                    `).join(' ')}
                </div>
            `;
        } else if (attackName === 'ZOO') {
            paramsDiv.innerHTML = `
                <label>Max Iter: <input type="number" onchange="updateParam(${idx}, 'max_iter', this.value)" /></label>
                <label>Learning Rate: <input type="number" step="0.01" onchange="updateParam(${idx}, 'learning_rate', this.value)" /></label>
                <label>Binary Search Steps: <input type="number" onchange="updateParam(${idx}, 'binary_search_steps', this.value)" /></label>
                <label>Initial Const: <input type="number" step="0.001" onchange="updateParam(${idx}, 'initial_const', this.value)" /></label>
                <label>Batch Size:</label>
                <div id="batch-options-${idx}">
                    ${[64, 128].map(size => `
                        <button type="button" onclick="setBatchSize(${idx}, ${size}, this)">${size}</button>
                    `).join(' ')}
                </div>
            `;
        }

        console.log("Ataques selecionados:", window.orderedAttacks);
    };

    window.updateParam = (idx, key, value) => {
        if (!window.orderedAttacks[idx]) return;
        window.orderedAttacks[idx].params[key] = parseFloat(value);
    };

    window.setBatchSize = (idx, value, btn) => {
    window.orderedAttacks[idx].params["batch_size"] = value;

    // Destacar o botão selecionado
    const buttons = btn.parentNode.querySelectorAll("button");
    buttons.forEach(b => b.style.backgroundColor = "");
    btn.style.backgroundColor = "#d0ebff";  // azul clarinho
    };

    window.setButtonParam = (idx, key, value, btn) => {
    // Atualiza os parâmetros no JS
    window.orderedAttacks[idx].params[key] = isNaN(value) ? value : parseFloat(value);

    // Estiliza botões (deseleciona os outros)
    const parent = btn.parentNode;
    parent.querySelectorAll("button").forEach(b => b.style.backgroundColor = "");
    btn.style.backgroundColor = "#d0ebff";
};


    window.runAttacks = async () => {
        const attacksToSend = window.orderedAttacks.filter(a => a && a.name);

        if (attacksToSend.length === 0) {
            alert("Nenhum ataque selecionado.");
            return;
        }

        const endpoint = currentMode === 'poisoning' ? '/run_poisoning' : '/run_evasion'; // <--- Atualiza aqui

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ attacks: attacksToSend })
        });

        const results = await response.json();
        showResults(results);
    };

  function showResults(data) {
    resultsDiv.innerHTML = '';
    document.getElementById('results-section').style.display = 'block';
    document.getElementById('save-button').style.display = 'inline-block';
    lastResults = data;

    data.forEach(res => {
        const div = document.createElement('div');
        div.className = 'attack-block';

        if (res.attack === 'Clean Accuracy') {
            const accDiv = document.createElement('div');
            accDiv.className = 'attack-block';
            accDiv.innerHTML = `
                <strong>Clean Accuracy:</strong> ${res.result.accuracy}%<br>
            `;
            resultsDiv.appendChild(accDiv);
            return; // já trataste, salta para o próximo
        }

        if (currentMode === 'poisoning') {
            div.innerHTML = `
                <strong>${res.attack}</strong><br>
                Accuracy in original samples: ${res.result.accuracy_original_samples}%<br>
                Accuracy After Adversarial Training: ${res.result.accuracy_after_training}%<br>
            `;
        } else {
            div.innerHTML = `
                <strong>${res.attack}</strong><br>
                Perturbadas: ${res.result.fooled}<br>
                Restantes: ${res.result.remaining}<br>
                Robust Accuracy: ${res.result.robust_accuracy}%<br>
                Attack Success Rate: ${res.result.asr}%<br>
                Tempo de execução: ${res.result.execution_time} s<br>
                RSD (Isolado): ${res.result.rsd}<br>
            `;
        }
        if (res.attack === 'Final Summary') {
            div.innerHTML = `
                <strong>Resumo Final</strong><br>
                Amostras Iniciais: ${res.result.initial}<br>
                Amostras Restantes: ${res.result.remaining}<br>
                Final Robust Accuracy: ${res.result.final_robust_accuracy}%<br>
                Final ASR: ${res.result.final_asr}%<br>
                Total Time: ${res.result.total_time}s<br>
                RSD: ${res.result.rsd}<br>

            `;
        }

        resultsDiv.appendChild(div);
    });
    }


  window.saveResults = () => {
    const tbody = document.querySelector("#history-table tbody");

    lastResults.forEach(res => {
        const row = document.createElement("tr");

        if (currentMode === 'poisoning') {
            row.innerHTML = `
                <td>${currentModelName || 'modelo.joblib'}</td>
                <td>${res.attack}</td>
                <td>${res.result.clean_accuracy}%</td>
                <td>${res.result.robust_accuracy}%</td>
            `;
        } else {
            row.innerHTML = `
                <td>${currentModelName || 'modelo.joblib'}</td>
                <td>${res.attack}</td>
                <td>${res.result.fooled}</td>
                <td>${res.result.remaining}</td>
            `;
        }


        tbody.appendChild(row);
    });

    resetAll();
  };

  function resetAll() {
    // Limpar formulário
    document.getElementById("upload-form").reset();

    // Esconder seções
    document.getElementById("attack-selection").style.display = "none";
    document.getElementById("run-button").style.display = "none";
    document.getElementById("save-button").style.display = "none";
    document.getElementById("results-section").style.display = "none";

    // Limpar conteúdo
    document.getElementById("attack-list").innerHTML = "";
    document.getElementById("results").innerHTML = "";

    // Reset de variáveis
    window.orderedAttacks = [];
    lastResults = [];
    currentModelName = '';
  }



});
