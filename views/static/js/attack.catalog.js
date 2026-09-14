(function () {
    const NORM_KEYS = new Set(['fgm_norm', 'pgd_norm', 'bim_norm', 'hsj_norm', 'norm']);

    function renderParam(param, mode) {
        const { key, label, type, default: def } = param;

        // Target class: shown ONLY in the Targeted tab (default class 0)
        if (key.endsWith('_target_class')) {
            if (mode !== 'targeted') return '';
            return `
            <div class="mb-2">
                <label class="form-label small text-info">Target Class:</label>
                <select class="form-select form-select-sm border-info target-class-selector" name="${key}">
                    <option value="0" selected>0</option>
                </select>
            </div>`;
        }

        // Targeted flag is implied by the tab — never rendered
        if (key.endsWith('_targeted')) return '';

        // In the Targeted tab, parameters are shared with the Untargeted tab,
        // so only the target class is shown here (no duplicated inputs).
        if (mode === 'targeted') return '';

        if (type === 'bool') return '';

        if (NORM_KEYS.has(key)) {
            return `
            <div class="mb-2">
                <label class="form-label small">${label}</label>
                <select class="form-select form-select-sm bg-dark text-white border-secondary" name="${key}">
                    <option value="inf"${def === 'inf' ? ' selected' : ''}>L&#x221E; (inf)</option>
                    <option value="2"${def === 2 ? ' selected' : ''}>L2</option>
                    <option value="1"${def === 1 ? ' selected' : ''}>L1</option>
                </select>
            </div>`;
        }

        const inputType = (type === 'float' || type === 'int') ? 'number' : 'text';
        const step = type === 'float' ? '0.001' : '1';

        return `
            <div class="mb-2">
                <label class="form-label small">${label}</label>
                <input type="${inputType}"
                       class="form-control form-control-sm bg-transparent text-white border-secondary"
                       name="${key}" value="${def}" step="${step}">
            </div>`;
    }

    function renderAttackItem(attack, mode) {
        const supportsTargeted = attack.params.some(p => p.key.endsWith('_targeted'));
        // Attacks that cannot be targeted (e.g., DeepFool) are hidden in the Targeted tab
        if (mode === 'targeted' && !supportsTargeted) return '';

        const sid = attack.id.replace(/[^a-zA-Z0-9]/g, '_') + '_' + mode;
        const fieldName = mode === 'targeted' ? 'selected_attacks_targeted' : 'selected_attacks';
        const paramsHtml = attack.params.map(p => renderParam(p, mode)).join('');
        const hasParams = paramsHtml.trim().length > 0;

        return `
        <div class="attack-item mb-3">
            <div class="d-flex align-items-center justify-content-between p-2 rounded bg-dark border border-secondary shadow-sm">
                <div class="d-flex align-items-center gap-3 ps-2">
                    <input class="form-check-input" type="checkbox" id="${sid}_check"
                           name="${fieldName}" value="${attack.id}" checked>
                    <div class="d-flex align-items-center gap-2">
                        <span class="text-white fw-semibold">${attack.label || attack.name}</span>
                    </div>
                </div>
                ${hasParams ? `
                <button type="button" class="btn btn-sm text-info fw-bold"
                        data-bs-toggle="collapse" data-bs-target="#${sid}_params">
                    <i class="bi bi-sliders"></i> Edit Configuration
                </button>` : ''}
            </div>
            ${hasParams ? `
            <div class="collapse" id="${sid}_params">
                <div class="card card-body bg-dark border-secondary mt-2">
                    ${paramsHtml}
                </div>
            </div>` : ''}
        </div>`;
    }

    async function init() {
        let catalog = [];
        try {
            const res = await fetch('/api/attacks/available');
            catalog = await res.json();
        } catch (e) {
            console.error('Failed to load attack catalog:', e);
            return;
        }

        const artAttacks = catalog.filter(a => a.framework === 'ART');

        const untContainer = document.getElementById('attack-list-untargeted');
        const tgtContainer = document.getElementById('attack-list-targeted');

        if (untContainer) {
            untContainer.innerHTML = artAttacks.map(a => renderAttackItem(a, 'untargeted')).join('');
        }
        if (tgtContainer) {
            tgtContainer.innerHTML = artAttacks.map(a => renderAttackItem(a, 'targeted')).join('');
        }

        // Re-apply param tooltips after dynamic elements are in the DOM
        if (typeof window._applyParamTooltips === 'function') {
            window._applyParamTooltips();
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
