(function () {
    const PARAM_TOOLTIPS = {
        // ZOO
        zoo_max_iter:            'Maximum optimization iterations. Higher values may improve the attack but increase runtime.',
        zoo_binary_search_steps: 'Steps to search for the minimum confidence boundary. More steps = finer precision but slower.',
        zoo_learning_rate:       'Adam optimizer step size for updating the perturbation. Smaller = more stable but slower convergence.',
        zoo_confidence:          'Required confidence margin above the decision boundary. Higher = harder-to-find but more robust adversarial examples.',
        // FGM
        fgm_eps:                 'Maximum perturbation budget (ε). Controls the maximum allowed change per feature. Larger ε = stronger but less realistic attack.',
        fgm_eps_step:            'Gradient step size per update. Should be ≤ ε. Controls the granularity of each perturbation step.',
        fgm_batch_size:          'Number of samples processed simultaneously. Larger batches are faster but require more memory.',
        fgm_norm:                'Norm used to measure perturbation size. L∞ = max change per feature; L2 = overall magnitude; L1 = sparse changes.',
        // JSMA
        jsma_theta:              'Amount of perturbation applied to each selected feature at each step. Controls per-feature modification magnitude.',
        jsma_gamma:              'Maximum fraction of features that can be modified (e.g. 0.2 = at most 20% of features). Controls attack sparsity.',
        jsma_batch_size:         'Number of samples processed per batch.',
        // PGD
        pgd_eps:                 'Maximum perturbation budget (ε). Total allowed feature change accumulated across all iterations.',
        pgd_eps_step:            'Size of each projected gradient step. Smaller = finer control over the perturbation path.',
        pgd_batch_size:          'Number of samples processed per batch.',
        pgd_max_iter:            'Number of projected gradient descent + projection steps. More iterations = stronger attack.',
        num_random_init_pgd:     'Number of random restarts before each attack. More restarts reduce the chance of getting stuck in local optima.',
        pgd_norm:                'Norm for measuring perturbation magnitude. L∞ = max change per feature; L2 = euclidean distance.',
        // Boundary Attack
        boundary_batch_size:     'Samples processed per batch. Boundary Attack is decision-based — requires only model output, not gradients.',
        boundary_max_iter:       'Number of random walk steps towards the decision boundary. More iterations = smaller, more refined perturbations.',
        // DeepFool
        deepfool_max_iter:       'Maximum linearization steps to find the minimal perturbation to cross the decision boundary.',
        deepfool_eps:            'Convergence overshoot factor. A very large value (e.g. 1000000) disables early stopping, ensuring full convergence.',
        deepfool_batch_size:     'Number of samples processed per batch.',
        // HopSkipJump
        adv_max_iter:            'Maximum number of optimization iterations. Higher = stronger attack at the cost of more computation.',
        max_eval:                'Maximum model queries per iteration for gradient estimation. More queries = better gradient approximation.',
        hsj_batch_size:          'Number of samples processed per batch.',
        norm:                    'Norm used to measure the perturbation distance. L2 = euclidean distance; L∞ = max change per feature.',
        // C&W
        confidence:              'Minimum confidence margin above the decision boundary. Higher = stronger but harder-to-find adversarial examples.',
        learning_rate:           'Adam optimizer learning rate for finding the optimal perturbation. Typical range: 0.001–0.1.',
        batch_size:              'Number of samples processed per batch.',
        // BIM
        bim_eps:                 'Maximum perturbation budget (ε). Total allowed cumulative change per feature across all iterations.',
        bim_eps_step:            'Gradient step size per iteration. Typically set to ε / max_iter for uniform budget distribution.',
        bim_max_iter:            'Number of iterative gradient steps. BIM is an iterative extension of FGM without random restarts.',
        bim_batch_size:          'Number of samples processed per batch.',
        // A2PM — Interval
        features_int:            'Continuous/float features to apply interval perturbations to (comma-separated indices or names).',
        integer_features:        'Integer-valued features to perturb. Values will be rounded after perturbation to preserve integrality.',
        ratio:                   'Base perturbation ratio — fraction of each feature\'s observed range that can be modified.',
        max_ratio:               'Maximum perturbation ratio allowed during the search. Caps how aggressively features are pushed.',
        interval_probability:    'Probability of applying the interval perturbation to each sample (0–1). Lower = more selective.',
        // A2PM — Settings
        a2pm_iterations:         'Number of optimization rounds the attack runs. More iterations allow finer perturbation refinement.',
        a2pm_patience:           'Rounds without improvement before early stopping. Prevents wasted computation on stagnant attacks.',
    };

    function addInfoIcon(input, tip) {
        const parent = input.closest('div');
        if (!parent) return;

        let label = null;
        let sib = input.previousElementSibling;
        while (sib) {
            if (sib.tagName === 'LABEL') { label = sib; break; }
            sib = sib.previousElementSibling;
        }
        if (!label) label = parent.querySelector('label');
        if (!label || label.querySelector('.param-info')) return;

        const icon = document.createElement('i');
        icon.className = 'bi bi-info-circle param-info';
        icon.setAttribute('data-bs-toggle', 'tooltip');
        icon.setAttribute('data-bs-placement', 'top');
        icon.setAttribute('data-bs-title', tip);
        label.appendChild(icon);
    }

    function applyAll() {
        for (const [name, tip] of Object.entries(PARAM_TOOLTIPS)) {
            document.querySelectorAll(`[name="${name}"]`).forEach(el => addInfoIcon(el, tip));
        }
        document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
            new bootstrap.Tooltip(el, { trigger: 'hover focus' });
        });
    }

    window._applyParamTooltips = applyAll;
    document.addEventListener('DOMContentLoaded', applyAll);
})();
