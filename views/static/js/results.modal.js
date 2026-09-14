function openGlobalCMModal(btn) {
  let fullPayload = {};
  try { fullPayload = JSON.parse(btn.dataset.full || "{}"); } catch { fullPayload = {}; }

  const cm = fullPayload.adv_confusion_global || null;
  const labels = fullPayload.class_labels || [];

  renderConfusionMatrix("global-confusion-matrix", cm, labels);

  const modal = new bootstrap.Modal(document.getElementById('globalCMModal'));
  modal.show();
}

function renderConfusionMatrix(containerId, matrix, labels) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  if (!matrix || matrix.length === 0) {
    container.innerHTML = `<p class="text-muted">No samples misclassified at this level.</p>`;
    return;
  }

  let html = `<table class="table table-sm table-bordered text-center mt-2">`;

  html += `<thead><tr><th>True \\ Adv Pred</th>`;
  labels.forEach(l => html += `<th>${l}</th>`);
  html += `</tr></thead><tbody>`;

  matrix.forEach((row, i) => {
    html += `<tr><th>${labels[i]}</th>`;
    row.forEach(v => html += `<td>${v}</td>`);
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}
