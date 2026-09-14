function toggleFoolboxParams() {
  const t = document.getElementById('foolbox_attack_type').value;

  const show = (ids, cond) =>
    ids.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = cond ? 'block' : 'none';
    });

  show(['epsilon-param'], ['FGM','FGSM','PGD','L2PGD','LinfBIM','L2BIM'].includes(t));
  show(['steps-param'], ['PGD','L2PGD','LinfBIM','L2BIM','L2DeepFool','LinfDeepFool','HopSkipJumpAttack'].includes(t));
  show(['rel-stepsize-param'], ['PGD','L2PGD','LinfBIM','L2BIM'].includes(t));
  show(['overshoot-param'], ['L2DeepFool','LinfDeepFool'].includes(t));
}
