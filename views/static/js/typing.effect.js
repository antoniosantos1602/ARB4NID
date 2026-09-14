document.addEventListener("DOMContentLoaded", () => {
  const target = document.getElementById("typing");
  if (!target) return;

  const text =
    "Adversarial Robustness Benchmark for Network Intrusion Detection";

  let i = 0;
  const speed = 25;

  function typeEffect() {
    if (i < text.length) {
      target.textContent += text.charAt(i);
      i++;
      setTimeout(typeEffect, speed);
    }
  }

  typeEffect();
});
