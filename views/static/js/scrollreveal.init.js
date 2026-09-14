document.addEventListener("DOMContentLoaded", () => {
  if (typeof ScrollReveal === "undefined") return;

  // ---------- MENU ----------
  ScrollReveal().reveal(".upload-card", {
    origin: "bottom",
    distance: "20px",
    duration: 800,
    easing: "ease-in-out",
    interval: 200
  });

  ScrollReveal().reveal(".form-section", {
    origin: "top",
    distance: "30px",
    duration: 1000,
    delay: 200
  });

  // ---------- INDEX ----------
  ScrollReveal().reveal(".tool-card", {
    origin: "bottom",
    distance: "40px",
    duration: 1000,
    delay: 200,
    easing: "ease-in-out",
    interval: 100
  });

  ScrollReveal().reveal(".fade-in-up", {
    origin: "bottom",
    distance: "40px",
    duration: 1000,
    easing: "ease-out"
  });
});
