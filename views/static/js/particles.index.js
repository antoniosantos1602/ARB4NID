document.addEventListener("DOMContentLoaded", () => {
  const subtleParticles = {
    particles: {
      number: { value: 70, density: { enable: true, value_area: 800 } },
      color: { value: "#00ffff" },
      shape: { type: "circle" },
      opacity: { value: 0.2 },
      size: { value: 2, random: true },
      line_linked: {
        enable: true,
        distance: 100,
        color: "#00ffff",
        opacity: 0.3,
        width: 1
      },
      move: {
        enable: true,
        speed: 0.6,
        out_mode: "bounce"
      }
    },
    interactivity: {
      events: {
        onhover: { enable: true, mode: "grab" }
      },
      modes: {
        grab: {
          distance: 150,
          line_linked: { opacity: 0.5 }
        }
      }
    },
    retina_detect: true
  };

  ["particles-features", "particles-comparison", "particles-about"].forEach(id => {
    if (document.getElementById(id)) {
      particlesJS(id, JSON.parse(JSON.stringify(subtleParticles)));
    }
  });
});
