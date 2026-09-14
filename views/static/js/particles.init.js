document.addEventListener("DOMContentLoaded", () => {
  if (!document.getElementById("particles-js")) return;

  particlesJS("particles-js", {
    particles: {
      number: { value: 65, density: { enable: true, value_area: 900 } },
      color: { value: "#2dd4bf" },
      shape: { type: "circle" },
      opacity: { value: 0.3 },
      size: { value: 3, random: true },
      line_linked: {
        enable: true,
        distance: 130,
        color: "#2dd4bf",
        opacity: 0.3,
        width: 1
      },
      move: {
        enable: true,
        speed: 1.6,
        out_mode: "bounce"
      }
    },
    interactivity: {
      events: {
        onhover: { enable: true, mode: "grab" }
      },
      modes: {
        grab: {
          distance: 180,
          line_linked: { opacity: 0.5 }
        }
      }
    },
    retina_detect: true
  });
});
