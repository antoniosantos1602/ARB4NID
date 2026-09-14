document.addEventListener("DOMContentLoaded", () => {
  // -------- FEATURES --------
  if (document.getElementById("particles-features")) {
    particlesJS("particles-features", {
      particles: {
        number: { value: 160, density: { enable: true, value_area: 600 } },
        color: { value: "#00ffff" },
        shape: { type: "circle" },
        opacity: { value: 0.4 },
        size: { value: 3, random: true },
        line_linked: {
          enable: true,
          distance: 120,
          color: "#00ffff",
          opacity: 0.4,
          width: 1
        },
        move: {
          enable: true,
          speed: 1.5,
          out_mode: "bounce"
        }
      },
      interactivity: {
        events: {
          onhover: { enable: true, mode: "grab" }
        },
        modes: {
          grab: {
            distance: 200,
            line_linked: { opacity: 0.6 }
          }
        }
      },
      retina_detect: true
    });
  }

  // -------- COMPARISON --------
  if (document.getElementById("particles-comparison")) {
    particlesJS("particles-comparison", {
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
    });
  }
});
