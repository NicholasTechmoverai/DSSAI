const form = document.getElementById("prediction-form");
const responsePanel = document.getElementById("response-panel");

form.addEventListener("submit", (event) => {
  event.preventDefault();

  if (responsePanel.hidden) {
    responsePanel.hidden = false;
    requestAnimationFrame(() => {
      responsePanel.classList.add("visible");
    });
  } else {
    responsePanel.classList.remove("visible");
    requestAnimationFrame(() => {
      responsePanel.classList.add("visible");
    });
  }
});
