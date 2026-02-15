const form = document.getElementById("prediction-form");
const responsePanel = document.getElementById("response-panel");
const symptomInputs = [
  document.getElementById("symptom-1"),
  document.getElementById("symptom-2"),
  document.getElementById("symptom-3")
];

let nextInputIndex = 0; 

function confidenceColor(confidence) {
  if (confidence < 20) return "red";
  if (confidence <= 60) return "blue";
  return "green";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const diseaseEl = document.getElementById("result-disease");
  const causeEl = document.getElementById("result-cause");
  const preventionEl = document.getElementById("result-prevention");

  diseaseEl.textContent = "Predicting...";
  causeEl.textContent = "Predicting...";
  preventionEl.textContent = "Predicting...";

  responsePanel.hidden = false;
  responsePanel.classList.remove("visible");
  requestAnimationFrame(() => responsePanel.classList.add("visible"));

  const payload = {
    "symptom-1": symptomInputs[0].value.trim(),
    "symptom-2": symptomInputs[1].value.trim(),
    "symptom-3": symptomInputs[2].value.trim()
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) throw new Error("Request failed");

    const data = await response.json();

    diseaseEl.textContent = data.disease?.name || data.disease || "N/A";
    diseaseEl.style.color = confidenceColor(
      data.disease?.confidence ?? data.disease_confidence ?? 0
    );

    causeEl.textContent = data.cause?.name || data.cause || "N/A";
    causeEl.style.color = confidenceColor(
      data.cause?.confidence ?? data.cause_confidence ?? 0
    );

    preventionEl.textContent = data.prevention || "N/A";
    preventionEl.style.color = confidenceColor(
      data.prevention_confidence ?? 50
    );

  } catch (error) {
    console.error("Error:", error);
    diseaseEl.textContent = "Error fetching prediction";
    causeEl.textContent = "";
    preventionEl.textContent = "";
  }
});

// Handle individual symptom button clicks (fill inputs one by one)
document.querySelectorAll(".symptom-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    if (nextInputIndex > 2) nextInputIndex = 0;

    symptomInputs[nextInputIndex].value = btn.textContent.trim();
    nextInputIndex++;

    responsePanel.classList.remove("visible");
    responsePanel.hidden = true;
  });
});

// Handle "Test All" buttons for full row autofill + auto-submit
document.querySelectorAll(".fill-row-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const row = btn.closest(".symptom-row");
    const symptoms = row.dataset.symptoms.split(",");

    symptomInputs[0].value = symptoms[0].trim();
    symptomInputs[1].value = symptoms[1].trim();
    symptomInputs[2].value = symptoms[2].trim();

    nextInputIndex = 0; 

    form.dispatchEvent(new Event("submit"));
  });
});
