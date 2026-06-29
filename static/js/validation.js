/**
 * Credit Card Approval Prediction - Form Validation & UI Helpers
 */

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictionForm");
    if (!form) return;

    const resetBtn = document.getElementById("resetBtn");
    if (resetBtn) {
        resetBtn.addEventListener("click", function (e) {
            e.preventDefault();
            form.reset();
            clearValidationErrors();
            const resultBox = document.getElementById("resultBox");
            if (resultBox) resultBox.style.display = "none";
        });
    }

    form.addEventListener("submit", function (e) {
        if (!validateForm()) {
            e.preventDefault();
        }
    });

    // Real-time validation on blur
    const fields = form.querySelectorAll("[data-validate]");
    fields.forEach(function (field) {
        field.addEventListener("blur", function () {
            validateField(field);
        });
    });
});

function validateForm() {
    const form = document.getElementById("predictionForm");
    if (!form) return true;

    let isValid = true;
    const fields = form.querySelectorAll("[data-validate]");
    fields.forEach(function (field) {
        if (!validateField(field)) {
            isValid = false;
        }
    });
    return isValid;
}

function validateField(field) {
    const rules = field.getAttribute("data-validate").split("|");
    const value = field.value.trim();
    let error = "";

    for (const rule of rules) {
        if (rule === "required" && !value) {
            error = "This field is required.";
            break;
        }
        if (rule.startsWith("min:")) {
            const min = parseFloat(rule.split(":")[1]);
            if (parseFloat(value) < min) {
                error = `Minimum value is ${min}.`;
                break;
            }
        }
        if (rule.startsWith("max:")) {
            const max = parseFloat(rule.split(":")[1]);
            if (parseFloat(value) > max) {
                error = `Maximum value is ${max}.`;
                break;
            }
        }
        if (rule === "number" && value && isNaN(parseFloat(value))) {
            error = "Please enter a valid number.";
            break;
        }
    }

    showFieldError(field, error);
    return !error;
}

function showFieldError(field, error) {
    field.classList.toggle("is-invalid", !!error);
    const feedback = field.parentElement.querySelector(".invalid-feedback");
    if (feedback) {
        feedback.textContent = error;
        feedback.style.display = error ? "block" : "none";
    }
}

function clearValidationErrors() {
    const form = document.getElementById("predictionForm");
    if (!form) return;
    form.querySelectorAll(".is-invalid").forEach(function (el) {
        el.classList.remove("is-invalid");
    });
    form.querySelectorAll(".invalid-feedback").forEach(function (el) {
        el.textContent = "";
        el.style.display = "none";
    });
}
