import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import sys

class MedicalSystemML:
    def __init__(self):
        self.diseases = {
            "Fever": ["Headache", "High Temperature", "Weakness", "Chills", "Sweating"],
            "Diabetes": ["Frequent urination", "Increased thirst", "Weight loss", "Fatigue", "Blurred vision"],
            "Asthma": ["Breathlessness", "Coughing", "Chest tightness", "Wheezing"],
            "Hypertension": ["Headache", "Dizziness", "Nosebleeds", "Blurred vision", "Chest pain"]
        }

        self.medicines = {
            "Fever": ["Paracetamol", "Ibuprofen", "Rest", "Hydration"],
            "Diabetes": ["Insulin", "Metformin", "Low sugar diet"],
            "Asthma": ["Inhaler", "Bronchodilator", "Avoid dust"],
            "Hypertension": ["Amlodipine", "Losartan", "Reduce salt intake"]
        }

        self.doctors = {
            "Fever": "General Physician",
            "Diabetes": "Endocrinologist",
            "Asthma": "Pulmonologist",
            "Hypertension": "Cardiologist"
        }

        self.X_texts, self.y_labels = self._build_training_examples()
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self._train_model()

    def _build_training_examples(self):
        X = []
        y = []
        for disease, symps in self.diseases.items():
            full = ", ".join(symps)
            X.append(full)
            y.append(disease)

            for s in symps:
                X.append(s)
                y.append(disease)

            for i in range(len(symps)):
                for j in range(i+1, min(i+3, len(symps))):
                    combo = f"{symps[i]}, {symps[j]}"
                    X.append(combo)
                    y.append(disease)


        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuf = [X[i] for i in indices]
        y_shuf = [y[i] for i in indices]
        return X_shuf, y_shuf

    def _train_model(self):
        try:
            self.model.fit(self.X_texts, self.y_labels)
        except Exception as e:
            raise RuntimeError(f"Model training failed: {e}")

    def predict(self, symptoms, top_k=3):

        if isinstance(symptoms, list):
            text = ", ".join(symptoms)
        else:
            text = str(symptoms)

        if not text.strip():
            return []

        probs = self.model.predict_proba([text])[0]
        classes = self.model.named_steps['multinomialnb'].classes_
        pairs = list(zip(classes, probs))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        return pairs_sorted[:top_k]

    def recommend(self, symptoms):
        preds = self.predict(symptoms, top_k=3)
        if not preds:
            return "No input symptoms provided."

        top_disease, top_prob = preds[0]
        result_lines = []
        result_lines.append(f"Predictions (top {len(preds)}):\n")
        for dis, prob in preds:
            meds = self.medicines.get(dis, ["Consult a doctor"])
            doc = self.doctors.get(dis, "Specialist")
            result_lines.append(f"Disease: {dis}  —  Confidence: {prob:.2%}")
            result_lines.append(f"Medicines: {', '.join(meds)}")
            result_lines.append(f"Doctor: {doc}\n")

        if top_prob < 0.25:
            result_lines.insert(0, "Low confidence in predictions. Please consult a doctor for accurate diagnosis.\n")

        return "\n".join(result_lines)


def run_app():
    try:
        
        pass
    except Exception:
        messagebox.showerror(
            "Missing dependency",
            "scikit-learn is required to run this app. Install with:\n\npip install scikit-learn"
        )
        return

    system = MedicalSystemML()

    def get_recommendation():
        raw = entry_symptoms.get()
        symptoms = [s.strip() for s in raw.split(",") if s.strip()]

        if not symptoms:
            messagebox.showwarning("Input Error", "Please enter at least one symptom (comma separated).")
            return

        try:
            result = system.recommend(symptoms)
        except Exception as e:
            result = f"Error during prediction: {e}"

        text_result.config(state="normal")
        text_result.delete("1.0", tk.END)
        text_result.insert(tk.END, result)
        text_result.config(state="disabled")

    root = tk.Tk()
    root.title("Medical Requirement System — ML Powered")
    root.geometry("600x480")
    root.config(bg="#f0f8ff")

    tk.Label(
        root,
        text="Medical Requirement System — ML Powered",
        font=("Arial", 16, "bold"),
        bg="#f0f8ff",
        fg="darkblue"
    ).pack(pady=10)

    tk.Label(root, text="Enter Symptoms (comma separated):",
             font=("Arial", 12), bg="#f0f8ff").pack(pady=5)

    entry_symptoms = tk.Entry(root, font=("Arial", 12), width=60)
    entry_symptoms.pack(pady=5)
    entry_symptoms.insert(0, "e.g. headache, high temperature")

    tk.Button(root, text="Search medicine", font=("Arial", 12, "bold"),
              bg="green", fg="white", command=get_recommendation).pack(pady=10)

    text_result = tk.Text(root, font=("Arial", 12), height=15, width=72, wrap="word", state="disabled")
    text_result.pack(pady=10)

    footer = tk.Label(root, text="⚠️ This app gives suggestions only — not a medical diagnosis. Consult a doctor.",
                      font=("Arial", 9), bg="#f0f8ff", fg="red")
    footer.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    # Quick check for scikit-learn; if missing, show a message and exit
    try:
        import sklearn  # noqa: F401
    except Exception:
        tk.Tk().withdraw()  # hide main root
        messagebox.showerror(
            "Missing dependency",
            "This application requires scikit-learn. Install it with:\n\npip install scikit-learn\n\nThen re-run the script."
        )
        sys.exit(1)

    run_app()
