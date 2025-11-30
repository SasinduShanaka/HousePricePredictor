from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    sqft_val = ""
    beds_val = ""
    baths_val = ""

    if request.method == "POST":
        try:
            # Preserve entered values for re-rendering
            sqft_val = request.form.get("sqft", "")
            beds_val = request.form.get("bedrooms", "")
            baths_val = request.form.get("bathrooms", "")

            sqft = float(sqft_val)
            beds = float(beds_val)
            baths = float(baths_val)

            # Basic server-side validation to avoid negative values
            if sqft < 0 or beds < 0 or baths < 0:
                raise ValueError("Values cannot be negative.")

            # Optional: additional sanity checks
            if sqft == 0:
                raise ValueError("Area (sqft) must be greater than 0.")
            # Bedrooms/Bathrooms should be whole numbers
            if beds % 1 != 0 or baths % 1 != 0:
                raise ValueError("Bedrooms and bathrooms must be whole numbers.")

            # Prepare data for prediction
            data = pd.DataFrame([[sqft, beds, baths]], 
                                columns=["sqft_living", "bedrooms", "bathrooms"])

            pred = model.predict(data)[0]

            # Format number nicely (e.g., 15,900,000 LKR)
            prediction = f"{int(pred):,} LKR"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, sqft_val=sqft_val, beds_val=beds_val, baths_val=baths_val)

if __name__ == "__main__":
    app.run(debug=True)
