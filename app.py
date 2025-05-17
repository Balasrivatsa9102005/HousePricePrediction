from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load(r'C:\Users\balas\OneDrive\Desktop\Balu\Myprojects\vs\House\vijayawada_price_predictor.pkl')
df = pd.read_excel(r'C:\Users\balas\OneDrive\Desktop\Balu\Myprojects\vs\House\vijayawada_house_prices.csv.xlsx')


df.columns = [col.lower() for col in df.columns]
df['locality'] = df['locality'].str.lower()
df['bhk'] = df['bhk'].astype(str)  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        locality_input = request.form['locality'].strip().lower()
        bhk_input = request.form['bhk'].strip()

        # Filter by both locality and BHK
        sample_row = df[(df['locality'] == locality_input) & (df['bhk'] == bhk_input)].head(1)

        if sample_row.empty:
            return render_template('index.html', prediction_text='❌ No matching house found for this locality and BHK combination. Try again.')

        expected_cols = model.feature_names_in_

        input_dict = {}
        for col in expected_cols:
            for df_col in df.columns:
                if df_col.replace(" ", "").lower() == col.replace(" ", "").lower():
                    input_dict[col] = sample_row[df_col].values[0]
                    break

        
        input_dict['BHK'] = int(input_dict['BHK'])
        input_dict['Size (sqft)'] = float(input_dict['Size (sqft)'])
        input_dict['Price per sqft'] = float(input_dict['Price per sqft'])

        input_data = pd.DataFrame([input_dict])

        predicted_price = round(model.predict(input_data)[0], 2)

        return render_template('index.html',
                               prediction_text=(
                                   f"📍 Locality: {locality_input.title()} | "
                                   f"🏙️ Region Type: {input_dict['Region Type']} | "
                                   f"🛏️ BHK: {input_dict['BHK']} | "
                                   f"📏 Size: {input_dict['Size (sqft)']} sqft | "
                                   f"💸 Price per sqft: ₹{input_dict['Price per sqft']:,.2f}"
                               ),
                               price_text=f"💰 Estimated Total Price: ₹{predicted_price:,.2f}")

    except Exception as e:
        return render_template('index.html',
                               prediction_text="⚠️ Error during prediction.",
                               price_text=str(e))

if __name__ == '__main__':
    app.run(debug=True)
