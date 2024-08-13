from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# تحميل النموذج المدرب و LabelEncoder
best_model = joblib.load('best_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # تحضير المدخلات الجديدة
    answers = data['answers']

    # تحويل المدخلات إلى نفس التنسيق المستخدم في التدريب
    input_data = {
        'Q1': [answers[0]],
        'Q2': [answers[1]],
        'Q3': [answers[2]],
        'Q4': [answers[3]],
        'Q5': [answers[4]],
        'Q6': [answers[5]],
        'Q7': [answers[6]]
    }

    input_df = pd.DataFrame(input_data)
    input_df = pd.get_dummies(input_df)

    # ضمان أن تكون الأعمدة في المدخلات الجديدة تتطابق مع الأعمدة المستخدمة في التدريب
    missing_cols = set(best_model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[best_model.feature_names_in_]

    # تنفيذ التنبؤ
    predicted_quality_encoded = best_model.predict(input_df)
    predicted_quality = label_encoder.inverse_transform(predicted_quality_encoded)

    # نصائح بناءً على الإجابات
    explanations_recommendations = {
        "Q1": ("Stress and anxiety can interfere with sleep.", "Practice relaxation techniques before bed."),
        "Q2": ("Nicotine is a stimulant that can disrupt sleep.", "Avoid nicotine for at least 4-6 hours before bedtime."),
        "Q3": ("Blue light from devices can suppress melatonin production.", "Avoid devices 1-2 hours before bedtime."),
        "Q4": ("Noise can disrupt sleep.", "Create a quiet sleep environment."),
        "Q5": ("Extreme temperatures can interfere with sleep.", "Maintain a warm bedroom temperature."),
        "Q6": ("Caffeine is a stimulant that can interfere with sleep.", "Avoid caffeine for at least 4-6 hours before bedtime."),
        "Q7": ("Eating close to bedtime can disrupt sleep.", "Finish eating at least 2-3 hours before bedtime."),
    }

    reasons = []
    recommendations = []

    if predicted_quality[0] in ['Poor', 'Average']:
        for i, answer in enumerate(answers, start=1):
            if answer in ['Yes','Moderately noisy', 'Noisy', 'Cool', 'Hot', 'Yes']:
                q_key = f"Q{i}"
                explanation, recommendation = explanations_recommendations[q_key]
                reasons.append(explanation)
                recommendations.append(recommendation)

    response = {
        'predicted_quality': predicted_quality[0],
        'reasons': reasons,
        'recommendations': recommendations
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
