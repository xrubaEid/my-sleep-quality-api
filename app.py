from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
import joblib

# تحميل النموذج والمحول (LabelEncoder)
model = joblib.load('best_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# إنشاء تطبيق FastAPI
app = FastAPI()

# تعريف هيكل المدخلات باستخدام Pydantic
class SleepQualityInput(BaseModel):
    Q1: str
    Q2: str
    Q3: str
    Q4: str
    Q5: str
    Q6: str
    Q7: str

# تعريف نقطة النهاية للتنبؤ
@app.post("/predict/")
def predict_sleep_quality(input_data: SleepQualityInput):
    try:
        input_df = pd.DataFrame([input_data.dict()])
        input_df = pd.get_dummies(input_df)

        # ضمان أن تكون الأعمدة في المدخلات الجديدة تتطابق مع الأعمدة المستخدمة في التدريب
        missing_cols = set(model.feature_names_in_) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

        # تنفيذ التنبؤ
        predicted_quality_encoded = model.predict(input_df)
        predicted_quality = label_encoder.inverse_transform(predicted_quality_encoded)

        return {"predicted_sleep_quality": predicted_quality[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
