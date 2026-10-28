import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Call API Key 
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ تنبيه: لم يتم العثور على مفتاح GROQ_API_KEY في ملف .env!")

# init Groq
client = Groq(api_key=api_key)

def generate_medical_report(age, gender, symptoms, model_prediction, confidence):
    prompt = f"""
    أنت طبيب استشاري متخصص في الأشعة والأمراض الصدرية. قم إعداد تقرير طبي احترافي بناءً على البيانات التالية:
    - عمر المريض: {age} سنة
    - النوع: {gender}
    - الأعراض التي يشتكي منها: {symptoms}
    - نتيجة الذكاء الاصطناعي لتحليل الأشعة: {model_prediction} (نسبة الثقة: {confidence:.2f}%)
    
    قم بتضمين الآتي في التقرير (باللغة العربية وبصياغة طبية احترافية):
    1. تحليل مبدئي للحالة وربط الأعراض بالسن والنتيجة المتوقعة.
    2. تفسير سبب الإصابة المحتمل (هل هو مرتبط بعدوى، مناعة، أو عوامل عمرية).
    3. الخطة العلاجية المقترحة المبدئية (للإرشاد العام وليست وصفة نهائية).
    4. نصائح الراحة المنزلية والمشروبات المفيدة (مثل المشروبات الدافئة، الأطعمة الداعمة للمناعة).
    5. متى يجب التوجه للطوارئ فوراً؟
    """
     
    try:
        completion = client.chat.completions.create(
        
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600 
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"عذراً، حدث خطأ أثناء توليد التقرير بالذكاء الاصطناعي: {str(e)}"
