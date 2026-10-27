import os
from dotenv import load_dotenv
from groq import Groq

# تحميل المتغيرات من ملف .env المحلي
load_dotenv()

# استدعاء الـ API Key بطريقة آمنة
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ تنبيه: لم يتم العثور على مفتاح GROQ_API_KEY في ملف .env!")

# تهيئة عميل Groq
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
            # استخدام موديل قوي ودقيق للتقارير الطبية
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600  # تقييد عدد الستوكينز لتوفير الاستهلاك وسرعة الاستجابة
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"عذراً، حدث خطأ أثناء توليد التقرير بالذكاء الاصطناعي: {str(e)}"







# import os
# from groq import Groq

# # ضع مفتاح الـ Groq الخاص بك أو اجعله في Environment Variable
# client = Groq(api_key="ضع_مفتاح_الـ_Groq_هنا")

# def generate_medical_report(age, gender, symptoms, model_prediction, confidence):
#     prompt = f"""
#     أنت طبيب استشاري متخصص في الأشعة والأمراض الصدرية. قم بإعداد تقرير طبي احترافي بناءً على البيانات التالية:
#     - عمر المريض: {age} سنة
#     - النوع: {gender}
#     - الأعراض التي يشتكي منها: {symptoms}
#     - نتيجة الذكاء الاصطناعي لتحليل الأشعة: {model_prediction} (نسبة الثقة: {confidence:.2f}%)
    
#     قم بتضمين الآتي في التقرير (باللغة العربية وبصياغة طبية احترافية):
#     1. تحليل مبدئي للحالة وربط الأعراض بالسن والنتيجة المتوقعة.
#     2. تفسير سبب الإصابة المحتمل (هل هو مرتبط بعدوى، مناعة، أو عوامل عمرية).
#     3. الخطة العلاجية المقترحة المبدئية (للإرشاد العام وليست وصفة نهائية).
#     4. نصائح الراحة المنزلية والمشروبات المفيدة (مثل المشروبات الدافئة، الأطعمة الداعمة للمناعة).
#     5. متى يجب التوجه للطوارئ فوراً؟
#     """
    
#     try:
#         completion = client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=1000
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"عذراً، حدث خطأ أثناء توليد التقرير بالذكاء الاصطناعي: {str(e)}"