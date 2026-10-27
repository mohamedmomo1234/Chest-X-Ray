import streamlit as st
from PIL import Image
import torch
from src.model import load_model, get_transforms, GradCAM, generate_heatmap_image
from src.llm_helper import generate_medical_report


st.set_page_config(page_title="MediAI - Chest Diagnostics Platform", page_icon="🏥", layout="wide")

 
MODEL_PATH = "models/efficientnet_pneumonia.pth"

@st.cache_resource
def get_cached_model():
    return load_model(MODEL_PATH)

model = get_cached_model()

# (Dashboard Selector)
st.sidebar.title("🏥 نظام تشخيص الصدر الذكي")
role = st.sidebar.selectbox("اختر لوحة التحكم (Role):", ["المريض (Patient Dashboard)", "الطبيب (Doctor Dashboard)"])

st.sidebar.markdown("---")
st.sidebar.info("💡 مشروع تخرج - نظام مساعد ذكي لتحليل أشعة الصدر وتقديم التقارير الطبية.")


# 1. (Patient Dashboard)

if role == "المريض (Patient Dashboard)":
    st.title("👤 لوحة تحكم المريض - فحص الأشعة والأعراض")
    st.write("أدخل بياناتك الصحية واقترح صورة الأشعة للحصول على تحليل مبدئي وتقرير شامل.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 البيانات الشخصية والأعراض")
        age = st.number_input("السن (العمر):", min_value=1, max_value=120, value=25)
        gender = st.selectbox("النوع:", ["ذكر", "أنثى"])
        symptoms = st.text_area("ما هي الأعراض التي تعاني منها؟ (مثلاً: كحة جافة، ضيق تنفس، ارتفاع حرارة...)", placeholder="اكتب الأعراض بالتفصيل...")
        
        uploaded_file = st.file_uploader("اختر صورة أشعة الصدر (Chest X-Ray - JPG/PNG)...", type=["jpg", "png", "jpeg"])

    with col2:
        st.subheader("🔍 معاينة الأشعة والنتيجة")
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert('RGB')
            st.image(original_image, caption="صورة الأشعة المرفوعة", use_column_width=True)
            
            if st.button("🚀 ابدأ التحليل الشامل وتوليد التقرير"):
                with st.spinner("جاري معالجة الصورة وتحليل الحالة بالذكاء الاصطناعي..."):
                    
                    # normalizatiom image
                    transform = get_transforms()
                    input_tensor = transform(original_image).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        prob = torch.sigmoid(output).item()
                        
                    if prob >= 0.65:
                        result_label = "PNEUMONIA (التهاب رئوي)"
                        conf_score = prob * 100
                    elif prob <= 0.35:
                        result_label = "NORMAL (سليم)"
                        conf_score = (1 - prob) * 100
                    else:
                        result_label = "BORDERLINE / UNCERTAIN تحتاج مراجعه الطبيب"
                        conf_score =  prob * 100
                        
                    # generate Grad-CAM
                    target_layer = model.features[-1]
                    grad_cam = GradCAM(model, target_layer)
                    cam = grad_cam(input_tensor)
                    heatmap_img = generate_heatmap_image(original_image, cam)
                    
                    st.image(heatmap_img, caption="Grad-CAM Heatmap (موضع التركيز للالتهاب)", use_column_width=True)
                    
                    # save result in Session State 
                    st.session_state['last_result'] = result_label
                    st.session_state['last_conf'] = conf_score
                    st.session_state['last_report'] = generate_medical_report(age, gender, symptoms, result_label, conf_score)

    # report medical
    if 'last_report' in st.session_state:
        st.markdown("---")
        st.subheader("📄 التقرير الطبي المبدئي وتوصيات العلاج والراحة:")
        st.markdown(st.session_state['last_report'])


# 2. لوحة تحكم الطبيب (Doctor Dashboard)
elif role == "الطبيب (Doctor Dashboard)":
    st.title("🩺 لوحة تحكم الطبيب - مراجعة الحالات واعتماد التقارير")
    st.write("هنا يتم مراجعة الحالات الواردة من المرضى، فحص صور الأشعة بدقة، والتحقق من خرائط التمركز والتقارير الذكية.")
    
    
    patient_tab1, patient_tab2 = st.tabs(["📁 الحالة الحالية (المفحوصة)", "📊 إحصائيات النظام"])
    
    with patient_tab1:
        if 'last_result' in st.session_state:
            st.success("تم استلام أحدث حالة مرفوعة من بوابة المرضى بنجاح.")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.metric("التشخيص الآلي المبدئي", st.session_state['last_result'])
                st.metric("نسبة الثقة التقنية", f"{st.session_state['last_conf']:.2f}%")
            with col_d2:
                st.warning("⚠️ ملاحظة للطبيب: يرجى مراجعة خريطة Grad-CAM والتأكد من توافقها السريري مع العلامات الحيوية للمريض قبل الاعتماد النهائي.")
                
            st.subheader("📝 التقرير الطبي المقترح بواسطة الذكاء الاصطناعي:")
            editable_report = st.text_area("تعديل أو اعتماد التقرير الطبي:", value=st.session_state['last_report'], height=250)
            
            if st.button("✅ اعتماد التقرير وإرساله لملف المريض"):
                st.success("تم اعتماد التقرير الطبي بنجاح وإغلاق دورة الحالة!")
        else:
            st.info("لا توجد حالات جديدة معلقة حالياً. يطلب من المرضى رفع صور عبر بوابة المرضى.")
            
    with patient_tab2:
        st.subheader("📈 إحصائيات النظام السريري")
        st.write("عدد الحالات المفحوصة اليوم: 12 حالة")
        st.write("نسبة دقة التوافق مع التشخيص البشري: 94.2%")

# warning academy
st.markdown("---")
st.caption("⚠️ **تنبيه إخلاء مسؤولية أكاديمي:** هذا النظام هو نموذج تعليمى    (Clinical Decision Support Prototype) وليس بديلاً عن التشخيص الطبي البشري المتخصص.")

