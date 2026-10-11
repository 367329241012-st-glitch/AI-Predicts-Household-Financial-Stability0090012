import os

# ให้ Qt (ที่ Orange เรียกใช้) ทำงานแบบไม่มีหน้าจอบนเซิร์ฟเวอร์
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pickle
import numpy as np
import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ระบบวิเคราะห์ความมั่นคงทางการเงิน",
    page_icon="💰",
    layout="wide"
)

# Custom CSS ตกแต่งสไตล์เพิ่มเติม
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1E293B;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    div[data-testid="stForm"] {
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ใช้แบบจำลอง Neural Network จากสถานการณ์ S3 (ข้อมูลดิบ + credit_score, 16 ตัวแปร)
# ไฟล์นี้ได้จาก Save Model ใน orenge_S3.ows — ภายในมีขั้นตอน Continuize + Normalize อยู่แล้ว
# จึงส่งตัวเลขดิบเข้าไปได้เลย ไม่ต้องปรับสเกลเอง
MODEL_PATH = "models/S3_nn_model.pkcls"
MODEL_INFO = "Neural Network · ชุดตัวแปร S3 (16 ตัวแปร) · ความถูกต้องบนชุดทดสอบ 96.3%"


# โหลดโมเดลที่บันทึกจาก Orange (Save Model -> .pkcls)
@st.cache_resource
def load_model():
    import Orange  # noqa: F401
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = None
try:
    model = load_model()
except FileNotFoundError:
    st.error(f"❌ ไม่พบไฟล์โมเดลที่ {MODEL_PATH}")
    st.info("💡 กรุณาตรวจสอบว่ามีไฟล์ S3_nn_model.pkcls อยู่ในโฟลเดอร์ models ของโปรเจกต์")
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")

# Mappings จากภาษาไทย -> ค่าภาษาอังกฤษที่โมเดล Orange รู้จัก
GENDER_MAP = {"ชาย": "Male", "หญิง": "Female", "อื่นๆ": "Other"}
EDUCATION_MAP = {"มัธยมศึกษา": "High School", "ปริญญาตรี": "Bachelor", "ปริญญาโท": "Master", "ปริญญาเอก": "PhD", "อื่นๆ": "Other"}
EMPLOYMENT_MAP = {"ทำงานประจำ": "Employed", "ธุรกิจส่วนตัว / รับจ้างอิสระ": "Self-employed", "นักเรียน / นักศึกษา": "Student", "ว่างงาน": "Unemployed"}
JOB_MAP = {
    "พนักงานขาย": "Salesperson", "ครู / อาจารย์": "Teacher", "ผู้จัดการ": "Manager",
    "นักเรียน / นักศึกษา": "Student", "นักบัญชี": "Accountant", "แพทย์ / บุคลากรทางการแพทย์": "Doctor",
    "ว่างงาน": "Unemployed", "พนักงานขับรถ": "Driver", "วิศวกร": "Engineer"
}
REGION_MAP = {"เอเชีย": "Asia", "ยุโรป": "Europe", "อเมริกาเหนือ": "North America", "แอฟริกา": "Africa", "อื่นๆ": "Other"}
HAS_LOAN_MAP = {"ไม่มี": "No", "มี": "Yes"}
LOAN_TYPE_MAP = {"การศึกษา": "Education", "ธุรกิจ": "Business", "ยานพาหนะ": "Car", "ที่อยู่อาศัย": "Home"}

DEFAULT_SELECT = "กรุณากรอกข้อมูล"


def get_model_domain(m):
    dom = getattr(m, "original_domain", None)
    return dom if dom is not None else m.domain


def build_orange_table(m, input_dict):
    from Orange.data import Domain, Table

    attrs = get_model_domain(m).attributes
    row = []
    missing_cols = []
    unknown_values = []

    for var in attrs:
        val = input_dict.get(var.name)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing_cols.append(var.name)
            row.append(np.nan)
            continue

        if var.is_discrete:
            text = str(val)
            if text in var.values:
                row.append(float(var.values.index(text)))
            else:
                unknown_values.append(f"{var.name} = {text}")
                row.append(np.nan)
        else:
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                unknown_values.append(f"{var.name} = {val}")
                row.append(np.nan)

    table = Table.from_numpy(Domain(attrs), np.array([row], dtype=float))
    return table, missing_cols, unknown_values


def get_class_info(val_str):
    val_lower = val_str.lower()
    if "unstable" in val_lower:
        return {
            'name': 'ไม่มั่นคง (Unstable)',
            'color': '#dc2626',
            'bg_color': '#fef2f2',
            'badge': '🔴 ไม่มั่นคง',
            'desc': 'มีความเสี่ยงทางการเงินสูง ควรปรับแผนรายจ่าย เร่งลดภาระหนี้สิน และสร้างเงินออมฉุกเฉิน'
        }
    elif "stable" in val_lower:
        return {
            'name': 'มั่นคง (Stable)',
            'color': '#16a34a',
            'bg_color': '#f0fdf4',
            'badge': '🟢 มั่นคง',
            'desc': 'สภาพคล่องทางการเงินแข็งแกร่ง มีภาระหนี้สินในระดับเหมาะสมและมีสัดส่วนเงินออมที่ดี'
        }
    else:
        return {
            'name': 'ปานกลาง (Moderate)',
            'color': '#d97706',
            'bg_color': '#fffbeb',
            'badge': '🟡 ปานกลาง',
            'desc': 'สภาพคล่องทางการเงินอยู่ในระดับปานกลาง ควรควบคุมการใช้จ่ายและเพิ่มสัดส่วนการออมฉุกเฉิน'
        }


# Header
st.markdown("<h1 class='main-title'>💰 ต้นแบบประเมินระดับความมั่นคงทางการเงินส่วนบุคคล</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>กรอกเฉพาะตัวเลขดิบ 16 รายการ ระบบคำนวณอัตราส่วนทางการเงินให้เอง · แบบจำลองสอนจากข้อมูลสังเคราะห์ ใช้เพื่อการศึกษา</p>", unsafe_allow_html=True)

# ฟอร์มกรอกข้อมูลแบ่ง 3 คอลัมน์
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("👤 ข้อมูลทั่วไปและอาชีพ")
        age = st.number_input("1. อายุ (ปี)", min_value=18, max_value=100, value=None, placeholder="กรุณากรอกข้อมูล")
        gender_th = st.selectbox("2. เพศ", [DEFAULT_SELECT, "ชาย", "หญิง", "อื่นๆ"])
        education_th = st.selectbox("3. ระดับการศึกษา", [DEFAULT_SELECT, "มัธยมศึกษา", "ปริญญาตรี", "ปริญญาโท", "ปริญญาเอก", "อื่นๆ"])
        employment_th = st.selectbox("4. สถานะการทำงาน", [DEFAULT_SELECT, "ทำงานประจำ", "ธุรกิจส่วนตัว / รับจ้างอิสระ", "นักเรียน / นักศึกษา", "ว่างงาน"])
        job_th = st.selectbox("5. อาชีพ", [
            DEFAULT_SELECT, "พนักงานขาย", "ครู / อาจารย์", "ผู้จัดการ", "นักเรียน / นักศึกษา",
            "นักบัญชี", "แพทย์ / บุคลากรทางการแพทย์", "ว่างงาน", "พนักงานขับรถ", "วิศวกร"
        ])
        region_th = st.selectbox("6. ภูมิภาค", [DEFAULT_SELECT, "เอเชีย", "ยุโรป", "อเมริกาเหนือ", "แอฟริกา", "อื่นๆ"])

with col2:
    with st.container(border=True):
        st.subheader("💵 รายได้ รายจ่าย และเครดิต")
        monthly_income = st.number_input("7. รายได้ต่อเดือน (USD)", min_value=0.0, step=100.0, value=None, placeholder="กรุณากรอกข้อมูล")
        monthly_expenses = st.number_input("8. รายจ่ายต่อเดือน (USD)", min_value=0.0, step=50.0, value=None, placeholder="กรุณากรอกข้อมูล")
        savings = st.number_input("9. เงินออมสะสม (USD)", min_value=0.0, step=500.0, value=None, placeholder="กรุณากรอกข้อมูล")
        credit_score = st.number_input("10. คะแนนเครดิต (Credit Score 300-850)", min_value=300, max_value=850, step=1, value=None, placeholder="กรุณากรอกข้อมูล")

with col3:
    with st.container(border=True):
        st.subheader("💳 ข้อมูลสินเชื่อและหนี้สิน")
        has_loan_th = st.selectbox("11. มีการกู้สินเชื่อหรือไม่", [DEFAULT_SELECT, "ไม่มี", "มี"])

        if has_loan_th == "มี":
            loan_type_th = st.selectbox("12. ประเภทสินเชื่อ", [DEFAULT_SELECT, "การศึกษา", "ธุรกิจ", "ยานพาหนะ", "ที่อยู่อาศัย"])
            loan_amount = st.number_input("13. จำนวนเงินกู้ (USD)", min_value=0.0, step=1000.0, value=None, placeholder="กรุณากรอกข้อมูล")
            loan_term = st.number_input("14. ระยะเวลากู้ (เดือน)", min_value=1, max_value=360, step=1, value=None, placeholder="กรุณากรอกข้อมูล")
            monthly_emi = st.number_input("15. ค่างวดผ่อนชำระต่อเดือน (USD)", min_value=0.0, step=50.0, value=None, placeholder="กรุณากรอกข้อมูล")
            loan_rate = st.number_input("16. อัตราดอกเบี้ย (%)", min_value=0.0, max_value=30.0, step=0.1, value=None, placeholder="กรุณากรอกข้อมูล")
        else:
            # ในข้อมูลที่ใช้สอนโมเดล ครัวเรือนที่ไม่มีหนี้มี loan_type = "None" และตัวเลขหนี้เป็น 0
            loan_type_th = DEFAULT_SELECT
            loan_amount = 0.0
            loan_term = 0
            monthly_emi = 0.0
            loan_rate = 0.0
            st.caption("ไม่มีหนี้ — ระบบจะใช้ยอดกู้ ค่างวด และดอกเบี้ยเป็น 0")

st.markdown("<br>", unsafe_allow_html=True)

# ปุ่มทำนายผล
if st.button("🔍 ประมวลผลและทำนายความมั่นคงทางการเงิน", type="primary", use_container_width=True):
    # ตรวจสอบการกรอกข้อมูล
    required_selects = [gender_th, education_th, employment_th, job_th, region_th, has_loan_th]
    required_numbers = [age, monthly_income, monthly_expenses, savings, credit_score]
    if has_loan_th == "มี":
        required_selects.append(loan_type_th)
        required_numbers += [loan_amount, loan_term, monthly_emi, loan_rate]

    if DEFAULT_SELECT in required_selects or any(v is None for v in required_numbers):
        st.warning("⚠️ กรุณากรอกข้อมูลให้ครบถ้วนก่อนทำการประมวลผล")
    elif monthly_income <= 0:
        st.warning("⚠️ รายได้ต่อเดือนต้องมากกว่า 0")
    else:
        # แปลงค่าจาก TH -> EN เพื่อนำเข้า Model (16 ตัวแปรของชุด S3)
        input_dict = {
            'age': age,
            'gender': GENDER_MAP.get(gender_th),
            'education_level': EDUCATION_MAP.get(education_th),
            'employment_status': EMPLOYMENT_MAP.get(employment_th),
            'job_title': JOB_MAP.get(job_th),
            'monthly_income_usd': monthly_income,
            'monthly_expenses_usd': monthly_expenses,
            'savings_usd': savings,
            'has_loan': HAS_LOAN_MAP.get(has_loan_th),
            'loan_type': LOAN_TYPE_MAP.get(loan_type_th) if has_loan_th == "มี" else "None",
            'loan_amount_usd': loan_amount,
            'loan_term_months': loan_term,
            'monthly_emi_usd': monthly_emi,
            'loan_interest_rate_pct': loan_rate,
            'credit_score': credit_score,
            'region': REGION_MAP.get(region_th)
        }

        if model is None:
            st.error("ยังโหลดโมเดลไม่สำเร็จ จึงยังทำนายผลไม่ได้")
        else:
            try:
                table, missing_cols, unknown_values = build_orange_table(model, input_dict)
                if missing_cols or unknown_values:
                    raise ValueError("ข้อมูลไม่ตรงกับที่โมเดลรู้จัก: " + ", ".join(missing_cols + unknown_values))

                values, probs = model(table, model.ValueProbs)
                class_var = get_model_domain(model).class_var
                prediction = class_var.values[int(values[0])]

                st.markdown("---")
                st.subheader("📊 ผลการทำนายความมั่นคงทางการเงิน")

                info = get_class_info(prediction)

                # 1. แสดง Card ผลการทำนายหลักแบบโดดเด่น
                st.markdown(f"""
                    <div style="
                        background-color: {info['bg_color']};
                        border-left: 6px solid {info['color']};
                        border-radius: 10px;
                        padding: 18px 24px;
                        margin-bottom: 25px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    ">
                        <h3 style="color: {info['color']}; margin: 0 0 8px 0; font-size: 22px; font-weight: 700;">
                            {info['badge']} — {info['name']}
                        </h3>
                        <p style="color: #374151; margin: 0; font-size: 15px; line-height: 1.6;">
                            <strong>คำแนะนำ:</strong> {info['desc']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # 2. แสดงสัดส่วนความน่าจะเป็นแบบ Visual (Progress Bars) ภาษาไทย
                st.write("### 📈 สัดส่วนความน่าจะเป็นของแต่ละระดับ")

                class_names = list(class_var.values)
                prob_values = probs[0] * 100

                for c_name, p_val in zip(class_names, prob_values):
                    c_info = get_class_info(c_name)
                    col_text, col_bar, col_val = st.columns([3, 6, 1.5])

                    with col_text:
                        st.markdown(f"**{c_info['badge']} {c_info['name']}**")

                    with col_bar:
                        st.progress(float(min(max(p_val / 100.0, 0.0), 1.0)))

                    with col_val:
                        st.markdown(f"**{p_val:.2f}%**")

                # 3. อัตราส่วนทางการเงินที่ระบบคำนวณให้ (เพื่ออธิบายผล ไม่ได้ส่งเข้าโมเดล)
                st.write("### 🧮 อัตราส่วนทางการเงินของคุณ (ระบบคำนวณให้)")
                surplus_ratio = (monthly_income - monthly_expenses) / monthly_income
                savings_ratio = savings / (monthly_income * 12)
                dti_ratio = monthly_emi / monthly_income
                m1, m2, m3 = st.columns(3)
                m1.metric("เงินคงเหลือต่อรายได้", f"{surplus_ratio * 100:.1f}%")
                m2.metric("เงินออมต่อรายได้ทั้งปี", f"{savings_ratio:.2f} เท่า")
                m3.metric("ค่างวดหนี้ต่อรายได้ (DTI)", f"{dti_ratio * 100:.1f}%")

                st.caption(
                    f"แบบจำลอง: {MODEL_INFO} · สอนจากข้อมูลการเงินส่วนบุคคลแบบสังเคราะห์ 32,424 รายการ "
                    "ผลลัพธ์ใช้เพื่อการศึกษา ไม่ใช่คำแนะนำทางการเงิน"
                )

            except Exception as err:
                st.error(f"เกิดข้อผิดพลาดขณะทำนายผล: {err}")
