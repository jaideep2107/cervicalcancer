import os
import json
import joblib
import pandas as pd
import datetime
import re
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sklearn.impute import SimpleImputer
from werkzeug.utils import secure_filename

app = Flask(__name__)
# In production, change this key!
app.secret_key = os.environ.get('SECRET_KEY', 'your_secure_random_key_here')
CORS(app)

# --- DATABASE CONFIG ---
# Automatically switches between Render (Postgres) and Local (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///medical_system.db')
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- CONFIG ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.String(50), primary_key=True)  # Username/PatientID
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False) # doctor, admin, radiologist, patient
    name = db.Column(db.String(100), nullable=False)

class PatientData(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    risk_status = db.Column(db.String(50), default="Pending")
    last_prob = db.Column(db.String(20), default="N/A")
    notes = db.Column(db.Text, default="[]") 
    images = db.Column(db.Text, default="[]")

# --- INITIAL SETUP ---
with app.app_context():
    db.create_all()
    # Create Admin/Doctor accounts if they don't exist
    if not User.query.get('admin1'):
        db.session.add(User(id='admin1', password='Admin@123', role='admin', name='System Administrator'))
        db.session.add(User(id='doctor1', password='Doctor@123', role='doctor', name='Dr. Saravana Kumar'))
        db.session.add(User(id='rad1', password='Rad@123', role='radiologist', name='Chief Radiologist'))
        db.session.commit()
        print("✓ System initialized with default Admin/Doctor accounts.")

# --- VALIDATION LOGIC ---
def validate_registration(data):
    # 1. Name: Only Alphabets and spaces
    if not re.match(r"^[A-Za-z\s]+$", data.get('name', '')):
        return False, "Name must contain only alphabets."
    
    # 2. Patient ID: Alphabets and Numbers only
    if not re.match(r"^[A-Za-z0-9]+$", data.get('patient_id', '')):
        return False, "Patient ID must contain only letters and numbers."
    
    # 3. Password: Min 8, Max 16, 1 Upper, 1 Number, 1 Special Char
    pwd = data.get('password', '')
    if len(pwd) < 8 or len(pwd) > 16:
        return False, "Password must be 8-16 characters long."
    if not re.search(r"[A-Z]", pwd):
        return False, "Password must contain at least one capital letter."
    if not re.search(r"\d", pwd):
        return False, "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", pwd):
        return False, "Password must contain at least one special character."
        
    return True, "Valid"

# --- ML LOADING ---
try:
    model = joblib.load('model_assets/cervical_cancer_model.pkl')
    scaler = joblib.load('model_assets/scaler.pkl')
    selected_features = joblib.load('model_assets/selected_features.pkl')
    feature_names = joblib.load('model_assets/feature_names.pkl')
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    feature_names = ['Age', 'Smokes (years)', 'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)']

# --- ROUTES ---

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    user = User.query.get(username)

    if user and user.password == password:
        session['user'] = user.id
        session['role'] = user.role
        session['name'] = user.name
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html', error="Invalid Credentials")

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('login_page'))
    
    if session['role'] == 'patient':
        records = PatientData.query.filter_by(id=session['user']).all()
    else:
        records = PatientData.query.all()
    
    patients_dict = {}
    for p in records:
        patients_dict[p.id] = {
            'name': p.name,
            'age': p.age,
            'risk_status': p.risk_status,
            'notes': json.loads(p.notes),
            'images': json.loads(p.images)
        }
    return render_template('dashboard.html', user=session, patients=patients_dict)

@app.route('/create_patient', methods=['POST'])
def create_patient():
    if session.get('role') not in ['admin', 'doctor']:
        return jsonify({'status': 'error', 'message': 'Unauthorized'})

    data = request.get_json()
    
    # Run Strict Validation
    is_valid, error_msg = validate_registration(data)
    if not is_valid:
        return jsonify({'status': 'error', 'message': error_msg})
    
    new_id = data.get('patient_id')
    if User.query.get(new_id):
        return jsonify({'status': 'error', 'message': 'Patient ID already exists'})

    # Create User & Patient Record
    try:
        new_user = User(id=new_id, password=data.get('password'), role='patient', name=data.get('name'))
        new_patient = PatientData(id=new_id, name=data.get('name'), age=data.get('age'))
        
        db.session.add(new_user)
        db.session.add(new_patient)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Patient Profile Created Successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        pid = data.get('patient_id_context')
        
        input_data = {}
        for feat in feature_names:
            val = data.get(feat, 0.0)
            try: input_data[feat] = float(val)
            except: input_data[feat] = 0.0
            
        if MODEL_LOADED:
            df = pd.DataFrame([input_data])
            imputer = SimpleImputer(strategy='mean')
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=feature_names)
            df_scaled = scaler.transform(df_imputed)
            df_selected = df_scaled[:, selected_features]
            prediction = int(model.predict(df_selected)[0])
            prob = float(model.predict_proba(df_selected)[0][1])
        else:
            # Fallback for UI testing
            prob = 0.85
            prediction = 1

        result_text = "High Risk" if prediction == 1 else "Low Risk"
        
        patient = PatientData.query.get(pid)
        if patient:
            patient.risk_status = result_text
            patient.last_prob = f"{prob*100:.1f}%"
            db.session.commit()

        return jsonify({'status': 'success', 'prediction': result_text, 'probability': f"{prob*100:.2f}%"})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --- NOTES & UPLOADS (Keep existing logic) ---
@app.route('/add_note', methods=['POST'])
def add_note():
    if session.get('role') != 'doctor': return jsonify({'status': 'error'})
    data = request.get_json()
    patient = PatientData.query.get(data.get('patient_id'))
    if patient:
        notes = json.loads(patient.notes)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        notes.append(f"[{ts}] {session['name']}: {data.get('note')}")
        patient.notes = json.dumps(notes)
        db.session.commit()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'})

@app.route('/upload_biopsy', methods=['POST'])
def upload_biopsy():
    if session.get('role') != 'radiologist': return jsonify({'status': 'error'})
    file = request.files.get('file')
    pid = request.form.get('patient_id')
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        filename = secure_filename(f"{pid}_{file.filename}")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        patient = PatientData.query.get(pid)
        if patient:
            imgs = json.loads(patient.images)
            imgs.append(filename)
            patient.images = json.dumps(imgs)
            db.session.commit()
            return jsonify({'status': 'success'})
    return jsonify({'status': 'error'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)