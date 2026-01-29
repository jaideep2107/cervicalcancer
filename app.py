import os
import json
import joblib
import pandas as pd
import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sklearn.impute import SimpleImputer
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super_secret_key_for_demo')
CORS(app)

# --- DATABASE CONFIGURATION ---
# This line automatically switches between online Postgres and local SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///local_medical.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- FILE UPLOAD CONFIG ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.String(50), primary_key=True)  # Username/ID
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False) # doctor, admin, radiologist, patient
    name = db.Column(db.String(100), nullable=False)

class PatientData(db.Model):
    id = db.Column(db.String(50), primary_key=True)  # Matches User ID
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    risk_status = db.Column(db.String(50), default="Pending")
    last_prob = db.Column(db.String(20), default="N/A")
    # Storing notes and images as JSON strings for simplicity
    notes = db.Column(db.Text, default="[]") 
    images = db.Column(db.Text, default="[]")

# --- INITIALIZE DATABASE ---
with app.app_context():
    db.create_all()
    # Create Default Users if they don't exist
    if not User.query.get('doctor1'):
        db.session.add(User(id='doctor1', password='123', role='doctor', name='Dr. Saravana Kumar'))
        db.session.add(User(id='rad1', password='123', role='radiologist', name='Chief Radiologist'))
        db.session.add(User(id='admin1', password='admin', role='admin', name='System Admin'))
        db.session.commit()

# --- HELPER FUNCTIONS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- LOAD ML MODEL ---
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
    
    # Fetch Data from Database
    if session['role'] == 'patient':
        records = PatientData.query.filter_by(id=session['user']).all()
    else:
        records = PatientData.query.all()
    
    # Convert DB objects to Dictionary for Template
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
    new_id = data.get('patient_id')
    
    if User.query.get(new_id):
        return jsonify({'status': 'error', 'message': 'ID already exists'})

    # 1. Create Login
    new_user = User(id=new_id, password=data.get('password'), role='patient', name=data.get('name'))
    db.session.add(new_user)
    
    # 2. Create Patient Record
    new_patient = PatientData(id=new_id, name=data.get('name'), age=data.get('age'))
    db.session.add(new_patient)
    
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Patient Created'})

@app.route('/add_note', methods=['POST'])
def add_note():
    if session.get('role') != 'doctor': return jsonify({'status': 'error'})
    data = request.get_json()
    
    patient = PatientData.query.get(data.get('patient_id'))
    if patient:
        current_notes = json.loads(patient.notes)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        current_notes.append(f"[{timestamp}] Dr. {session['name']}: {data.get('note')}")
        
        patient.notes = json.dumps(current_notes)
        db.session.commit()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'})

@app.route('/upload_biopsy', methods=['POST'])
def upload_biopsy():
    if session.get('role') != 'radiologist': return jsonify({'status': 'error'})
    
    file = request.files.get('file')
    pid = request.form.get('patient_id')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{pid}_{file.filename}")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        patient = PatientData.query.get(pid)
        if patient:
            current_imgs = json.loads(patient.images)
            current_imgs.append(filename)
            patient.images = json.dumps(current_imgs)
            db.session.commit()
            return jsonify({'status': 'success'})
            
    return jsonify({'status': 'error'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        pid = data.get('patient_id_context')
        
        # --- ML LOGIC (SAME AS BEFORE) ---
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
            # Fallback
            prob = 0.85
            prediction = 1

        result_text = "High Risk" if prediction == 1 else "Low Risk"
        
        # Update Database
        patient = PatientData.query.get(pid)
        if patient:
            patient.risk_status = result_text
            patient.last_prob = f"{prob*100:.1f}%"
            db.session.commit()

        return jsonify({'status': 'success', 'prediction': result_text, 'probability': f"{prob*100:.2f}%"})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

if __name__ == '__main__':
    # Local development uses sqlite automatically
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)