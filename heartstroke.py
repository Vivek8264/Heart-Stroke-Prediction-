import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string
import csv
import os
from datetime import datetime
import webbrowser

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Global variables
model = None
scaler = None
le = LabelEncoder()
data_storage = 'predictions.csv'

def load_and_train_model():
    global model, scaler, le
    # ------------------------------------------
    # STEP 1: Load Dataset
    # ------------------------------------------
    # STEP 1: Load Dataset
    # ------------------------------------------
    data = pd.read_csv("healthcare-dataset-stroke-data.csv")

    # ------------------------------------------
    # STEP 2: Handle Missing Values
    # ------------------------------------------
    # Use non-chained assignment to avoid pandas Copy-on-Write warnings/errors.
    data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

    # ------------------------------------------
    # STEP 3: Encode Categorical Data
    # ------------------------------------------
    le = LabelEncoder()

    categorical_cols = ['gender', 'ever_married', 'work_type', 'smoking_status']
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # ------------------------------------------
    # STEP 4: Feature Selection
    # ------------------------------------------
    X = data[
        [
            'age',
            'gender',
            'hypertension',
            'heart_disease',
            'bmi',
            'avg_glucose_level',
            'smoking_status',
            'work_type',
            'ever_married'
        ]
    ]

    y = data['stroke']

    # ------------------------------------------
    # STEP 5: Train-Test Split
    # ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------
    # STEP 6: Feature Scaling
    # ------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------------------------------------
    # STEP 7: Train ML Model
    # ------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # ------------------------------------------
    # STEP 8: Model Accuracy
    # ------------------------------------------
    y_pred = model.predict(X_test)
    print("\nStroke Model Accuracy:", accuracy_score(y_test, y_pred))

# Load model on startup
load_and_train_model()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CardioSense — Health Risk Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: linear-gradient(135deg, #080e14 0%, #0a1118 50%, #0d1820 100%);
    --surface: #0d1820;
    --card: linear-gradient(145deg, #111d28 0%, #141f2c 100%);
    --border: #1e3448;
    --accent: #00d4aa;
    --accent2: #e05c5c;
    --accent3: #f0a04b;
    --text: #d8eaf5;
    --muted: #5a7a92;
    --white: #ffffff;
    --glow: 0 0 30px rgba(0,212,170,0.15);
    --glow-strong: 0 0 50px rgba(0,212,170,0.3);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html { scroll-behavior: smooth; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Outfit', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
    animation: bgShift 20s ease-in-out infinite alternate;
  }

  /* ── NOISE TEXTURE OVERLAY ── */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0;
  }

  /* ── ANIMATED BG GRID ── */
  body::after {
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(0,212,170,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,170,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none; z-index: 0;
  }

  .wrap {
    position: relative; z-index: 1;
    max-width: 780px;
    margin: 0 auto;
    padding: 48px 24px 96px;
  }

  /* ── HEADER ── */
  header {
    text-align: center;
    margin-bottom: 56px;
    animation: fadeDown 0.7s ease both;
  }

  .logo-mark {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 8px 20px;
    margin-bottom: 28px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--accent);
    text-transform: uppercase;
    animation: float 3s ease-in-out infinite;
  }

  .pulse-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.8s ease-in-out infinite;
  }

  h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(36px, 6vw, 58px);
    line-height: 1.1;
    color: var(--white);
    letter-spacing: -0.5px;
  }

  h1 em {
    font-style: italic;
    color: var(--accent);
  }

  header p {
    margin-top: 16px;
    color: var(--muted);
    font-size: 15px;
    font-weight: 300;
    max-width: 440px;
    margin-inline: auto;
    line-height: 1.7;
  }

  /* ── STEPPER ── */
  .stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin-bottom: 40px;
    animation: fadeUp 0.7s 0.2s ease both;
  }

  .step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    position: relative;
  }

  .step-item:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 18px;
    left: calc(50% + 18px);
    width: calc(100% - 8px);
    height: 1px;
    background: var(--border);
    transition: background 0.4s;
  }

  .step-item.done:not(:last-child)::after { background: var(--accent); }

  .step-circle {
    width: 36px; height: 36px;
    border-radius: 50%;
    border: 2px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: var(--muted);
    transition: all 0.3s;
    background: var(--surface);
  }

  .step-item.active .step-circle {
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 16px rgba(0,212,170,0.3);
  }

  .step-item.done .step-circle {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--bg);
    font-weight: 600;
  }

  .step-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: var(--muted);
    white-space: nowrap;
    text-transform: uppercase;
  }

  .step-item.active .step-label { color: var(--accent); }
  .step-item.done .step-label { color: var(--text); }

  .step-connector { width: 72px; height: 1px; background: var(--border); margin: 0 4px; margin-bottom: 24px; transition: background 0.4s; }
  .step-connector.done { background: var(--accent); }

  /* ── CARD ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 4px 40px rgba(0,0,0,0.4), var(--glow-strong);
    animation: fadeUp 0.6s ease both;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }

  #step1 {
    background: #ffffff;
    color: #111;
    border-color: #d9dde1;
  }

  #step1 .card-title {
    color: #111;
  }

  #step1 .card-sub {
    color: #607088;
  }

  #step1 input,
  #step1 select {
    background: #f7f9fb;
    color: #111;
    border-color: #d1d9e0;
  }

  #step1 label {
    color: #5e6a7a;
  }

  #step2 {
    background: #ffffff;
    color: #111;
    border-color: #d9dde1;
  }

  #step2 .card-title {
    color: #111;
  }

  #step2 .card-sub {
    color: #607088;
  }

  #step2 input,
  #step2 select {
    background: #f7f9fb;
    color: #111;
    border-color: #d1d9e0;
  }

  #step2 label {
    color: #5e6a7a;
  }

  #step2 .toggle-group {
    border-color: #d1d9e0;
    background: #f7f9fb;
  }

  #step2 .toggle-group label {
    background: transparent;
    color: #24344f;
  }

  #step2 .toggle-group input[type=radio]:checked + label {
    background: linear-gradient(135deg, #00d4aa, #0bd6c0);
    color: #ffffff;
    font-weight: 700;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
  }

  .card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 60px rgba(0,0,0,0.5), var(--glow-strong);
  }

  .card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: var(--white);
    margin-bottom: 6px;
  }

  .card-sub {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 32px;
    font-weight: 300;
  }

  /* ── FORM GRID ── */
  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  .form-grid .full { grid-column: 1 / -1; }

  .field {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
  }

  input, select {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    color: var(--text);
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    appearance: none;
    -webkit-appearance: none;
  }

  input:focus, select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,212,170,0.12);
  }

  input::placeholder { color: var(--muted); }

  select option { background: var(--surface); }

  /* custom select arrow */
  .select-wrap { position: relative; }
  .select-wrap::after {
    content: '▾';
    position: absolute; right: 14px; top: 50%; transform: translateY(-50%);
    color: var(--muted); pointer-events: none; font-size: 13px;
  }

  /* ── TOGGLE BUTTONS ── */
  .toggle-group {
    display: flex;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }

  .toggle-group input[type=radio] { display: none; }

  .toggle-group label {
    flex: 1;
    text-align: center;
    padding: 12px;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.3px;
    text-transform: none;
    color: var(--muted);
    cursor: pointer;
    background: var(--surface);
    transition: all 0.2s;
    border: none;
  }

  .toggle-group input[type=radio]:checked + label {
    background: var(--accent);
    color: var(--bg);
    font-weight: 600;
  }

  /* ── BUTTONS ── */
  .btn-row {
    display: flex;
    gap: 12px;
    justify-content: flex-end;
    margin-top: 36px;
  }

  .btn {
    padding: 13px 32px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.4px;
    transition: all 0.2s;
  }

  .btn-ghost {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
  }

  .btn-ghost:hover { border-color: var(--text); color: var(--text); }

  .btn-primary {
    background: linear-gradient(135deg, var(--accent) 0%, #00f5c4 100%);
    color: var(--bg);
    box-shadow: 0 4px 20px rgba(0,212,170,0.3);
    transition: all 0.3s ease;
  }

  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,212,170,0.5);
    background: linear-gradient(135deg, #00f5c4 0%, var(--accent) 100%);
  }

  .btn-primary:active { transform: translateY(0); }

  /* ── RESULTS ── */
  .result-section {
    display: none;
    animation: fadeUp 0.6s ease both;
  }

  .result-section.visible { display: block; }

  .result-header {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 28px;
  }

  .result-icon {
    width: 52px; height: 52px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
  }

  .icon-safe { background: rgba(0,212,170,0.ṇ15); }
  .icon-warn { background: rgba(240,160,75,0.15); }
  .icon-danger { background: rgba(224,92,92,0.15); }

  .result-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    margin-bottom: 4px;
    font-family: 'DM Mono', monospace;
  }

  .result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 24px;
    color: var(--white);
  }

  /* ── RISK METER ── */
  .meter-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
  }

  .meter-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 14px;
    font-family: 'DM Mono', monospace;
  }

  .meter-bar-bg {
    height: 8px;
    border-radius: 100px;
    background: var(--border);
    overflow: visible;
    position: relative;
    margin-bottom: 12px;
  }

  .meter-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s cubic-bezier(.22,.68,0,1.2);
    position: relative;
    background: linear-gradient(90deg, var(--accent) 0%, #00f5c4 100%);
  }

  .meter-bar-fill::after {
    content: '';
    position: absolute;
    right: -1px; top: 50%;
    transform: translateY(-50%);
    width: 14px; height: 14px;
    border-radius: 50%;
    background: inherit;
    box-shadow: 0 0 12px currentColor;
  }

  .fill-safe   { background: linear-gradient(90deg, var(--accent) 0%, #00f5c4 100%); }
  .fill-warn   { background: linear-gradient(90deg, var(--accent3) 0%, #ffb347 100%); }
  .fill-danger { background: linear-gradient(90deg, var(--accent2) 0%, #ff6b6b 100%); }

  .meter-score {
    display: flex; justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--muted);
  }

  .meter-score strong { font-size: 28px; color: var(--white); font-family: 'DM Serif Display', serif; }

  /* ── RESULT CARDS ── */
  .result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 24px;
  }

  .result-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
  }

  .result-block.span2 { grid-column: 1 / -1; }

  .rb-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    margin-bottom: 6px;
  }

  .rb-value {
    font-size: 18px;
    font-weight: 600;
    color: var(--white);
  }

  .rb-desc {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.6;
    margin-top: 8px;
  }

  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 6px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
  }

  .badge-safe   { background: rgba(0,212,170,0.15); color: var(--accent); }
  .badge-warn   { background: rgba(240,160,75,0.15); color: var(--accent3); }
  .badge-danger { background: rgba(224,92,92,0.15); color: var(--accent2); }

  .divider {
    height: 1px;
    background: var(--border);
    margin: 28px 0;
  }

  .restart-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 36px;
    padding: 14px;
    border: 1px dashed var(--border);
    border-radius: 12px;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    width: 100%;
    transition: all 0.2s;
  }

  .restart-btn:hover { border-color: var(--accent); color: var(--accent); }

  /* ── DISCLAIMER ── */
  .disclaimer {
    margin-top: 40px;
    padding: 18px 22px;
    background: rgba(224,92,92,0.06);
    border: 1px solid rgba(224,92,92,0.2);
    border-radius: 12px;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.7;
    display: flex;
    gap: 12px;
    align-items: flex-start;
  }

  .disclaimer span { font-size: 16px; flex-shrink: 0; }

  /* ── SECTION SEPARATOR ── */
  .section-divider {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 28px 0 24px;
  }

  .section-divider span {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    white-space: nowrap;
    font-family: 'DM Mono', monospace;
  }

  .section-divider::before, .section-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ── ANIMATIONS ── */
  @keyframes fadeDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.5); opacity: 0.6; }
  }

  @keyframes bgShift {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
  }

  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }

  /* ── HIDDEN ── */
  .hidden { display: none !important; }

  /* ── RESPONSIVE ── */
  @media (max-width: 560px) {
    .card { padding: 24px; }
    .form-grid { grid-template-columns: 1fr; }
    .form-grid .full { grid-column: 1; }
    .result-grid { grid-template-columns: 1fr; }
    .result-block.span2 { grid-column: 1; }
    .step-label { display: none; }
    .step-connector { width: 40px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <!-- HEADER -->
  <header>
    <div class="logo-mark"><div class="pulse-dot"></div> CardioSense AI</div>
    <h1>Know Your<br><em>Heart Risk</em></h1>
    <p>A two-step clinical screening tool for stroke and heart attack risk assessment based on your personal health data.</p>
  </header>

  <!-- STEPPER -->
  <div class="stepper" id="stepper">
    <div class="step-item active" id="s1">
      <div class="step-circle">01</div>
      <div class="step-label">Basic Info</div>
    </div>
    <div class="step-connector" id="sc1"></div>
    <div class="step-item" id="s2">
      <div class="step-circle">02</div>
      <div class="step-label">Heart Details</div>
    </div>
    <div class="step-connector" id="sc2"></div>
    <div class="step-item" id="s3">
      <div class="step-circle">03</div>
      <div class="step-label">Results</div>
    </div>
  </div>

  <!-- ── STEP 1: BASIC INFO ── -->
  <div class="card" id="step1">
    <div class="card-title">Patient Basic Details</div>
    <div class="card-sub">Used to assess stroke risk factors</div>

    <div class="form-grid">
      <div class="field">
        <label>Age</label>
        <input type="number" id="age" placeholder="e.g. 45" min="1" max="120"/>
      </div>
      <div class="field">
        <label>Gender</label>
        <div class="toggle-group">
          <input type="radio" name="gender" id="gm" value="1"/>
          <label for="gm">Male</label>
          <input type="radio" name="gender" id="gf" value="0" checked/>
          <label for="gf">Female</label>
        </div>
      </div>

      <div class="field">
        <label>BMI</label>
        <input type="number" id="bmi" placeholder="e.g. 24.5" step="0.1"/>
      </div>
      <div class="field">
        <label>Average Glucose Level (mg/dL)</label>
        <input type="number" id="glucose" placeholder="e.g. 90" step="0.1"/>
      </div>

      <div class="field">
        <label>Hypertension</label>
        <div class="toggle-group">
          <input type="radio" name="hyper" id="hy1" value="1"/>
          <label for="hy1">Yes</label>
          <input type="radio" name="hyper" id="hy0" value="0" checked/>
          <label for="hy0">No</label>
        </div>
      </div>
      <div class="field">
        <label>Heart Disease</label>
        <div class="toggle-group">
          <input type="radio" name="heartd" id="hd1" value="1"/>
          <label for="hd1">Yes</label>
          <input type="radio" name="heartd" id="hd0" value="0" checked/>
          <label for="hd0">No</label>
        </div>
      </div>

      <div class="field">
        <label>Smoking Status</label>
        <div class="select-wrap">
          <select id="smoking">
            <option value="0">Never Smoked</option>
            <option value="1">Formerly Smoked</option>
            <option value="2">Currently Smokes</option>
          </select>
        </div>
      </div>
      <div class="field">
        <label>Work Type</label>
        <div class="select-wrap">
          <select id="work">
            <option value="0">Private</option>
            <option value="1">Government</option>
            <option value="2">Self-employed</option>
          </select>
        </div>
      </div>

      <div class="field full">
        <label>Marital Status</label>
        <div class="toggle-group">
          <input type="radio" name="married" id="mr1" value="1"/>
          <label for="mr1">Married</label>
          <input type="radio" name="married" id="mr0" value="0" checked/>
          <label for="mr0">Not Married</label>
        </div>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn btn-primary" onclick="goStep2()">Continue →</button>
    </div>
  </div>

  <!-- ── STEP 2: HEART ATTACK DETAILS ── -->
  <div class="card hidden" id="step2">
    <div class="card-title">Heart Attack Risk Factors</div>
    <div class="card-sub">Additional clinical indicators for comprehensive assessment</div>

    <div class="form-grid">
      <div class="field">
        <label>Systolic BP (mmHg)</label>
        <input type="number" id="bp_sys" placeholder="e.g. 120"/>
      </div>
      <div class="field">
        <label>Diastolic BP (mmHg)</label>
        <input type="number" id="bp_dia" placeholder="e.g. 80"/>
      </div>

      <div class="field">
        <label>Blood Sugar Level (mg/dL)</label>
        <input type="number" id="blood_sugar" placeholder="e.g. 100"/>
      </div>
      <div class="field">
        <label>Cholesterol Level (mg/dL)</label>
        <input type="number" id="cholesterol" placeholder="e.g. 180"/>
      </div>

      <div class="field full">
        <label>Heart Rate (bpm)</label>
        <input type="number" id="heart_rate" placeholder="e.g. 72"/>
      </div>
    </div>

    <div class="section-divider"><span>Symptoms & History</span></div>

    <div class="form-grid">
      <div class="field">
        <label>Chest Pain (left side)</label>
        <div class="toggle-group">
          <input type="radio" name="chest" id="ch1" value="1"/>
          <label for="ch1">Yes</label>
          <input type="radio" name="chest" id="ch0" value="0" checked/>
          <label for="ch0">No</label>
        </div>
      </div>
      <div class="field">
        <label>Shortness of Breath</label>
        <div class="toggle-group">
          <input type="radio" name="breath" id="br1" value="1"/>
          <label for="br1">Yes</label>
          <input type="radio" name="breath" id="br0" value="0" checked/>
          <label for="br0">No</label>
        </div>
      </div>
      <div class="field">
        <label>Cold Sweats</label>
        <div class="toggle-group">
          <input type="radio" name="sweat" id="sw1" value="1"/>
          <label for="sw1">Yes</label>
          <input type="radio" name="sweat" id="sw0" value="0" checked/>
          <label for="sw0">No</label>
        </div>
      </div>
      <div class="field">
        <label>Nausea / Dizziness</label>
        <div class="toggle-group">
          <input type="radio" name="nausea" id="na1" value="1"/>
          <label for="na1">Yes</label>
          <input type="radio" name="nausea" id="na0" value="0" checked/>
          <label for="na0">No</label>
        </div>
      </div>
      <div class="field">
        <label>High Stress / Anger</label>
        <div class="toggle-group">
          <input type="radio" name="stress" id="st1" value="1"/>
          <label for="st1">Yes</label>
          <input type="radio" name="stress" id="st0" value="0" checked/>
          <label for="st0">No</label>
        </div>
      </div>
      <div class="field">
        <label>Family History</label>
        <div class="toggle-group">
          <input type="radio" name="family" id="fh1" value="1"/>
          <label for="fh1">Yes</label>
          <input type="radio" name="family" id="fh0" value="0" checked/>
          <label for="fh0">No</label>
        </div>
      </div>
      <div class="field full">
        <label>Mental Health Issues</label>
        <div class="toggle-group">
          <input type="radio" name="mental" id="mh1" value="1"/>
          <label for="mh1">Yes</label>
          <input type="radio" name="mental" id="mh0" value="0" checked/>
          <label for="mh0">No</label>
        </div>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost" onclick="goStep1()">← Back</button>
      <button id="predict-btn" class="btn btn-primary" onclick="runPrediction()">Analyse Risk ⟶</button>
    </div>
  </div>

  <!-- ── STEP 3: RESULTS ── -->
  <div class="result-section hidden" id="step3">

    <div class="card">
      <div class="card-title">Risk Assessment Report</div>
      <div class="card-sub" id="res-timestamp"></div>

      <!-- STROKE RESULT -->
      <div class="result-header">
        <div class="result-icon" id="stroke-icon"></div>
        <div>
          <div class="result-label">Stroke Risk (ML Estimate)</div>
          <div class="result-title" id="stroke-title"></div>
        </div>
      </div>

      <div class="meter-wrap">
        <div class="meter-label">Stroke Risk Score</div>
        <div class="meter-bar-bg">
          <div class="meter-bar-fill" id="stroke-bar" style="width:0%"></div>
        </div>
        <div class="meter-score">
          <strong id="stroke-score-val"></strong>
          <span id="stroke-score-label"></span>
        </div>
      </div>

      <div class="divider"></div>

      <!-- HEART ATTACK RESULT -->
      <div class="result-header">
        <div class="result-icon" id="ha-icon"></div>
        <div>
          <div class="result-label">Heart Attack Risk (Rule-Based)</div>
          <div class="result-title" id="ha-title"></div>
        </div>
      </div>

      <div class="meter-wrap">
        <div class="meter-label">Risk Score (out of 13)</div>
        <div class="meter-bar-bg">
          <div class="meter-bar-fill" id="ha-bar" style="width:0%"></div>
        </div>
        <div class="meter-score">
          <strong id="ha-score-val"></strong>
          <span id="ha-score-label"></span>
        </div>
      </div>

      <!-- SUMMARY GRID -->
      <div class="result-grid" id="summary-grid"></div>

      <!-- DISCLAIMER -->
      <div class="disclaimer">
        <span>⚕️</span>
        <div>This tool provides an <strong>indicative screening estimate only</strong>. Stroke prediction uses the full ML model from your Python backend. Heart attack scoring uses the same rule-based algorithm as your code. <strong>Always consult a qualified medical professional for diagnosis and treatment.</strong></div>
      </div>

      <button class="restart-btn" onclick="restart()">↺ &nbsp; Start a New Assessment</button>
    </div>

  </div><!-- /result-section -->

</div><!-- /wrap -->

<script>
  function getRadio(name) {
    const el = document.querySelector(`input[name="${name}"]:checked`);
    return el ? parseInt(el.value) : 0;
  }

  function goStep1() {
    document.getElementById('step2').classList.add('hidden');
    document.getElementById('step1').classList.remove('hidden');
    document.getElementById('step3').classList.add('hidden');
    document.querySelector('.result-section').classList.remove('visible');
    setStep(1);
  }

  function goStep2() {
    const age = parseFloat(document.getElementById('age').value);
    const bmi = parseFloat(document.getElementById('bmi').value);
    const glucose = parseFloat(document.getElementById('glucose').value);

    if (!age || age < 1 || age > 120) { alert('Please enter a valid age.'); return; }
    if (!bmi || bmi < 5 || bmi > 80) { alert('Please enter a valid BMI.'); return; }
    if (!glucose || glucose < 50) { alert('Please enter a valid average glucose level.'); return; }

    document.getElementById('step1').classList.add('hidden');
    document.getElementById('step2').classList.remove('hidden');
    setStep(2);
  }

  function setStep(n) {
    [1,2,3].forEach(i => {
      const el = document.getElementById('s'+i);
      el.classList.remove('active','done');
      if (i < n) el.classList.add('done');
      else if (i === n) el.classList.add('active');
    });
    [1,2].forEach(i => {
      const el = document.getElementById('sc'+i);
      el.classList.toggle('done', i < n);
    });
  }

  async function runPrediction() {
    const predictBtn = document.getElementById('predict-btn');
    if (!predictBtn) return;
    if (predictBtn.disabled) return; // prevent double click

    predictBtn.disabled = true;
    predictBtn.textContent = 'Analyzing ...';

    const bp_sys = parseFloat(document.getElementById('bp_sys').value);
    const bp_dia = parseFloat(document.getElementById('bp_dia').value);
    const blood_sugar = parseFloat(document.getElementById('blood_sugar').value);
    const cholesterol = parseFloat(document.getElementById('cholesterol').value);
    const heart_rate = parseInt(document.getElementById('heart_rate').value);

    if (!bp_sys || !bp_dia) { alert('Please enter blood pressure values.'); predictBtn.disabled = false; predictBtn.textContent = 'Analyse Risk ⟶'; return; }
    if (!blood_sugar) { alert('Please enter blood sugar level.'); predictBtn.disabled = false; predictBtn.textContent = 'Analyse Risk ⟶'; return; }
    if (!cholesterol) { alert('Please enter cholesterol level.'); predictBtn.disabled = false; predictBtn.textContent = 'Analyse Risk ⟶'; return; }
    if (!heart_rate) { alert('Please enter heart rate.'); predictBtn.disabled = false; predictBtn.textContent = 'Analyse Risk ⟶'; return; }

    // Collect all data
    const data = {
      age: parseFloat(document.getElementById('age').value),
      gender: getRadio('gender'),
      bmi: parseFloat(document.getElementById('bmi').value),
      glucose: parseFloat(document.getElementById('glucose').value),
      hyper: getRadio('hyper'),
      heartd: getRadio('heartd'),
      smoking: parseInt(document.getElementById('smoking').value),
      work: parseInt(document.getElementById('work').value),
      married: getRadio('married'),
      bp_sys: bp_sys,
      bp_dia: bp_dia,
      blood_sugar: blood_sugar,
      cholesterol: cholesterol,
      heart_rate: heart_rate,
      stress: getRadio('stress'),
      family: getRadio('family'),
      mental: getRadio('mental'),
      chest: getRadio('chest'),
      breath: getRadio('breath'),
      sweat: getRadio('sweat'),
      nausea: getRadio('nausea')
    };

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const result = await response.json();

      // Show results
      document.getElementById('step2').classList.add('hidden');
      const rs = document.getElementById('step3');
      rs.classList.remove('hidden');
      rs.classList.add('visible');
      setStep(3);

      document.getElementById('res-timestamp').textContent = result.timestamp;

      // Stroke
      const siEl = document.getElementById('stroke-icon');
      const stEl = document.getElementById('stroke-title');
      const sbEl = document.getElementById('stroke-bar');
      const svEl = document.getElementById('stroke-score-val');
      const slEl = document.getElementById('stroke-score-label');

      sbEl.className = 'meter-bar-fill';
      if (result.stroke_high) {
        siEl.innerHTML = '⚠️'; siEl.className = 'result-icon icon-danger';
        stEl.textContent = 'High Risk of Stroke';
        sbEl.classList.add('fill-danger');
      } else if (result.stroke_mid) {
        siEl.innerHTML = '⚡'; siEl.className = 'result-icon icon-warn';
        stEl.textContent = 'Moderate Risk of Stroke';
        sbEl.classList.add('fill-warn');
      } else {
        siEl.innerHTML = '✅'; siEl.className = 'result-icon icon-safe';
        stEl.textContent = 'Low Risk of Stroke';
        sbEl.classList.add('fill-safe');
      }
      svEl.textContent = result.stroke_score + '/10';
      slEl.textContent = result.stroke_high ? 'High Risk' : result.stroke_mid ? 'Moderate Risk' : 'Low Risk';
      setTimeout(() => { sbEl.style.width = result.stroke_width + '%'; }, 200);

      // Heart Attack
      const hiEl = document.getElementById('ha-icon');
      const htEl = document.getElementById('ha-title');
      const hbEl = document.getElementById('ha-bar');
      const hvEl = document.getElementById('ha-score-val');
      const hlEl = document.getElementById('ha-score-label');

      hbEl.className = 'meter-bar-fill';
      if (result.ha_high) {
        hiEl.innerHTML = '🚨'; hiEl.className = 'result-icon icon-danger';
        htEl.textContent = 'High Risk — Seek Medical Attention';
        hbEl.classList.add('fill-danger');
      } else if (result.ha_mid) {
        hiEl.innerHTML = '⚠️'; hiEl.className = 'result-icon icon-warn';
        htEl.textContent = 'Moderate Risk — Consult a Doctor';
        hbEl.classList.add('fill-warn');
      } else {
        hiEl.innerHTML = '✅'; hiEl.className = 'result-icon icon-safe';
        htEl.textContent = 'Low Risk — Maintain Healthy Habits';
        hbEl.classList.add('fill-safe');
      }
      hvEl.textContent = result.ha_score + '/13';
      hlEl.textContent = result.ha_high ? 'High Risk' : result.ha_mid ? 'Moderate Risk' : 'Low Risk';
      setTimeout(() => { hbEl.style.width = result.ha_width + '%'; }, 300);

      // Summary grid
      document.getElementById('summary-grid').innerHTML = result.summary_grid;

      setTimeout(() => { rs.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 100);
    } catch (error) {
      const predictBtn = document.getElementById('predict-btn');
      if (predictBtn) {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Analyse Risk ⟶';
      }
      alert('Error running prediction: ' + error.message);
    }
  }

  function restart() {
    document.getElementById('step3').classList.add('hidden');
    document.querySelector('.result-section').classList.remove('visible');
    document.getElementById('step1').classList.remove('hidden');
    // Reset form
    document.querySelectorAll('input[type=number]').forEach(i => i.value = '');
    document.querySelectorAll('input[type=radio][value="0"]').forEach(r => r.checked = true);
    document.querySelectorAll('select').forEach(s => s.selectedIndex = 0);
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.textContent = 'Analyse Risk ⟶';
    }
    setStep(1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return {'error': 'No data provided'}, 400

    # Extract data
    age = float(data['age'])
    gender = int(data['gender'])
    bmi = float(data['bmi'])
    glucose = float(data['glucose'])
    hypertension = int(data['hyper'])
    heart_disease = int(data['heartd'])
    smoking = int(data['smoking'])
    work = int(data['work'])
    married = int(data['married'])

    bp_sys = float(data['bp_sys'])
    bp_dia = float(data['bp_dia'])
    blood_sugar = float(data['blood_sugar'])
    cholesterol = float(data['cholesterol'])
    heart_rate = int(data['heart_rate'])

    stress = int(data['stress'])
    family_history = int(data['family'])
    mental_health = int(data['mental'])
    chest_pain = int(data['chest'])
    breath_short = int(data['breath'])
    cold_sweat = int(data['sweat'])
    nausea = int(data['nausea'])

    # Stroke prediction using ML model
    user_data = np.array([[age, gender, hypertension, heart_disease, bmi, glucose, smoking, work, married]])
    user_data_scaled = scaler.transform(user_data)
    stroke_pred = model.predict(user_data_scaled)[0]
    stroke_result = "HIGH RISK" if stroke_pred == 1 else "LOW RISK"

    # Stroke risk score heuristic (for display, but ML is used for prediction)
    sScore = 0
    if age > 65: sScore += 3
    elif age > 55: sScore += 2
    elif age > 45: sScore += 1
    if hypertension == 1: sScore += 2
    if heart_disease == 1: sScore += 2
    if glucose > 200: sScore += 2
    elif glucose > 140: sScore += 1
    if bmi > 35: sScore += 1
    if smoking == 2: sScore += 1
    sScore = min(sScore, 10)

    # Heart attack score
    risk_score = 0
    if age > 45: risk_score += 1
    if stress == 1: risk_score += 1
    if bp_sys >= 140 or bp_dia >= 90: risk_score += 1
    if blood_sugar >= 140: risk_score += 1
    if cholesterol >= 240: risk_score += 1
    if heart_rate > 100: risk_score += 1
    if family_history == 1: risk_score += 1
    if mental_health == 1: risk_score += 1
    if chest_pain == 1: risk_score += 2
    if breath_short == 1: risk_score += 1
    if cold_sweat == 1: risk_score += 1
    if nausea == 1: risk_score += 1

    # Determine results
    stroke_high = sScore >= 6
    stroke_mid = sScore >= 3
    ha_high = risk_score >= 7
    ha_mid = risk_score >= 4

    stroke_score = f"{sScore}/10"
    stroke_width = (sScore / 10) * 100

    ha_score = f"{risk_score}/13"
    ha_width = (risk_score / 13) * 100

    # Summary grid
    stroke_badge = 'danger' if stroke_high else 'warn' if stroke_mid else 'safe'
    ha_badge = 'danger' if ha_high else 'warn' if ha_mid else 'safe'
    bmi_cat = 'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'
    bp_status = 'Elevated' if bp_sys >= 140 or bp_dia >= 90 else 'Normal'
    glucose_status = 'High' if glucose > 200 else 'Borderline' if glucose > 140 else 'Normal'

    summary_grid = f"""
      <div class="result-block">
        <div class="rb-label">Age</div>
        <div class="rb-value">{age} yrs</div>
        <div class="rb-desc">{'Above the primary risk threshold (45+)' if age > 45 else 'Below the primary risk threshold'}</div>
      </div>
      <div class="result-block">
        <div class="rb-label">BMI</div>
        <div class="rb-value">{bmi}</div>
        <div class="rb-desc">Category: <strong>{bmi_cat}</strong></div>
      </div>
      <div class="result-block">
        <div class="rb-label">Blood Pressure</div>
        <div class="rb-value">{bp_sys}/{bp_dia} mmHg</div>
        <div class="rb-desc">Status: <strong>{bp_status}</strong></div>
      </div>
      <div class="result-block">
        <div class="rb-label">Avg Glucose</div>
        <div class="rb-value">{glucose} mg/dL</div>
        <div class="rb-desc">Status: <strong>{glucose_status}</strong></div>
      </div>
      <div class="result-block">
        <div class="rb-label">Stroke Risk</div>
        <div class="rb-value">{sScore}/10</div>
        <span class="badge badge-{stroke_badge}">{'High' if stroke_high else 'Moderate' if stroke_mid else 'Low'}</span>
      </div>
      <div class="result-block">
        <div class="rb-label">Heart Attack Risk</div>
        <div class="rb-value">{risk_score}/13</div>
        <span class="badge badge-{ha_badge}">{'High' if ha_high else 'Moderate' if ha_mid else 'Low'}</span>
      </div>
    """

    from datetime import datetime
    timestamp = datetime.now().strftime("Generated %B %d, %Y at %I:%M %p")

    # Store the data
    with open(data_storage, 'a', newline='') as f:
        writer = csv.writer(f)
        if not os.path.exists(data_storage) or os.stat(data_storage).st_size == 0:
            writer.writerow(['Timestamp', 'Age', 'Stroke Risk', 'Heart Attack Risk'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), age, stroke_result, f"{'High' if ha_high else 'Moderate' if ha_mid else 'Low'} Risk"])

    return {
        'timestamp': timestamp,
        'stroke_high': stroke_high,
        'stroke_mid': stroke_mid,
        'stroke_score': sScore,
        'stroke_width': stroke_width,
        'ha_high': ha_high,
        'ha_mid': ha_mid,
        'ha_score': risk_score,
        'ha_width': ha_width,
        'summary_grid': summary_grid
    }

if __name__ == '__main__':
    # Open the browser in a new tab
    webbrowser.open_new_tab('http://127.0.0.1:5000')
    app.run(debug=True)