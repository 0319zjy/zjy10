from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 1. 初始化Flask
app = Flask(__name__)
model = None  # 全局模型

# 2. 简单训练/加载模型（去掉复杂验证）
def init_model():
    global model
    # 读取CSV（强制指定编码，忽略警告）
    df = pd.read_csv('insurance-chinese.csv', encoding='gbk')
    df.columns = df.columns.str.strip()
    
    # 补充BMI列（不管有没有，直接加）
    if 'BMI' not in df.columns:
        df['BMI'] = 21.2
    
    # 筛选列
    df = df[['年龄', '性别', '子女数量', '是否吸烟', '区域', 'BMI', '医疗费用']].dropna()
    
    # 训练模型（固定参数，避免版本问题）
    X = df.drop('医疗费用', axis=1)
    y = df['医疗费用']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 预处理（用旧版sparse=False，兼容所有sklearn）
   preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['年龄', '子女数量', 'BMI']),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['性别', '是否吸烟', '区域'])
    ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    model.fit(X_train, y_train)
    print("模型初始化完成")

# 3. 首页（极简HTML）
@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>医疗费用预测</title>
        <style>
            body { padding: 20px; font-family: Arial; }
            .form-group { margin: 10px 0; }
            input, select { padding: 8px; width: 200px; }
            button { padding: 10px 20px; background: #0088ff; color: white; border: none; cursor: pointer; }
            #result { margin-top: 20px; font-size: 18px; }
        </style>
    </head>
    <body>
        <h1>医疗费用预测</h1>
        <form id="form">
            <div class="form-group">年龄：<input type="number" name="age" min="0" max="120" required></div>
            <div class="form-group">性别：
                <select name="gender" required>
                    <option value="男性">男性</option>
                    <option value="女性">女性</option>
                </select>
            </div>
            <div class="form-group">BMI：<input type="number" name="bmi" step="0.01" min="10" max="50" required></div>
            <div class="form-group">子女数量：<input type="number" name="children" min="0" max="10" required></div>
            <div class="form-group">是否吸烟：
                <select name="smoker" required>
                    <option value="是">是</option>
                    <option value="否">否</option>
                </select>
            </div>
            <div class="form-group">区域：
                <select name="region" required>
                    <option value="东南部">东南部</option>
                    <option value="西南部">西南部</option>
                    <option value="西北部">西北部</option>
                    <option value="东北部">东北部</option>
                </select>
            </div>
            <button type="submit">预测</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById('form').onsubmit = async function(e) {
                e.preventDefault();
                let formData = new FormData(this);
                let res = await fetch('/predict', { method: 'POST', body: formData });
                let data = await res.json();
                document.getElementById('result').innerText = data.success ? '预测费用：' + data.prediction + ' 元' : '错误：' + data.error;
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

# 4. 预测接口（极简逻辑）
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 读取参数
        age = int(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        # 预测
        features = pd.DataFrame({
            '年龄': [age], '性别': [gender], '子女数量': [children],
            '是否吸烟': [smoker], '区域': [region], 'BMI': [bmi]
        })
        pred = round(model.predict(features)[0], 2)
        
        return jsonify({'success': True, 'prediction': pred})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# 5. 启动（禁用重载器，避免信号错误）
if __name__ == "__main__":
    # 先初始化模型
    init_model()
    # 启动Flask（关键：use_reloader=False）
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
