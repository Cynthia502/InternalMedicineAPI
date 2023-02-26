import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import traceback
import json 
import joblib

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

CORS(app, resources={r'/*': {'origins': '*'}})

# 数据处理
# data processingfs
def final_Result(inputcsv):
    # 导入模型
    # import model
    clf=joblib.load("/pred.m")
    # print(inputcsv)
    result=clf.predict([inputcsv])
    return result

@app.route('/data_generate', methods=['POST'])
def data_generate():
  
  if request.method == 'POST':
    
    try:
      post_data = request.get_json()

      # 接收前端传入参数 testtxt search
      # Receive front-end incoming parameters testtxt search
      age = (post_data.get('age'))
      height = float(post_data.get('height'))
      weight = post_data.get('weight')
      active = post_data.get('active')
      alco = post_data.get('alco')
      cholesterol = post_data.get('cholesterol')
      ap_lo = post_data.get('ap_lo')
      gender = post_data.get('gender')
      gluc = post_data.get('gluc')
      smoke = post_data.get('smoke')
      ap_hi = post_data.get('ap_hi')
      
      alldata=[]
      alldata.append(age)
      alldata.append(gender)
      alldata.append(height)
      alldata.append(weight)
      alldata.append(ap_hi)
      alldata.append(ap_lo)
      alldata.append(cholesterol)
      alldata.append(gluc)
      alldata.append(smoke)
      alldata.append(alco)
      alldata.append(active)
      resultdata=final_Result(alldata)
    

      print('result:', resultdata[0])
      # this is the translation of note
      # Predict the received test data with the model to get the result
      # resultdata=final_Result(testtxt)
      # When testing front-end and back-end communication, you can first pass a string to the front-end to ensure communication, and then call the model calculation formally
      # Prepare to return information to the front end

      # 将接收到的测试数据用模型预测 得到结果
      # resultdata=final_Result(testtxt)
      # 测试前后端通信时可先给前端穿个字符串 保证通信 然后在正式调用模型计算
      # 准备返回前端的信息
      message = {'status': 'success','resultdata': str(resultdata[0])}
    
    except Exception as e:
      traceback.print_exc()
      return jsonify(str(e))
    
    else:
      return jsonify(message)





if __name__ == '__main__':
  app.run(port=5000)
