from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn import preprocessing
app=Flask(__name__)
@app.route("/")
def home():
    return "Hello, Flask!"
app = Flask(__name__,static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Home page
@app.route('/')
def index():
   return render_template('index.html')
@app.route("/welcome/")
def welcome():
    return "Welcome to my webpage!"
 

# rendering the upload page to upload files
# The file is in csv or xlx format and saves into local folder named file.csv
# Takes cluster count as an input
@app.route('/uploader', methods=['GET', 'POST']) 
def upload_file():
   print('uploader')
   if request.method == 'POST':
      f = request.Files['File']
      print(f.filename)
      
      no_clusters = request.form['no_clusters']
      f.save(secure_filename(f.filename))
      if '.csv' in f.filename:
          dataset = pd.read_csv(f.filename)
      elif'.xlsx' in f.filename:
          dataset = pd.read_excel(f.filename)
      dataset.to_csv('file.csv')

      Text = preprocessing()
      a = int(no_clusters)
      #evaluation = evaluation_array(Text, a)
      #print('eval:',evaluation, flush=True)
      #dataframe = pd.DataFrame(evaluation, columns =["Name", "Silhoutte Score", "Calinski Score", "Davies Score"])
      #dataframe = dataframe.sort_values(by=['Silhoutte Score'], ascending=False)

      # print("heyyyyyyyyyyy", no_clusters)
      return render_template("upload.html",old = dataset.shape, new = Text.shape,res = "Analysis of uploaded file", clusters= a)
if __name__ == '__main__':
  app.run(host="localhost", port=5000, debug=True)
  