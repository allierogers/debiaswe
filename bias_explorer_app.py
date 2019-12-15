import numpy as np
import os

from argparse import ArgumentParser
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from debiaswe.we import WordEmbedding
import model

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.mkdir(app.config['UPLOAD_FOLDER'])
            fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fp)

            rep_word_one = request.form['rep_word_one']
            rep_word_two = request.form['rep_word_two']
            return redirect(url_for('analogies', filename=filename, rep_word_one=rep_word_one, rep_word_two=rep_word_two))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type="file" name="file">
      <p>Representative Words</p>
      <input type="text" name="rep_word_one" value="he">
      <input type="text" name="rep_word_two" value="she">
      <input type="submit" value="Upload">
    </form>
    '''

@app.route('/analogies/<filename>/<rep_word_one>/<rep_word_two>', methods=['GET'])
def analogies(filename, rep_word_one, rep_word_two):
    fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    embedding = WordEmbedding(fp)
    rep_words = [rep_word_one, rep_word_two]
    v_protected = model.compute_bias_direction(rep_words)
    
    # Analogies based on the protected direction
    a_protected = embedding.best_analogies_dist_thresh(v_protected)
    print(a_protected)
    
    return '''
    <!doctype html>
    <title>Analogies</title>
    <p>{a_protected}</p>
    '''.format(a_protected=a_protected)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=True)