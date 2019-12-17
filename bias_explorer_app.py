import numpy as np
import os

from argparse import ArgumentParser
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from debiaswe.debias import debias
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


            return redirect(url_for('bias_explorer', filename=filename))

    return '''
    <!doctype html>
    <titleBias Explorer</title>
    <h1>Welcome to Bias Explorer!</h1>
    <h2>Upload new Word Embedding</h2>
    <p>.txt file is recommended</p>
    <form method=post enctype=multipart/form-data>
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


@app.route('/bias_explorer/<filename>', methods=['GET', 'POST'])
def bias_explorer(filename):
    if request.method == 'POST':
        fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        embedding = WordEmbedding(fp)
        
        rep_word_str = request.form['rep_word_str']
        rep_words = rep_word_str.split(', ')
        
        v_protected = model.compute_bias_direction(embedding, rep_words)
        
        if request.form['submit_button'] == 'Analogies':
            # Analogies based on the protected direction
            a_protected = embedding.best_analogies_dist_thresh(v_protected)
            print(a_protected)

            return '''
            <!doctype html>
            <title>Bias Explorer</title>
            <h1>Analogies</h1> 
            <p>{a_protected}</p>
            '''.format(a_protected=a_protected)
    
        if request.form['submit_button'] == 'Bias Scores':
            wordset1, wordset2 = model.compute_bias_scores(embedding, v_protected)

            return '''
            <!doctype html>
            <title>Bias Scores</title>
            <h1>Welcome to the Bias Scores Page!</h1>
            <p>{wordset1}</p>
            <p>{wordset2}</p>
            '''.format(wordset1=wordset1,
                       wordset2=wordset2)
        
        if request.form['submit_button'] == 'Debiasing':
            de = debias(embedding, 
                        request.form['specific_words'].split(', '), 
                        request.form['equalize_pairs'].split(', '),
                        v_protected)
            
            de.save(os.path.join(app.config['UPLOAD_FOLDER'], 'debiased.txt'))
            
            return '''
            <!doctype html>
            <title>Debiased Embedding</title>
            <h1>Your embedding has been debiased</h1>
            <form action="/uploads/{filename}" method="POST">
            <input type="hidden" name="param1" value="value1">
            <input type="submit" name="submit_button" value="Download Me">
            </form>
            '''.format(filename=filename)       
    
    return '''
        <!doctype html>
        <title>Bias Explorer</title>
        <form method=post enctype=multipart/form-data>
          <h2>Representative Words</h2>
          <p>(Required for all options)</p>
          <p>Input words that represent your area of potential bias</p>
          <p>For example: for gender, ['he', 'she'].</p>
          <input type="text" name="rep_word_str" value="he, she">
          <h2>Words of Interest</h2>
          <p>(Required for Bias Scores)</p>
          <input type="text" name="word_list" value="software engineer, detail-oriented, expert" size=150>
          <h2>Debiasing Inputs</h2>
          <p>Dimension-Specific Words</p>
          <input type="text" name="specific_words" value="husband, wife" size=150>
          <p>Definitional Words</p>
          <input type="text" name="definitional_words" value="woman, man" size=150>
          <p>Equalize Words</p>
          <input type="text" name="equalize_pairs" value="monastery, convent" size=150>
          <p></p>
          <input type="submit" name="submit_button" value="Analogies">
          <input type="submit" name="submit_button" value="Bias Scores">
          <input type="submit" name="submit_button" value="Debiasing">
        </form>
        '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(directory=app.config['UPLOAD_FOLDER'], 
                               filename=filename, 
                               as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)