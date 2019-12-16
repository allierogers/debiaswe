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


            return redirect(url_for('analogies', filename=filename))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

@app.route('/analogies/<filename>', methods=['GET', 'POST'])
def analogies(filename):
    fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    embedding = WordEmbedding(fp)
    
    if (request.method == 'POST') & (request.form.get('rep_word_one') is not None):
        rep_word_one = request.form['rep_word_one']
        rep_word_two = request.form['rep_word_two']
        rep_words = [rep_word_one, rep_word_two]
        v_protected = model.compute_bias_direction(embedding, rep_words)

        # Analogies based on the protected direction
        a_protected = embedding.best_analogies_dist_thresh(v_protected)
        print(a_protected)

        return '''
        <!doctype html>
        <title>Analogies</title>
        <p>{a_protected}</p>
        '''.format(a_protected=a_protected)
    
    if (request.method == 'POST') & (request.form.get('word_list') is not None):
        v_protected = model.compute_bias_direction(embedding, ['he', 'she'])
        wordset1, wordset2 = model.compute_bias_scores(embedding, v_protected)
        
        return '''
        <!doctype html>
        <title>Bias Scores</title>
        <h1>Welcome to the Bias Scores Page!</h1>
        <p>{wordset1}</p>
        <p>{wordset2}</p>
        '''.format(wordset1=wordset1,
                   wordset2=wordset2)
    
    return '''
        <!doctype html>
        <title>Input Representative Words</title>
        <h1>Input words that represent your area of potential bias</h1>
        <p>For example: for gender, ['he', 'she'].</p>
        <form method=post enctype=multipart/form-data>
          <p>Representative Words</p>
          <input type="text" name="rep_word_one" value="he">
          <input type="text" name="rep_word_two" value="she">
          <input type="submit" value="Get Analogies">
        </form>
        <form method=post enctype=multipart/form-data>
          <h2>Words of Interest</h2>
          <input type="text" name="word_list" value="software engineer, detail-oriented, expert">
          <input type="submit" value="Get Bias Scores">
        </form>
        '''

@app.route('/bias_scores/<filename>', methods=['GET', 'POST'])
def bias_scores(filename):
    if request.method == 'POST':
        fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        embedding = WordEmbedding(fp)
        word_list = request.form['word_list']

        return '''
        <!doctype html>
        <title>Bias Scores</title>
        <h1>Bias scores for key words</h1>
        '''
    
    return '''
        <!doctype html>
        <title>Bias Scores</title>
        <h1>Input words for which you would like to see bias scores.</h1>
        <p>For example, if you plan to use the embedding for a resume ranking task, then I suggest you look at bias scores for terms related to the job posting, like “software engineer”, “detail-oriented”, or “expert.”</p>
        <p>If you are interested in a more general task or set of tasks, you could look at bias scores for general positive and negative words like “good”, “violent”, “beautiful”, and “criminal.” </p>
        <form method=post enctype=multipart/form-data>
          <h2>Words of Interest</h2>
          <input type="text" name="word_list" value="software engineer, detail-oriented, expert">
          <input type="submit" value="Get Bias Scores">
        </form>
        '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=True)