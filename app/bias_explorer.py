import numpy as np
import os

from argparse import ArgumentParser
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
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

    return render_template('home.html')


@app.route('/bias_explorer/<filename>', methods=['GET', 'POST'])
def bias_explorer(filename):
    if request.method == 'POST':
        fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        embedding = WordEmbedding(fp)
        
        rep_word_str = request.form['rep_word_str']
        rep_words = rep_word_str.split(', ')
        
        v_protected = model.compute_bias_direction(embedding, rep_words)
        
        if request.form['submit_button'] == 'Get Analogies':
            # Analogies based on the protected direction
            a_protected = embedding.best_analogies_dist_thresh(v_protected)
            print(a_protected)
            
            return render_template('analogies.html', a_protected=a_protected)

    
        if request.form['submit_button'] == 'Get Bias Scores':
            wordset1, wordset2 = model.compute_bias_scores(embedding, v_protected)

            return render_template('bias_scores.html', 
                                   wordset1=wordset1, 
                                   wordset2=wordset2)
        
        if request.form['submit_button'] == 'Debias Embedding':
            de = debias(embedding, 
                        request.form['specific_words'].split(', '), 
                        request.form['equalize_pairs'].split(', '),
                        v_protected)
            
            de.save(os.path.join(app.config['UPLOAD_FOLDER'], 'debiased.txt'))
            
            return render_template('debiasing.html', filename=filename)       
    
    return render_template('exploration_page.html')

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