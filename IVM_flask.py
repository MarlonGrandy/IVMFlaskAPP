from flask import Flask, render_template, request
from pipeline import run_pipeline
import os
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', static='')


@app.route('/generate', methods=['POST'])
def generate_text():
    txt1 = request.form['paragraph1']
    txt2 = request.form['paragraph2']
    combined_text = run_pipeline(txt1, txt2)
    return render_template('output.html', text=combined_text)


@app.route('/original_page')
def original_page():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
