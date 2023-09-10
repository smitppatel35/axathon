from flask import Flask, render_template, request
import json

from Helper import predict

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.context_processor
def list_processor():
    def extract_list(str):
        return str['items']

    return dict(extract_list=extract_list)


@app.route('/')
def index():
    # helper.print_df()
    formList = gen_form()['fields']
    return render_template('index.html', formList=formList)


@app.route('/', methods=['POST'])
def prediction():
    _form_data = request.form.to_dict()
    result = predict(_form_data)

    if result == 0:
        return "Not a Fraud 🫡"
    else:
        return "Fraud!!! 😢"


def gen_form():
    f = open('static/form.json')
    return json.load(f)


if __name__ == '__main__':
    app.run(debug=True)
