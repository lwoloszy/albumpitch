from flask import render_template, session, redirect, url_for, current_app
from flask import request
from .. import db
from . import main

import gen_recc as gr

@main.route('/', methods=['GET'])
def index():
    aq = request.args.get('aq', '22108-eleven-into-fifteen-a-130701-compilation')

    results = gr.get_20_albums()
    results = [(result[0], result[1], result[2], 1, 1) for result in results]

    #print(aq)
    #print(gr.gen_reccs_from_album(db, aq))
    return render_template('index.html', album_list=results)
