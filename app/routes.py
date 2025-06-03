from flask import Blueprint, request, jsonify, render_template
import os

from app.utils.utils import *
from app.routes_folder.upload_handler import *

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')


#Upload Handler datasets and models
@main.route('/upload', methods=['POST'])
def upload():
    return handle_upload()

def process_upload(mode):
    print(f"Processando upload em modo: {mode.upper()}")
    return upload()


@main.route('/upload_evasion', methods=['POST'])
def upload_evasion():
    print("Modo: EVASION")
    return process_upload(mode='evasion')


@main.route('/upload_poisoning', methods=['POST'])
def upload_poisoning():
    print("Modo: POISONING")
    return process_upload(mode='poisoning')

@main.route('/get_attacks', methods=['GET'])
def get_attacks():
    mode = request.args.get('mode')

    if mode == 'evasion':
        attacks = ['FGSM', 'JSMA', 'ZOO']
    elif mode == 'poisoning':
        attacks = ['FGSM', 'PGD']
    else:
        attacks = []

    return jsonify(attacks)
