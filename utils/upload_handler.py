import os
from flask import request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads/'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def handle_upload():
    test_file = request.files.get('test_dataset')

    model_files = request.files.getlist('model[]')

    if test_file and model_files and test_file.filename != '':
        test_path = os.path.join(UPLOAD_FOLDER, secure_filename(test_file.filename))
        test_file.save(test_path)

        model_paths = []
        for mf in model_files:
            if mf.filename != '':
                model_path = os.path.join(UPLOAD_FOLDER, secure_filename(mf.filename))
                mf.save(model_path)
                model_paths.append(model_path)

        return None, test_path, model_paths
    else:
        return None, None, []
