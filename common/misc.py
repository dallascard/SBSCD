import os
import unicodedata

def get_model_name(model_name_or_path):
    if len(os.path.split(model_name_or_path)) > 1:
        model_path, model_name = os.path.split(model_name_or_path)
        if len(model_name) == 0:
            model_name = os.path.split(model_path)[-1]
    else:
        model_name = model_name_or_path
    return model_name

    
def get_subdir(basedir, model_name_or_path, strip_accents=False, prefix='tokenized'):
    model_name = get_model_name(model_name_or_path)
    outdir = os.path.join(basedir, prefix + '_' + model_name)
    if strip_accents:
        outdir += '_SA'
    return outdir
    

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
