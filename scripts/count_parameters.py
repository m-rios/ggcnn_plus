import argparse
import glob
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('path', help='path to the root directory of a model')

args = parser.parse_args()

model_fns = glob.glob(args.path + '/*.hdf5')

for model_fn in model_fns:
    try:
        model = load_model(model_fn)
    except:
        print 'Could not load model {}'.format(model_fn)

    model_name = model_fn.split('/')[-1].replace('.hdf5', '')
    print '{}: {}'.format(model_name, model.count_params())
