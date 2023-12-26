"""
    Inference Module

    Provides functions for species classifier inference

    TODO:
    merge load_classifier with classifiers.load_model

    @ Kyra Swanson 2023
"""
from tensorflow.keras.models import load_model
import torch
import pandas as pd
import numpy as np
from os.path import isfile
from time import time
from humanfriendly import format_timespan
from . import generator, file_management
from .classifiers import EfficientNet


def load_classifier(model_file, class_file, device='cpu'):
    """
    Load classifier model
    If .pt file, assumes CTL EfficientNet

    Args
        - model_file (str): path to classifier model
        - class_file (str): path to associated class list
        - device (str): specify to run model on cpu or gpu, default to cpu

    Returns
        model: model object
        classes: dataframe of classes
    """
    if not isfile(model_file):
        raise AssertionError("The given model file does not exist.")
    if not isfile(class_file):
        raise AssertionError("The given class file does not exist.")

    classes = pd.read_csv(class_file)

    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'

    start_time = time()
    # TensorFlow
    if model_file.endswith('.h5'):
        model = load_model(model_file)
    # PyTorch dict
    elif model_file.endswith('.pt'):
        model = EfficientNet(len(classes))
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
    # PyTorch full model
    elif model_file.endswith('.pth'):
        model = torch.load(model_file)
        model.to(device)
        model.eval()
    else:
        raise ValueError('Unrecognized model format: {}'.format(model_file))
    elapsed = time() - start_time
    print('Loaded model in {}'.format(format_timespan(elapsed)))

    return model, classes


def predict_species(detections, model, classes, device='cpu', out_file=None,
                    resize=299, batch=1, workers=1):
    """
    Predict species using classifier model

    Args
        - detections (pd.DataFrame): dataframe of (animal) detections
        - model: preloaded classifier model
        - classes: preloaded class list
        - device (str): specify to run model on cpu or gpu, default to cpu
        - out_file (str): path to save prediction results to
        - resize (int): image input size
        - batch (int): data generator batch size
        - workers (int): number of cores

    Returns
        - detections (pd.DataFrame): MD detections with classifier prediction and confidence
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    if isinstance(detections, pd.DataFrame):
        filecol = "Frame" if "Frame" in detections.columns else "file"

        if any(detections.columns.isin(["bbox1"])):

            # pytorch
            if type(model) == EfficientNet:
                dataset = generator.create_dataloader(detections, batch, workers, filecol)
                with torch.no_grad():
                    for ix, (data, _) in enumerate(dataset):
                        data.to(device)
                        output = model(data)

                        pred = classes['x'].values[torch.argmax(output, 1).numpy()[0]]
                        probs = torch.max(torch.nn.functional.softmax(output, dim=1), 1).values.numpy()[0]

                        detections.loc[ix, 'prediction'] = pred
                        detections.loc[ix, 'confidence'] = probs

            else:  # tensorflow
                dataset = generator.TFGenerator(detections, filecol=filecol, resize=resize, batch=batch)
                output = model.predict(dataset, workers=workers, verbose=1)

                detections['prediction'] = [classes['species'].values[int(np.argmax(x))] for x in output]
                detections['confidence'] = [np.max(x) for x in output]
#         else:
#            raise AssertionError("Model architechture not supported.")
        else:
            raise AssertionError("Input must be a data frame of crops or vector of file names.")
    else:
        raise AssertionError("Input must be a data frame of crops or vector of file names.")

    if out_file:
        file_management.save_data(detections, out_file)

    return detections
