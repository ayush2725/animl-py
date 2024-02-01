"""
    Inference Module

    Provides functions for species classifier inference

    @ Kyra Swanson 2023
"""
import torch
import pandas as pd
import numpy as np
from scipy.special import softmax
from tqdm import trange
from . import generator, file_management
from .classifiers import EfficientNet, to_numpy


def predict_species(detections, model, classes, device='cpu', out_file=None,
                    file_col='Frame', framework="", resize=299, batch=1, workers=1):
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
        # cropped images
        if any(detections.columns.isin(["bbox1"])):
            # Tensorflow
            if model.framework == "tensorflow":

                dataset = generator.TFGenerator(detections, file_col=file_col, resize=resize, batch=batch)
                output = model.predict(dataset, workers=workers, verbose=1)

                detections['prediction'] = [classes['species'].values[int(np.argmax(x))] for x in output]
                detections['confidence'] = [np.max(x) for x in output]
            # PyTorch or Onnx
            else:
                predictions = []
                probabilities = []

                dataset = generator.create_dataloader(detections, batch_size=batch, workers=workers, file_col=file_col)
                progressBar = trange(len(dataset))

                with torch.no_grad():
                    for ix, batch in enumerate(dataset):
                        data = batch[0]
                        # name = batch[1]

                        # PyTorch
                        if type(model) == EfficientNet:
                            data = data.to(device)
                            output = model(data)
                            pred_label = torch.argmax(output, dim=1).cpu().detach().numpy()
                            predictions.extend(pred_label)

                            probs = torch.max(torch.nn.functional.softmax(output, dim=1), 1)[0]
                            probs = probs.cpu().detach().numpy()
                            probabilities.extend(probs)

                        # ONNX
                        elif model.framework == 'onnx':
                            ort_inputs = {model.get_inputs()[0].name: to_numpy(data)}
                            ort_outs = model.run(None, ort_inputs)[0]
                            pred_label = np.argmax(ort_outs, axis=1)
                            predictions.extend(pred_label)

                            probs = np.max(softmax(ort_outs), axis=1)
                            probabilities.extend(probs)

                        else:
                            raise AssertionError("Model architechture not supported.")

                        progressBar.update(1)

                detections['prediction'] = [classes['species'].values[x] for x in predictions]
                detections['confidence'] = probabilities
                progressBar.close()

        # non-cropped images (not supported)
        else:
            raise AssertionError("Input must be a data frame of crops or vector of file names.")
    else:
        raise AssertionError("Input must be a data frame of crops or vector of file names.")

    if out_file:
        file_management.save_data(detections, out_file)

    return detections
