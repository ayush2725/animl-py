'''
    This script will take a torch model (.pt file) and produce a new onnx model (.onnx file)
'''
import argparse
import yaml
import onnx
import torch
import numpy as np

from . import classifiers


def load_model(architecture, num_classes):
    '''
        Creates a model instance. 
    '''
    
    if (architecture=="CTL"):
        model_instance = classifiers.CTLClassifier(num_classes)
    elif (architecture=="efficientnet_v2_m"):
        model_instance = classifiers.EfficientNet(num_classes,tune=False)        
    else:
        raise AssertionError('Please provide the correct model')

    return model_instance


def main():
    '''
    Command line function

    Example usage:
    > python torchToOnnx.py --torchModel "/mnt/machinelearning/Models/Andes/Experiment5/30.pt" 
                            --classFile "/mnt/machinelearning/Training Data - Andes/andes_classes.csv" 
                            
                    or 
                   
      python torchToOnnx.py --config "/mnt/machinelearning/Models/Andes/Experiment5/exp5.yaml"
    '''
           
    parser = argparse.ArgumentParser(description='convert torch model to onnx')
    
    #collect the torch model file path 
    parser.add_argument('--torchModel', help='Path to torch model (.pt file)')
    
    #collect the classification labels file path - to build torch model
    parser.add_argument('--classFile', help='Path to classification categories (.csv file)')
    #for architecture
    parser.add_argument('--architecture', help='Model architecture (default = CTL)')
    #for onnx model file name
    parser.add_argument('--onnxFileName', help="desired location for onnx mode (.onnx file)')
     
                       
    #optional: to use parameters from yaml file
    parser.add_argument('--config', help='path to .yaml config file')
      
    args = parser.parse_args() #add the parser arguments 
    
    modelLoaded = False
                        
    #load paths directly from the command line
    if (args.config == None):
        
        device = 'cpu'
        
        batch_size = 1
        image_size = [299,299]
        
        #take torch model path from command line input
        torchModelPath = args.torchModel
                                
        if args.architecture == None: 
            architecture="CTL" #default
        else:
            architecture = args.architecture
                        
        if args.classFile == None: 
            modelTorch = load_model(architecture, num_classe=53)
            pathModelAll = torch.load(args.torchModel, map_location=torch.device(device))
            modelTorch.load_state_dict(pathModelAll["model"])
            modelTorch.eval()
            modelLoaded = True
        else: 
            classFile = args.classFile
                        
                           
    #if yaml file given, take varaibles from that       
    else 
        # load config
        print(f'Using config "{args.config}"')
        cfg = yaml.safe_load(open(args.config, 'r'))
        
        # check if GPU is available
        device = cfg['device']
        if device != 'cpu' and not torch.cuda.is_available():
            print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
            device = 'cpu'
        
        batch_size = cfg["batch_size"]
        image_size = cfg["image_size"]
        
        #take torch model path from yaml config file
        torchModelPath = cfg['active_model']
        classFile = cfg['class_file']
        architecture = cfg['architecture']

                        
    #load the torch model
    if not modelLoaded: 
        modelTorch, classTorch = classifiers.load_model(torchModelPath, classFile, device=device, 
                                                    architecture=architecture, overwrite=False)
        modelLoaded = True
    
    #define the location of the output onnx model
    if args.onnxFileName == None:
        onnxFileName = "onnxModel.onnx"
    else: 
        onnxFileName = args.onnxFileName
                        
    # Input to the model
    x = torch.randn(batch_size, 3, image_size[0], image_size[1], requires_grad=True)
    torch_out = modelTorch(x.to(device))

    # Export the model
    torch.onnx.export(modelTorch,                # model being run
                      x.to(device),              # model input (or a tuple for multiple inputs)
                      onnxFileName,              # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}}
                     )


if __name__ == '__main__':
    main()
