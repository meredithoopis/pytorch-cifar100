import contextlib
import os
import time
import torch
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity


def warm_up(device, model = None,input_shape = None,onnx=False, ort_session = None): 
    """warm up model for base model and onnx model"""
    if onnx: 
        input = torch.randn(*input_shape, dtype=torch.float)  
        ort_input = {ort_session.get_inputs()[0].name: input.detach().cpu().numpy() if input.requires_grad else input.cpu().numpy()}
        for _ in range(10): 
            _ = ort_session.run(None, ort_input)

    else: 
        input = torch.randn(*input_shape, dtype=torch.float).to(device)
        for _ in range(10): 
            _ = model(input)


def evaluate(device, test_loader, model = None, onnx= False, ort_session= None): 
    """Perform evaluations on base model and onnx model"""
    if onnx: 
        correct_1 = 0.0
        correct_5 = 0.0
        #total = 0
        dt = TP(gpu=False)

        warm_up(device,input_shape = (1,1,64,64), onnx=True, ort_session= ort_session)
        
        with torch.no_grad():
            for image, label in tqdm(test_loader, desc = "Evaluating ONNX.."):
                #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))
                ort_input = {ort_session.get_inputs()[0].name: input.detach().cpu().numpy() if input.requires_grad else input.cpu().numpy()}
                with dt:    
                    output = ort_session.run(None, ort_input)
                output = torch.from_numpy(output[0])
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()

                #compute top 5
                correct_5 += correct[:, :5].sum()

                #compute top1
                correct_1 += correct[:, :1].sum()
    
    else: 
        model.eval()
        correct_1 = 0.0
        correct_5 = 0.0
        #total = 0
        dt = TP(gpu=(device=='cuda'))

        warm_up(device, model = model, input_shape = (1,1,64,64))
        
        with torch.no_grad():
            for image, label in tqdm(test_loader, desc = "Evaluating..."):
                #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))
                image, label = image.to(device), label.to(device)
                with dt:    
                    output = model(image)
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()

                #compute top 5
                correct_5 += correct[:, :5].sum()

                #compute top1
                correct_1 += correct[:, :1].sum()
    
    top1_err = (1 - correct_1 / len(test_loader.dataset)).item()
    top5_err = (1 - correct_5 / len(test_loader.dataset)).item()

    return top1_err, top5_err, dt.t / len(test_loader.dataset) * 1e3


class TP(contextlib.ContextDecorator): 
    """Creating a torch profiler""" 
    def __init__(self, t = 0.0, gpu = False): 
        self.t = t  
        self.gpu = gpu 
    def __enter__(self): 
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback): 
        self.dt = self.time() - self.start
        self.t += self.dt

    def time(self): 
        if self.gpu: 
            torch.cuda.synchronize()
        return time.time()

def create_profiler(model, input, device = 'cpu'): 
    model, input = model.to(device), input.to(device)
    #profile_activity = [ProfilerActivity.CPU] if device == 'cpu' else [ProfilerActivity.CUDA] 
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            model(input)
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=15))


def model_size(model):
    """Calculate the size of a PyTorch model in kilobytes."""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    os.remove("temp.p")
    return size / 1e3 