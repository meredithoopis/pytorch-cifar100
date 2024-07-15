# A REPORT ON CHALLENGES

## 1. Challenge 1: Baseline model: VGG19
- Reason for model choice: VGG19 is an advanced CNN with pre-trained layers and a great understanding of what defines an image in terms of shape, color, and structure. VGG19 is very deep and has been trained on millions of diverse images with complex classification tasks. 

- Training: To train the baseline model, run the following: 
```bash
python train.py -net vgg19 
```
- Results: 
    - Top1 error: 0.0019
    - Top5 error: 0.0003 
    - Baseline accuracy: 0.988

## 2. Challenge 2: Improving the baseline model 
- Improving model speed: Quantization reduces the number of bits required to represent the weights and activations of the model. By converting from a higher precision (e.g., 32-bit floating point) to a lower precision (e.g., 8-bit integer), the computational load and memory usage are significantly decreased. This results in faster inference times because the arithmetic operations on lower precision data are quicker and require less power. Additionally, quantized models can take advantage of specialized hardware accelerators designed for low-precision arithmetic, further enhancing speed.

- Improving model performance: Depthwise convolutions involve applying a single convolutional filter per input channel, rather than applying a filter across all input channels as in standard convolutions. This reduces the number of parameters and computational complexity, allowing for deeper networks with more layers or channels within the same computational budget. Depthwise convolutions can capture more fine-grained features and spatial hierarchies in the data, leading to improved performance. When combined with pointwise convolutions (1x1 convolutions), this technique forms depthwise separable convolutions, which further enhance efficiency and performance by reducing the number of parameters while maintaining representational power.

- How to run: The code for quantization can be seen on the Quantizing_model.ipynb file, while to train depthwise convolution, run: 
```bash
python train.py -net vgg19depthwise 
```

- Results: 

| Method                | Top-1 Error | Top-5 Error | Runtime (ms) | Accuracy (%) | Model Size (MB) |
|-----------------------|-------------|-------------|--------------|--------------|-----------------|
| Baseline              | 0.19%       | 0.03%       | 18.11        | 98.8%        | 196.5           |
| Quantization (float16)| 0.19%       | 0.03%       | 16.23        | 98.8%        | 196.5           |
| Quantization (int8)   | 0.19%       | 0.03%       | 16.17        | 98.0%        | 109.3           |
| Depthwise Convolution | 0.21%       | 0.01%       | 13.05        | 99.1%        | 125.6           |

- Analysis: 
    - Baseline: Highest runtime (18.11 ms) and largest model size (196.5 MB).
    - Quantization (float16): Maintains baseline accuracy with a slight runtime improvement (16.23 ms).
    - Quantization (int8): Reduces model size (109.3 MB) and runtime (16.17 ms) but slightly decreases accuracy (98.0%).
    - Depthwise Convolution: Best accuracy (99.1%) and fastest runtime (13.05 ms) with moderate model size (125.6 MB).

- Conclusion:
    - Quantization (int8): Best for size and speed with minor accuracy trade-off.
    - Depthwise Convolution: Best overall performance with highest accuracy and fastest runtime.

## 3. Challenge 3: 
- I didn't change the architecture of the model, instead I defined a mapping from label to corrsponding Cangjie character. 
For inference with the depthwise convolution model, run: 
```bash
python inference.py -net vgg19depthwise -weights one_of_the_weights_file 
```
- The training and testing file can be found on: train3.py file and test3.py file 

## Additional details: 
All models can be found on this link: [Models](https://drive.google.com/drive/folders/1POlhdIlJTAnJR3HOjLbakfmV6Xhea8mB?usp=drive_link)
