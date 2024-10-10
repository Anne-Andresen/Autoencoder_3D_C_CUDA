#include <autoencoder.h>
#include <loss.h>
#include <utils.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <conv3d.h>

#define NUM_KERNELS 3

typedef struct {
    Conv3D convs[NUM_KERNELS];
} ConvLayer;

typedef struct {
    ConvLayer conv1;
    ConvLayer conv2;
} Encoder;

typedef struct {
    ConvLayer deconv1;
    ConvLayer deconv2;
} Decoder;

typedef struct {
    Encoder encoder;
    Decoder decoder;
} Autoencoder;





#include "conv3d.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv3d_kernel(float* input, float* weights, float* biases, float* output, int D, int H, int W, int kD, int kH, int kW) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        float value = biases[0]; // Assuming a single bias value for simplicity
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int in_d = z - kd + kD / 2;
                    int in_h = y - kh + kH / 2;
                    int in_w = x - kw + kW / 2;
                    if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        value += input[(in_d * H + in_h) * W + in_w] * weights[(kd * kH + kh) * kW + kw];
                    }
                }
            }
        }
        output[(z * H + y) * W + x] = value;
    }
}

void conv3d_init(Conv3D* conv, int inputDepth, int inputHeight, int inputWidth, int kernelD, int kernelH, int kernelW) {
    conv->D = inputDepth;
    conv->H = inputHeight;
    conv->W = inputWidth;
    conv->kernelD = kernelD;
    conv->kernelH = kernelH;
    conv->kernelW = kernelW;
    /*
    conv->weights = (float*)malloc(kernelD * kernelH * kernelW * sizeof(float));
    conv->biases = (float*)malloc(sizeof(float));
    conv->grad_weights = (float*)malloc(kernelD * kernelH * kernelW * sizeof(float));
    conv->grad_biases = (float*)malloc(sizeof(float));
    */
    cudaMalloc(&(conv->weights), kernelD * kernelH * kernelW * sizeof(float));
    cudaMalloc(&(conv->biases), sizeof(float));
    cudaMalloc(&(conv->grad_weights), kernelD * kernelH * kernelW * sizeof(float));
    cudaMalloc(&(conv->grad_biases), sizeof(float));

    // Initialize weights and biases
    float* h_weights = (float*)malloc(kernelD * kernelH * kernelW * sizeof(float));
    for (int i = 0; i < kernelD * kernelH * kernelW; i++) {
        h_weights[i] = (float)rand() / RAND_MAX;
}
cudaMemcpy(conv->weights, h_weights, kernelD * kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice);
free(h_weights);
float h_bias;
h_bias = (float)rand() / RAND_MAX;
cudaMemcpy(conv->biases, &h_bias, sizeof(float), cudaMemcpyHostToDevice);
}

void conv3d_set_input(Conv3D* conv, float* input) {
    conv->input = input;
}
void conv3d_set_kernel(Conv3D* conv, const float* kernelData) {
    size_t kernelSize = conv->kernelD * conv->kernelH * conv->kernelW * sizeof(float);
    cudaMemcpy(conv->weights, kernelData, kernelSize, cudaMemcpyHostToDevice);
}
void conv3d_execute(Conv3D* conv, float* output) {
    conv->output = output;
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((conv->W + blockDim.x - 1) / blockDim.x, (conv->H + blockDim.y - 1) / blockDim.y, (conv->D + blockDim.z - 1) / blockDim.z);
    conv3d_kernel<<<gridDim, blockDim>>>(conv->input, conv->weights, conv->biases, conv->output, conv->D, conv->H, conv->W, conv->kernelD, conv->kernelH, conv->kernelW);
}

void conv3d_backprop(Conv3D* conv, float* d_grad_output, float* d_grad_input) {
    cudaMemset(conv->grad_weights, 0, conv->kernelD * conv->kernelH * conv->kernelW * sizeof(float));
    cudaMemset(conv->grad_biases, 0, sizeof(float));

    dim3 blockDim(8, 8, 8);
    dim3 gridDim(
        (conv->W + blockDim.x - 1) / blockDim.x,
        (conv->H + blockDim.y - 1) / blockDim.y,
        (conv->D + blockDim.z - 1) / blockDim.z
    );

    conv3d_backward_kernel<<<gridDim, blockDim>>>(
        conv->input,
        conv->weights,
        d_grad_output,
        d_grad_input,
        conv->grad_weights,
        conv->grad_biases,
        conv->D,
        conv->H,
        conv->W,
        conv->kernelD,
        conv->kernelH,
        conv->kernelW
    );
    cudaErrorCheck();
}
__global__ void conv3d_backward_kernel(
    const float* input, const float* weights, const float* d_grad_output,
    float* d_grad_input, float* grad_weights, float* grad_biases,
    int D, int H, int W, int kD, int kH, int kW) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; // Width index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Height index
    int z = blockIdx.z * blockDim.z + threadIdx.z; // Depth index

    if (x < W && y < H && z < D) {
        // Compute gradient w.r.t input
        float grad_input = 0.0f;
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int out_d = z + kd - kD / 2;
                    int out_h = y + kh - kH / 2;
                    int out_w = x + kw - kW / 2;
                    if (out_d >= 0 && out_d < D && out_h >= 0 && out_h < H && out_w >= 0 && out_w < W) {
                        float grad_out = d_grad_output[(out_d * H + out_h) * W + out_w];
                        float weight = weights[(kd * kH + kh) * kW + kw];
                        grad_input += grad_out * weight;

                        // Compute gradients w.r.t weights
                        float input_val = input[(z * H + y) * W + x];
                        atomicAdd(&grad_weights[(kd * kH + kh) * kW + kw], grad_out * input_val);
                    }
                }
            }
        }
        d_grad_input[(z * H + y) * W + x] = grad_input;

        // Compute gradients w.r.t biases
        atomicAdd(&grad_biases[0], d_grad_output[(z * H + y) * W + x]);
    }
}


void conv3d_update_weights(Conv3D* conv, float learning_rate) {
    int size = conv->kernelD * conv->kernelH * conv->kernelW;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    update_weights_kernel<<<blocks, threads>>>(conv->weights, conv->grad_weights, learning_rate, size);
    cudaErrorCheck();

    // Update biases
    update_weights_kernel<<<1, 1>>>(conv->biases, conv->grad_biases, learning_rate, 1);
    cudaErrorCheck();
}

void conv3d_free(Conv3D* conv) {
    cudaFree(conv->weights);
    cudaFree(conv->biases);
    cudaFree(conv->grad_weights);
    cudaFree(conv->grad_biases);
}

__global__ void relu_activation_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}
__global__ void relu_backward_kernel(const float* d_output, const float* activation_input, float* d_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_input[idx] = activation_input[idx] > 0 ? d_output[idx] : 0.0f;
    }
}

/*
// After conv3d_forward_kernel
int size = conv->D * conv->H * conv->W;
int threads = 256;
int blocks = (size + threads - 1) / threads;
relu_activation_kernel<<<blocks, threads>>>(d_temp_output, size);
cudaErrorCheck();

*/


float mean_square_error(const float* d_output, const float* d_target, int size) {
    float mse = 0.0f;
    float* h_output = (float*)malloc(size * sizeof(float));
    float* h_target = (float*)malloc(size * sizeof(float));
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_target, d_target, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        float diff = h_output[i] - h_target[i];
        mse += diff * diff;
    }
    mse /= size;

    free(h_output);
    free(h_target);

    return mse;
}

#define cudaErrorCheck()  do { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


void init_conv_layer(ConvLayer* layer, int inputDepth, int inputHeight, int inputWidth, int kernelSize) {
    for (int i = 0; i < NUM_KERNELS; i++) {
        conv3d_init(&layer->convs[i], inputDepth, inputHeight, inputWidth, kernelSize, kernelSize, kernelSize);
    }
}


void init_encoder(Encoder* encoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize) {
    init_conv_layer(&encoder->conv1, inputDepth, inputHeight, inputWidth, kernelSize);
    init_conv_layer(&encoder->conv2, inputDepth, inputHeight, inputWidth, kernelSize);
}

void init_decoder(Decoder* decoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize) {
    init_conv_layer(&decoder->deconv1, inputDepth, inputHeight, inputWidth, kernelSize);
    init_conv_layer(&decoder->deconv2, inputDepth, inputHeight, inputWidth, kernelSize);
}

void init_autoencoder(Autoencoder* autoencoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize) {
    init_encoder(&autoencoder->encoder, inputDepth, inputHeight, inputWidth, kernelSize);
    init_decoder(&autoencoder->decoder, inputDepth, inputHeight, inputWidth, kernelSize);
}

__global__ void conv3d_forward_kernel(const float* input, const float* kernel, const float* biases, float* output,
                                      int inputDepth, int inputHeight, int inputWidth,
                                      int kernelDepth, int kernelHeight, int kernelWidth,
                                      int outputDepth, int outputHeight, int outputWidth) {
    // Obtain the thread's position in the output volume
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < outputDepth && h < outputHeight && w < outputWidth) {
        float result = biases[0]; // Assuming a single bias per output channel

        // Iterate over the kernel volume
        for (int kd = 0; kd < kernelDepth; kd++) {
            for (int kh = 0; kh < kernelHeight; kh++) {
                for (int kw = 0; kw < kernelWidth; kw++) {
                    int in_d = d + kd;
                    int in_h = h + kh;
                    int in_w = w + kw;

                    if (in_d < inputDepth && in_h < inputHeight && in_w < inputWidth) {
                        result += input[in_d * inputHeight * inputWidth + in_h * inputWidth + in_w] *
                                  kernel[kd * kernelHeight * kernelWidth + kh * kernelWidth + kw];
                    }
                }
            }
        }

        output[d * outputHeight * outputWidth + h * outputWidth + w] = result;
    }
}



void forward_conv_layer(ConvLayer* layer, float* d_input, float* d_output) {
    float* d_inter_output = d_input;
    float* d_temp_output = NULL;

    for (int i = 0; i < NUM_KERNELS; i++) {
        Conv3D* conv = &layer->convs[i];

        // Allocate output buffer for this layer
        cudaMalloc(&d_temp_output, conv->D * conv->H * conv->W * sizeof(float));

        // Set up grid and block dimensions
        dim3 blockDim(8, 8, 8);
        dim3 gridDim(
            (conv->W + blockDim.x - 1) / blockDim.x,
            (conv->H + blockDim.y - 1) / blockDim.y,
            (conv->D + blockDim.z - 1) / blockDim.z
        );

        // Perform convolution
        conv3d_forward_kernel<<<gridDim, blockDim>>>(
            d_inter_output,
            conv->weights,
            conv->biases,
            d_temp_output,
            conv->D,
            conv->H,
            conv->W,
            conv->kernelD,
            conv->kernelH,
            conv->kernelW
        );
        cudaErrorCheck();

        // Save pre-activation output
        cudaMalloc(&conv->pre_activation_output, conv->D * conv->H * conv->W * sizeof(float));
        cudaMemcpy(conv->pre_activation_output, d_temp_output, conv->D * conv->H * conv->W * sizeof(float), cudaMemcpyDeviceToDevice);

        // Apply ReLU activation
        int size = conv->D * conv->H * conv->W;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        relu_activation_kernel<<<blocks, threads>>>(d_temp_output, size);
        cudaErrorCheck();

        // Free or update pointers appropriately
        if (i < NUM_KERNELS - 1) {
            if (d_inter_output != d_input){
                cudaFree(d_inter_output);
            }
            d_inter_output = d_temp_output;
            d_temp_output = NULL;
        } else {
            cudaMemcpy(d_output, d_temp_output, conv->D * conv->H * conv->W * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaFree(d_temp_output);
            d_temp_output = NULL;
        }
    }
}



void forward_encoder(Encoder* encoder, float* d_input, float* d_output) {
    float* d_inter_output;
    cudaMalloc(&d_inter_output, encoder->conv1.convs[0].D * encoder->conv1.convs[0].H * encoder->conv1.convs[0].W * sizeof(float));
    forward_conv_layer(&encoder->conv1, d_input, d_inter_output);
    forward_conv_layer(&encoder->conv2, d_inter_output, d_output);

    cudaFree(d_inter_output); 
}
void forward_decoder(Decoder* decoder, float* d_input, float* d_output) {
    float* d_inter_output;
    cudaMalloc(&d_inter_output, decoder->deconv1.convs[0].D * decoder->deconv1.convs[0].H * decoder->deconv1.convs[0].W * sizeof(float));
    forward_conv_layer(&decoder->deconv1, d_input, d_inter_output);
    forward_conv_layer(&decoder->deconv2, d_inter_output, d_output);

    //cudaFree(d_input);
}

void forward_autoencoder(Autoencoder* autoencoder, float* d_input, float* d_output) {
    float* d_latent_space;
    cudaMalloc(&d_latent_space, autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].D * autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].H * autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].W * sizeof(float));
    forward_encoder(&autoencoder->encoder, d_input, d_latent_space);
    forward_decoder(&autoencoder->decoder, d_latent_space, d_output);

    cudaFree(d_latent_space);


    
}

/*
void forward_encoder_batch(Encoder* encoder, float** d_input_batch, float** d_output_batch, int batch_size) {
    float** d_inter_output_batch = (float**)malloc(batch_size * sizeof(float*));
    for (int i = 0; i < batch_size; i++) {
        cudaMalloc(&d_inter_output_batch[i], encoder->conv1.convs[0].D * encoder->conv1.convs[0].H * encoder->conv1.convs[0].W * sizeof(float));
    }

    forward_conv_layer(&encoder->conv1, d_input_batch, d_inter_output_batch, batch_size);
    forward_conv_layer(&encoder->conv2, d_inter_output_batch, d_output_batch, batch_size);

    for (int i = 0; i < batch_size; i++) {
        cudaFree(d_inter_output_batch[i]);
    }
    free(d_inter_output_batch);
}*/
void forward_encoder_batch(Encoder* encoder, float** d_input_batch, float** d_output_batch, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        forward_encoder(encoder, d_input_batch[i], d_output_batch[i]);
    }
}



void forward_autoencoder_batch(Autoencoder* autoencoder, float** d_input_batch, float** d_output_batch, int batch_size) {
    float** d_latent_space_batch = (float**)malloc(batch_size * sizeof(float*));
    for (int i = 0; i < batch_size; i++) {
        cudaMalloc(&d_latent_space_batch[i], autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].D * autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].H * autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].W * sizeof(float));
    }

    forward_encoder_batch(&autoencoder->encoder, d_input_batch, d_latent_space_batch, batch_size);
    forward_decoder_batch(&autoencoder->decoder, d_latent_space_batch, d_output_batch, batch_size);

    for (int i = 0; i < batch_size; i++) {
        cudaFree(d_latent_space_batch[i]);
    }
    free(d_latent_space_batch);
}

void backward_conv_layer(ConvLayer* layer, float* d_grad_output, float* d_grad_input) {
    float* d_inter_grad_output = d_grad_output;
    float* d_temp_grad_input = NULL;

    for (int i = NUM_KERNELS - 1; i >= 0; i--) {
        Conv3D* conv = &layer->convs[i];

        // Allocate memory for the gradient w.r.t input
        cudaMalloc(&d_temp_grad_input, conv->D * conv->H * conv->W * sizeof(float));

        // Set up grid and block dimensions
        dim3 blockDim(8, 8, 8);
        dim3 gridDim(
            (conv->W + blockDim.x - 1) / blockDim.x,
            (conv->H + blockDim.y - 1) / blockDim.y,
            (conv->D + blockDim.z - 1) / blockDim.z
        );

        // Backward convolution kernel call
        conv3d_backward_kernel<<<gridDim, blockDim>>>(
            conv->input,
            conv->weights,
            d_inter_grad_output,
            d_temp_grad_input,
            conv->grad_weights,
            conv->grad_biases,
            conv->D,
            conv->H,
            conv->W,
            conv->kernelD,
            conv->kernelH,
            conv->kernelW
        );
        cudaErrorCheck();

        // Apply ReLU backward if ReLU was used
        int size = conv->D * conv->H * conv->W;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        relu_backward_kernel<<<blocks, threads>>>(
            d_temp_grad_input,
            conv->pre_activation_output,
            d_temp_grad_input,
            size
        );
        cudaErrorCheck();

        // Free pre_activation_output as it's no longer needed
        cudaFree(conv->pre_activation_output);

        if (i > 0) {
            // Prepare for next iteration
            if (d_inter_grad_output != d_grad_output) {
                cudaFree(d_inter_grad_output);
            }
            d_inter_grad_output = d_temp_grad_input;
            d_temp_grad_input = NULL; // Reset to NULL to avoid double free
        } else {
            // For the first layer, copy the gradient to d_grad_input
            cudaMemcpy(d_grad_input, d_temp_grad_input, conv->D * conv->H * conv->W * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaFree(d_temp_grad_input);
            d_temp_grad_input = NULL;
        }
    }

    // Free d_inter_grad_output if needed
    if (d_inter_grad_output != d_grad_output && d_inter_grad_output != NULL) {
        cudaFree(d_inter_grad_output);
    }
}


void backward_autoencoder(Autoencoder* autoencoder, float* d_input, float* d_output, float* d_target, float learning_rate) {
    int input_size = autoencoder->encoder.conv1.convs[0].D *
                 autoencoder->encoder.conv1.convs[0].H *
                 autoencoder->encoder.conv1.convs[0].W;

    int size = autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].D *
               autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].H *
               autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].W;

    float* d_grad_output;
    cudaMalloc(&d_grad_output, size * sizeof(float));

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    compute_grad_output_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_output,
        d_target,
        d_grad_output,
        size
    );
    cudaErrorCheck();

    // Backpropagate through the decoder
    float* d_grad_latent_space;
    int latent_size = autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].D *
                  autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].H *
                  autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].W;
    
    int size_of_latent_space = latent_size * sizeof(float);

    float* d_grad_latent_space_prev;
    cudaMalloc(&d_grad_latent_space_prev, latent_size * sizeof(float));

    float* d_grad_input;
    cudaMalloc(&d_grad_input, input_size * sizeof(float)); // Define input_size appropriately
    cudaMalloc(&d_grad_latent_space, size_of_latent_space);
    backward_conv_layer(&autoencoder->decoder.deconv2, d_grad_output, d_grad_latent_space);
    backward_conv_layer(&autoencoder->decoder.deconv1, d_grad_latent_space, d_grad_latent_space_prev);
    // Backpropagate through the encoder
    backward_conv_layer(&autoencoder->encoder.conv2, d_grad_latent_space_prev, d_grad_input);
    backward_conv_layer(&autoencoder->encoder.conv1, d_grad_input, NULL);

    // Free allocated memory
    cudaFree(d_grad_output);
    cudaFree(d_grad_latent_space);
    cudaFree(d_grad_latent_space_prev);
    cudaFree(d_grad_input);
}



void scale_gradients(Autoencoder* autoencoder, float scaling_factor) {
    // Scale gradients in encoder
    for (int i = 0; i < NUM_KERNELS; i++) {
        scale_conv_gradients(&autoencoder->encoder.conv1.convs[i], scaling_factor);
        scale_conv_gradients(&autoencoder->encoder.conv2.convs[i], scaling_factor);
    }
    // Scale gradients in decoder
    for (int i = 0; i < NUM_KERNELS; i++) {
        scale_conv_gradients(&autoencoder->decoder.deconv1.convs[i], scaling_factor);
        scale_conv_gradients(&autoencoder->decoder.deconv2.convs[i], scaling_factor);
    }
}

void scale_conv_gradients(Conv3D* conv, float scaling_factor) {
    int weight_size = conv->kernelD * conv->kernelH * conv->kernelW;
    int threads = 256;
    int blocks = (weight_size + threads - 1) / threads;

    scale_kernel<<<blocks, threads>>>(conv->grad_weights, scaling_factor, weight_size);
    cudaErrorCheck();

    // Scale biases
    scale_kernel<<<1, 1>>>(conv->grad_biases, scaling_factor, 1);
    cudaErrorCheck();
}

__global__ void scale_kernel(float* data, float scaling_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scaling_factor;
    }
}

void update_weights_autoencoder(Autoencoder* autoencoder, float learning_rate) {
    // Update weights in encoder
    for (int i = 0; i < NUM_KERNELS; i++) {
        conv3d_update_weights(&autoencoder->encoder.conv1.convs[i], learning_rate);
        conv3d_update_weights(&autoencoder->encoder.conv2.convs[i], learning_rate);
    }
    // Update weights in decoder
    for (int i = 0; i < NUM_KERNELS; i++) {
        conv3d_update_weights(&autoencoder->decoder.deconv1.convs[i], learning_rate);
        conv3d_update_weights(&autoencoder->decoder.deconv2.convs[i], learning_rate);
    }
}


void backward_autoencoder_batch(Autoencoder* autoencoder, float** d_input_batch, float** d_output_batch, float** d_target_batch, int batch_size, float learning_rate) {
    // Reset gradients
    reset_gradients(autoencoder);

    // Loop over batch
    for (int i = 0; i < batch_size; i++) {
        // Perform backward pass for each sample
        backward_autoencoder(autoencoder, d_input_batch[i], d_output_batch[i], d_target_batch[i], learning_rate);
    }

    // Scale gradients by batch size
    float scaling_factor = 1.0f / batch_size;
    scale_gradients(autoencoder, scaling_factor);

    // Update weights
    update_weights_autoencoder(autoencoder, learning_rate);
}



__global__ void accumulate_gradients_kernel(float* accumulated_gradients, const float* gradients, int num_elements, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        accumulated_gradients[idx] += gradients[idx] / batch_size;
    }
}
void reset_gradients(Autoencoder* autoencoder) {
    // Reset gradients in encoder conv1
    for (int i = 0; i < NUM_KERNELS; i++) {
        Conv3D* conv = &autoencoder->encoder.conv1.convs[i];
        int weight_size = conv->kernelD * conv->kernelH * conv->kernelW * sizeof(float);
        cudaMemset(conv->grad_weights, 0, weight_size);
        cudaMemset(conv->grad_biases, 0, sizeof(float));
    }

    // Reset gradients in encoder conv2
    for (int i = 0; i < NUM_KERNELS; i++) {
        Conv3D* conv = &autoencoder->encoder.conv2.convs[i];
        int weight_size = conv->kernelD * conv->kernelH * conv->kernelW * sizeof(float);
        cudaMemset(conv->grad_weights, 0, weight_size);
        cudaMemset(conv->grad_biases, 0, sizeof(float));
    }

    // Reset gradients in decoder deconv1
    for (int i = 0; i < NUM_KERNELS; i++) {
        Conv3D* conv = &autoencoder->decoder.deconv1.convs[i];
        int weight_size = conv->kernelD * conv->kernelH * conv->kernelW * sizeof(float);
        cudaMemset(conv->grad_weights, 0, weight_size);
        cudaMemset(conv->grad_biases, 0, sizeof(float));
    }

    // Reset gradients in decoder deconv2
    for (int i = 0; i < NUM_KERNELS; i++) {
        Conv3D* conv = &autoencoder->decoder.deconv2.convs[i];
        int weight_size = conv->kernelD * conv->kernelH * conv->kernelW * sizeof(float);
        cudaMemset(conv->grad_weights, 0, weight_size);
        cudaMemset(conv->grad_biases, 0, sizeof(float));
    }
}



__global__ void compute_grad_output_kernel(const float* d_output, const float* d_target, float* d_grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_grad_output[idx] = 2.0f * (d_output[idx] - d_target[idx]);
    }
}


__global__ void update_weights(float* weights, const float* gradients, float learning_rate, int num_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_weights) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}


// Define structure to hold momentum terms
typedef struct {
    float* velocity;
} Optimizer;




__global__ void update_weights_kernel(float* weights, float* grad_weights, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * grad_weights[idx];
    }
}

void conv3d_update_weights(Conv3D* conv, float learning_rate) {
    int size = conv->kernelD * conv->kernelH * conv->kernelW;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    update_weights_kernel<<<blocks, threads>>>(conv->weights, conv->grad_weights, learning_rate, size);
    cudaErrorCheck();

    // Update biases
    update_weights_kernel<<<1, 1>>>(conv->biases, conv->grad_biases, learning_rate, 1);
    cudaErrorCheck();
}



void train_autoencoder_old(Autoencoder* autoencoder, float* d_input, float* d_target, int inputSize, int epochs, float learning_rate) {
    float* d_output;
    cudaMalloc(&d_output, inputSize * sizeof(float));
    for (int epoch = 0; epoch < epochs; epoch++) {
        forward_autoencoder(autoencoder, d_input, d_output);
        float loss = mean_square_error(d_output, d_target, inputSize);
        printf("Epoch %d, Loss: %f\n", epoch, loss);
        backward_autoencoder(autoencoder, d_input, d_output, d_target, learning_rate);
    }
    cudaFree(d_output);

}

void train_autoencoder(Autoencoder* autoencoder, float** d_input_batch, float** d_target_batch, int batch_size, int input_size, int epochs, float learning_rate) {
    // Allocate memory for the batch output
    float** d_output_batch = (float**)malloc(batch_size * sizeof(float*));
    for (int i = 0; i < batch_size; i++) {
        cudaMalloc(&d_output_batch[i], input_size * sizeof(float));
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;  // Track the loss over all batches in the epoch

        // Forward pass for the entire batch
        forward_autoencoder_batch(autoencoder, d_input_batch, d_output_batch, batch_size);
        
        // Calculate loss and accumulate it
        for (int i = 0; i < batch_size; i++) {
            float loss = mean_square_error(d_output_batch[i], d_target_batch[i], input_size);
            total_loss += loss;
        }

        // Compute average loss for this batch
        total_loss /= batch_size;

        printf("Epoch %d, Loss: %f\n", epoch, total_loss);

        // Backpropagation for the entire batch
        backward_autoencoder_batch(autoencoder, d_input_batch, d_output_batch, d_target_batch, batch_size, learning_rate);
    }

    // Free allocated GPU memory for output batch
    for (int i = 0; i < batch_size; i++) {
        cudaFree(d_output_batch[i]);
    }
    free(d_output_batch);
}
/***
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(d_input_batch[i], h_input_batch[i], size, cudaMemcpyHostToDevice, stream);
// Perform kernel operations in the same stream
conv3d_forward_kernel<<<gridDim, blockDim, 0, stream>>>(...);
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
trial!!!! 
***/
void free_conv_layer(ConvLayer* layer) {
    for (int i = 0; i < NUM_KERNELS; i ++) {
        conv3d_free(&layer->convs[i]);
    }

}

void free_encoder(Encoder* encoder) {
    free_conv_layer(&encoder->conv1);
    free_conv_layer(&encoder->conv2);

}

void free_decoder(Decoder* decoder) {
    free_conv_layer(&decoder->deconv1);
    free_conv_layer(&decoder->deconv2);
}

void free_autoencoder(Autoencoder* autoencoder) {
    free_encoder(autoencoder->encoder);
    free_decoder(autoencoder->decoder);
}