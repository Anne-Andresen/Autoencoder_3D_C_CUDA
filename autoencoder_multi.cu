#include <autoencoder.h>
#include <loss.h>
#include <utils.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <conv3d.h>

#define NUM_KERNELS = 3

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


#define cudaErrorCheck() {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        print("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaErrorString(e));
        exit(EXIT_FAILURE);
    }
}


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
    init_conv_layer(&decoder->deconv1, inputDepth. inputHeight, inputWidth, kernelSize);
    init_conv_layer(&decoder->deconv2, inputDepth, inputHeight, inputWidth, kernelSize);
}

void init_autoencoder(Autoencoder* autoencoder, int inputDepth, int, inputHeight, int inputWidth, int kernelSize) {
    init_encoder(&autoencoder->encoder, inputDepth, inputHeight, inputWidth, kernelSize);
    init_decoder(&autoencoder->decoder, inputDepth, inputHeight, inputWidth, kernelSize);
}

__global__ void conv3d_forward_kernel(const float* input, const float* kernel, float* output,
                                      int inputDepth, int inputHeight, int inputWidth,
                                      int kernelDepth, int kernelHeight, int kernelWidth,
                                      int outputDepth, int outputHeight, int outputWidth) {
    // Obtain the thread's position in the output volume
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < outputDepth && h < outputHeight && w < outputWidth) {
        float result = 0.0f;

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

__global__ void conv3d_backward_kernel(const float* d_output, const float* kernel, float* d_input,
                                       int inputDepth, int inputHeight, int inputWidth,
                                       int kernelDepth, int kernelHeight, int kernelWidth,
                                       int outputDepth, int outputHeight, int outputWidth) {
    // Identify position in input volume
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < inputDepth && h < inputHeight && w < inputWidth) {
        float gradient = 0.0f;

        for (int kd = 0; kd < kernelDepth; kd++) {
            for (int kh = 0; kh < kernelHeight; kh++) {
                for (int kw = 0; kw < kernelWidth; kw++) {
                    int out_d = d - kd;
                    int out_h = h - kh;
                    int out_w = w - kw;

                    if (out_d >= 0 && out_h >= 0 && out_w >= 0 && out_d < outputDepth && out_h < outputHeight && out_w < outputWidth) {
                        gradient += d_output[out_d * outputHeight * outputWidth + out_h * outputWidth + out_w] *
                                    kernel[kd * kernelHeight * kernelWidth + kh * kernelWidth + kw];
                    }
                }
            }
        }

        d_input[d * inputHeight * inputWidth + h * inputWidth + w] = gradient;
    }
}



void forward_conv_layer_old(ConvLayer* layer, float* d_input, float* d_output) {
    float* d_temp_output;
    cudaMalloc(&d_temp_output, layer->convs[0].D * layer->convs[0].H * layer->convs[0].W * sizeof(float));
    float* d_inter_output = d_input;
    for (int i = 0; i < NUM_KERNELS; i++) {
        conv3d_forward_kernel<<<  /* grid and block dimensions */, /* shared memory size */, /* stream */ >>(// kernel params

        );
        cudaErrorCheck();
        if (i < NUM_KERNELS - 1) {
            d_inter_output = d_temp_output;
        }
        else {
            cudaMemcpy(d_output, d_temp_output, layer->convs[i].D * layer->convs[i].H * layer->convs[i].W * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }
    cudaFree(d_temp_output);
}

void forward_conv_layer(ConvLayer* layer, float** d_input_batch, float** d_output_batch, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        float* d_temp_output;
        cudaMalloc(&d_temp_output, layer->convs[0].D * layer->convs[0].H * layer->convs[0].W * sizeof(float));
        float* d_inter_output = d_input_batch[i];
        
        for (int j = 0; j < NUM_KERNELS; j++) {
            // Run convolution kernel for each layer
            conv3d_forward_kernel<<< /* grid and block dimensions */ >>>(
                // Pass appropriate arguments
            );
            cudaErrorCheck();
            
            if (j < NUM_KERNELS - 1) {
                d_inter_output = d_temp_output;
            } else {
                cudaMemcpy(d_output_batch[i], d_temp_output, layer->convs[j].D * layer->convs[j].H * layer->convs[j].W * sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }
        
        cudaFree(d_temp_output);
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

    cudaFree(d_input);
}

void forward_autoencoder(Autoencoder* autoencoder, float* d_input, float* d_output) {
    float* d_latent_space;
    cudaMalloc(&d_latent_space, autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].D * autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].H * autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].W * sizeof(float));
    forward_encoder(&autoencoder->encoder, d_input, d_latent_space);
    forward_decoder(&autoencoder->decoder, d_latent_space, d_output);

    cudaFree(d_latent_space);


    
}


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
    float* d_temp_grad_input;
    cudaMalloc(&d_grad_temp_input, layer->convs[0].D * layer->convs[0].H * layer->convs[0].W * sizeof(float));
    float* d_inter_grad_output = d_grad_output;

    for (int i = NUM_KERNELS - 1; i >= 0; i--) {
        conv3d_backward_kernel<<<>>>();
        cudaErrorCheck();
        if (i > 0) {
            d_inter_grad_output = d_temp_grad_input;
        }
        else {
            cudaMemcpy(d_grad_input, d_temp_grad_input, layer->convs[i].D * layer->convs[i].H * layer->convs[i].W * sizeof(float));
        }
    }
    
    cudaFree(d_temp_grad_input);
    
}

void backward_autoencoder(Autoencoder* autoencoder, float* d_input, float* d_output, float* d_target, float learning_rate) {
    float* d_grad_output;
    cudaMalloc(&d_grad_output, autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].D * autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].H * autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].W * sizeof(float));
    for (int i = 0; i < autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].D * autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].H * autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].W; i++) {
        grad_output[i] = 2 * (d_output[i] - d_target[i]);
    }

    float* d_grad_latent_space;
    cudaMalloc(&d_grad_latent_space, autoencoder->decoder.deconv1.convs[NUM_KERNELS - 1].D * autoencoder->decoder.deconv1.convs[NUM_KERNELS - 1].H * autoencoder->decoder.deconv1.convs[NUM_KERNELS - 1].W * sizeof(float));
    backward_conv_layer(&autoencoder->decoder.deconv2, d_grad_output, d_grad_latent_space);

    float* d_grad_intermediate;
    cudaMalloc(&d_grad_intermediate, autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].D * autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].H * autoencoder->encoder.conv2.convs[NUM_KERNELS - 1].W * sizeof(float));
    backward_conv_layer(&autoencoder->decoder.deconv1.convs[NUM_KERNELS - 1], d_grad_latent_space, d_grad_intermediate);

    float* d_grad_input;
    cudaMalloc(&d_grad_input, autoencoder->encoder.conv1.convs[NUM_KERNELS - 1].D * autoencoder->encoder.conv1.convs[NUM_KERNELS - 1].H * autoencoder->encoder.conv1.convs[NUM_KERNELS - 1].W * sizeof(float));
    backward_conv_layer(&autoencoder->encoder.conv2, d_grad_intermediate, d_grad_input);


    backward_conv_layer(&autoencoder->encoder.conv1, d_grad_input, NULL);

    for (int i = 0; i < NUM_KERNELS; i++) {
        conv3d_update_weights(&autoencoder->encoder.conv1[i], learning_rate);
        conv3d_update_weights(&autoencoder->encoder.conv2[i], learning_rate);
        conv3d_update_weights(&autoencoder->decoder.deconv1[i], learning_rate);
        conv3d_update_weights(&autoencoder->decoder.deconv2[i], learning_rate);
    }

    cudaFree(d_grad_input);
    cudaFree(d_grad_latent_space);
    cudaFree(d_grad_intermediate);
    cudaFree(d_grad_input);

}

void backward_autoencoder_batch(Autoencoder* autoencoder, float** d_input_batch, float** d_output_batch, float** d_target_batch, int batch_size, float learning_rate) {
    float** d_grad_output_batch = (float**)malloc(batch_size * sizeof(float*));
    for (int i = 0; i < batch_size; i++) {
        cudaMalloc(&d_grad_output_batch[i], autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].D * autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].H * autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].W * sizeof(float));
        for (int j = 0; j < autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].D * autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].H * autoencoder->decoder.deconv2.convs[NUM_KERNELS - 1].W; j++) {
            d_grad_output_batch[i][j] = 2 * (d_output_batch[i][j] - d_target_batch[i][j]);
        }
    }

    // Run backward operations for each layer in the autoencoder, using the same process as above.
    // Ensure to accumulate gradients across the batch

    for (int i = 0; i < NUM_KERNELS; i++) {
        conv3d_update_weights(&autoencoder->encoder.conv1.convs[i], learning_rate / batch_size);
        conv3d_update_weights(&autoencoder->encoder.conv2.convs[i], learning_rate / batch_size);
        conv3d_update_weights(&autoencoder->decoder.deconv1.convs[i], learning_rate / batch_size);
        conv3d_update_weights(&autoencoder->decoder.deconv2.convs[i], learning_rate / batch_size);
    }

    for (int i = 0; i < batch_size; i++) {
        cudaFree(d_grad_output_batch[i]);
    }
    free(d_grad_output_batch);
}
void accumulate_gradients(float* accumulated_gradients, const float* gradients, int num_elements, int batch_size) {
    for (int i = 0; i < num_elements; i++) {
        accumulated_gradients[i] += gradients[i] / batch_size;
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

void update_weights_with_momentum(Conv3D* conv, float* gradients, Optimizer* optimizer, float learning_rate, float momentum, int num_weights) {
    // Allocate space for velocity if not already done
    if (optimizer->velocity == NULL) {
        cudaMalloc(&(optimizer->velocity), num_weights * sizeof(float));
        cudaMemset(optimizer->velocity, 0, num_weights * sizeof(float)); // Initialize to zero
    }

    // Update weights using gradient and momentum
    update_weights_with_momentum_kernel<<<gridDim, blockDim>>>(conv->weights, gradients, optimizer->velocity, learning_rate, momentum, num_weights);
    cudaErrorCheck();
}


// Launching the kernel
void update_weights_host(float* d_weights, float* d_gradients, float learning_rate, int num_weights) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_weights + threads_per_block - 1) / threads_per_block;

    update_weights<<<blocks_per_grid, threads_per_block>>>(d_weights, d_gradients, learning_rate, num_weights);
    cudaDeviceSynchronize();
}

float* allocate_large_memory_block(int total_size) {
    float* d_memory_block;
    cudaMalloc((void**)&d_memory_block, total_size * sizeof(float));
    cudaMemset(d_memory_block, 0, total_size * sizeof(float)); // Initialize to zero
    return d_memory_block;
}

void free_large_memory_block(float* d_memory_block) {
    if (d_memory_block != NULL) {
        cudaFree(d_memory_block);
    }
}


void train_autoencoder_old(Autoencoder* autoencoder, float* d_input, float* d_target, int inputSize, int epochs, float learning_rate) {
    float* d_output
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
    free_conv_layer(encoder->conv1);
    free_conv_layer(encoder->conv2);

}

void free_decoder(Decoder* decoder) {
    free_conv_layer(decoder->deconv1);
    free_conv_layer(decoder->deconv2);
}

void free_autoencoder(Autoencoder* autoencoder) {
    free_encoder(autoencoder->encoder);
    free_decoder(autoencoder->decoder);
}