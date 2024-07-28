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

__global__ void conv3d_forward_kernel() {

}


__global__ void conv3d_backward_kernel() {
    
}

void forward_conv_layer(ConvLayer* layer, float* d_input, float* d_output) {
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



void train_autoencoder(Autoencoder* autoencoder, float* d_input, float* d_target, int inputSize, int epochs, float learning_rate) {
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