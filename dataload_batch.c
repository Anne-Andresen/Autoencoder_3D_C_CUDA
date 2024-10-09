#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nifti1_io.h>
#include <cuda_runtime.h>

#define MAX_BATCH_SIZE 8

// Structure to hold image data for a single image
typedef struct {
    float *data;  // Pointer to the image data
    int depth, height, width;
} NIfTIImage;

// Structure to hold batch of images
typedef struct {
    NIfTIImage images[MAX_BATCH_SIZE];
    int batch_size;
} NIfTIBatch;

// Function to load a NIfTI image from file
NIfTIImage load_nifti_image(const char *filepath) {
    nifti_image *nim = nifti_image_read(filepath, 1);
    if (nim == NULL) {
        fprintf(stderr, "Error: Could not load NIfTI image %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    NIfTIImage img;
    img.depth = nim->nz;
    img.height = nim->ny;
    img.width = nim->nx;

    // Allocate memory for the image data
    size_t data_size = nim->nvox * sizeof(float);
    img.data = (float*)malloc(data_size);
    memcpy(img.data, nim->data, data_size);

    nifti_image_free(nim);
    return img;
}

// Function to load a batch of images
NIfTIBatch load_batch(const char *nifti_dir, const char **file_list, int batch_size) {
    NIfTIBatch batch;
    batch.batch_size = batch_size;

    for (int i = 0; i < batch_size; ++i) {
        char filepath[256];
        snprintf(filepath, sizeof(filepath), "%s/%s", nifti_dir, file_list[i]);
        batch.images[i] = load_nifti_image(filepath);
    }
    return batch;
}

// Function to copy a batch of images to CUDA
float** copy_batch_to_cuda(NIfTIBatch *batch) {
    float **d_data_array = (float**)malloc(batch->batch_size * sizeof(float*));

    for (int i = 0; i < batch->batch_size; ++i) {
        size_t data_size = batch->images[i].depth * batch->images[i].height * batch->images[i].width * sizeof(float);
        cudaMalloc((void**)&d_data_array[i], data_size);
        cudaMemcpy(d_data_array[i], batch->images[i].data, data_size, cudaMemcpyHostToDevice);
    }
    return d_data_array;
}

// Function to free a batch of images
void free_batch(NIfTIBatch *batch) {
    for (int i = 0; i < batch->batch_size; ++i) {
        free(batch->images[i].data);
    }
}

// Free the GPU memory for a batch
void free_cuda_batch(float **d_data_array, int batch_size) {
    for (int i = 0; i < batch_size; ++i) {
        cudaFree(d_data_array[i]);
    }
    free(d_data_array);
}

int main() {
    // Directory containing NIfTI files and the list of files to load
    const char *nifti_directory = "/path/to/nifti/files";
    const char *file_list[] = {"file1.nii", "file2.nii", "file3.nii", "file4.nii"}; // Example file list

    int batch_size = 4;  // Set batch size
    NIfTIBatch batch = load_batch(nifti_directory, file_list, batch_size);

    // Copy batch to CUDA
    float **d_image_data = copy_batch_to_cuda(&batch);

    // TODO: Pass d_image_data array to your CUDA kernels for processing

    // Free resources
    free_cuda_batch(d_image_data, batch_size);
    free_batch(&batch);

    return 0;
}
