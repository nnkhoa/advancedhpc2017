#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2017, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            printf("labwork 1 CPU-OMP ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3)); 
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
	#pragma omp parallel for schedule(dynamic)
	    for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int deviceCount;
    struct cudaDeviceProp gpuProp;

    cudaGetDeviceCount(&deviceCount);
    
    printf("number of GPU: %d\n", deviceCount);

    for(int i = 0; i < deviceCount; i++){
        cudaGetDeviceProperties(&gpuProp, i);
        printf("Name: %s\n", gpuProp.name);
        printf("Max Threads per Block: %d\n", gpuProp.maxThreadsPerBlock);
        printf("Total Core: %d\n", getSPcores(gpuProp));
        printf("Clock Rate: %d\n", gpuProp.clockRate);
        printf("Multi Processor Count: %d\n", gpuProp.multiProcessorCount);
        printf("Warp Size: %d\n", gpuProp.warpSize);
        printf("Memory Clock Rate: %d\n", gpuProp.memoryClockRate);
        printf("Memory Bus Width: %d\n", gpuProp.memoryBusWidth);
        printf("Memory Bandwidth: %d\n", gpuProp.memoryClockRate*gpuProp.memoryBusWidth);
    }

}

__global__ void grayscaleConvert(char* input, char* output, int imagePixelCount){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i < imagePixelCount){

        char g = (char) (((int) input[i * 3] + (int) input[i * 3 + 1] +
                                      (int) input[i * 3 + 2]) / 3);
        output[i * 3] =  output[i * 3 + 1] = output[i * 3 + 2] = g;
    }
}

void Labwork::labwork3_GPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3)); 

    char *blockSizeEnv = getenv("CUDA_BLOCK_SIZE");

    if(!blockSizeEnv){
        printf("No Environment Variable specified\n");
        printf("Please use > CUDA_BLOCK_SIZE=block_size ./labwork ...\n");
        return;
    }

    int blockSize = atoi(blockSizeEnv);
    int numBlock = pixelCount/blockSize;
    
    char *cuInput, *cuOutput;

    if(cudaMalloc(&cuInput, pixelCount*3*sizeof(char)) != cudaSuccess){
    	fprintf(stderr, "Cuda Memory Alloc error!\n");
	return;
    }
    if(cudaMalloc(&cuOutput, pixelCount*3*sizeof(char)) != cudaSuccess){
    	fprintf(stderr, "Cuda Memory Alloc error!\n");
	return;
    }
    
    if(cudaMemcpy(cuInput, inputImage->buffer, pixelCount*3*sizeof(char), cudaMemcpyHostToDevice) != cudaSuccess){
    	fprintf(stderr, "Copy input buffer error!\n");
	return;
    }
    
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
    	grayscaleConvert<<<numBlock, blockSize>>>(cuInput, cuOutput, pixelCount);
    }
    if(cudaMemcpy(outputImage, cuOutput, pixelCount*3*sizeof(char), cudaMemcpyDeviceToHost) != cudaSuccess){
    	fprintf(stderr, "Copy output buffer error!\n");
	return;
    }
    
    cudaFree(cuOutput);
    cudaFree(cuInput);
}

__global__ void grayscaleConvert2D(unsigned char* input, unsigned char* output, int imagePixelCount, int imageWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int i = row * imageWidth + threadIdx.x +blockIdx.x * blockDim.x;

    if(i < imagePixelCount){

        unsigned char g = (unsigned char) (((int) input[i * 3] + (int) input[i * 3 + 1] +
                                      (int) input[i * 3 + 2]) / 3);
        output[i * 3] =  output[i * 3 + 1] = output[i * 3 + 2] = g;
    }
}

void Labwork::labwork4_GPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3)); 

    char *blockSizeEnv = getenv("CUDA_BLOCK_SIZE");

    if(!blockSizeEnv){
        printf("No Environment Variable specified\n");
        printf("Please use > CUDA_BLOCK_SIZE=block_size ./labwork ...\n");
        return;
    }

    int blockSizeValue = atoi(blockSizeEnv);
    
    int gridWidth = (inputImage->width + blockSizeValue - 1)/blockSizeValue;
    int gridHeight = (inputImage->height + blockSizeValue - 1)/blockSizeValue;

    dim3 gridSize = dim3(gridWidth,gridHeight);
    dim3 blockSize = dim3(blockSizeValue,blockSizeValue);

    unsigned char *cuInput, *cuOutput;
    cudaMalloc(&cuInput, pixelCount*3*sizeof(unsigned char));
    cudaMalloc(&cuOutput, pixelCount*3*sizeof(unsigned char));
    
    cudaMemcpy(cuInput, inputImage->buffer, pixelCount*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        grayscaleConvert2D<<<gridSize, blockSize>>>(cuInput, cuOutput, pixelCount, inputImage->width);
    }
    cudaMemcpy(outputImage, cuOutput, pixelCount*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    cudaFree(cuOutput);
    cudaFree(cuInput);   
}

__global__ void gaussianConvolution(unsigned char *input, unsigned char *output, int imageWidth, int imageHeight, int pixelCount){
    int gKernel[7][7] = {{0, 0, 1, 2, 1, 0, 0},
                        {0, 3, 13, 22, 13, 3, 0},
                        {1, 13, 59, 97, 59, 13, 1},
                        {2, 22, 97, 159, 97, 22, 2},
                        {1, 13, 59, 97, 59, 13, 1},
                        {0, 3, 13, 22, 13, 3, 0},
                        {0, 0, 1, 2, 1, 0, 0}};
    int sum = 0;
    int c = 0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x +blockIdx.x * blockDim.x;
    if (col >= imageWidth) return;
    for(int y = -3; y <= 3; y++){
        for(int x = -3; x <= 3; x++){
            int i = col + x;
            int j = row + y;

            if( i < 0 || i >= imageWidth || j < 0 || j >= imageHeight)
                continue;

            int tid = j * imageWidth + i;
            unsigned char pixelValue = input[tid*3];
            int coefficient = gKernel[y+3][x+3];
            sum += pixelValue*coefficient;
            c += coefficient;
        }
    }
    sum /= c;
    int postOut = row * imageWidth + col;
    if(postOut < pixelCount){
    	output[postOut * 3] = output[postOut * 3 + 1] = output[postOut * 3 + 2] = sum;
    }
}

void Labwork::labwork5_GPU() {

    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    char *blockSizeEnv = getenv("CUDA_BLOCK_SIZE");

    if(!blockSizeEnv){
        printf("No Environment Variable specified\n");
        printf("Please use > CUDA_BLOCK_SIZE=block_size ./labwork ...\n");
        return;
    }
    int blockSizeValue = atoi(blockSizeEnv);
    
    int gridWidth = (inputImage->width + blockSizeValue - 1)/blockSizeValue;
    int gridHeight = (inputImage->height + blockSizeValue - 1)/blockSizeValue;

    dim3 gridSize = dim3(gridWidth,gridHeight);
    dim3 blockSize = dim3(blockSizeValue,blockSizeValue);

    unsigned char *cuInput, *cuOutput, *cuGray;
    cudaMalloc(&cuInput, pixelCount*3*sizeof(unsigned char));
    cudaMalloc(&cuOutput, pixelCount*3*sizeof(unsigned char));
    cudaMalloc(&cuGray, pixelCount*3*sizeof(unsigned char));

    cudaMemcpy(cuInput, inputImage->buffer, pixelCount*3*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    grayscaleConvert2D<<<gridSize, blockSize>>>(cuInput, cuGray, pixelCount, inputImage->width);

    gaussianConvolution<<<gridSize, blockSize>>>(cuGray, cuOutput, inputImage->width, inputImage->height, pixelCount);
    
    cudaMemcpy(outputImage, cuOutput, pixelCount*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    cudaFree(cuOutput);
    cudaFree(cuInput);
    cudaFree(cuGray);
}

void Labwork::labwork6_GPU() {

}

void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
