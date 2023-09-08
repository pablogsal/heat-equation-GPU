#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

const int WIDTH = 1024;
const int HEIGHT = 1024;
const int NUM_THREADS_X = 16;
const int NUM_THREADS_Y = 16;
const int NUM_BLOCKS_X = WIDTH / NUM_THREADS_X;
const int NUM_BLOCKS_Y = HEIGHT / NUM_THREADS_Y;

const float ALPHA = 0.1f;

float* d_input = nullptr;
float* d_output = nullptr;
cudaGraphicsResource* cudaGLResource;

// CUDA kernel for solving the heat equation with shared memory
__global__ void
heatSolver(float* d_input, float* d_output, int width, int height, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float sharedData[NUM_THREADS_Y + 2][NUM_THREADS_X + 2];

    int sharedX = threadIdx.x + 1;
    int sharedY = threadIdx.y + 1;

    if (i < width && j < height) {
        int idx = j * width + i;
        float center = d_input[idx];

        // Load data into shared memory with proper boundary checks
        if (threadIdx.x == 0) {
            sharedData[sharedY][sharedX - 1] = (i > 0) ? d_input[idx - 1] : center;
        }
        if (threadIdx.x == blockDim.x - 1) {
            sharedData[sharedY][sharedX + 1] = (i < width - 1) ? d_input[idx + 1] : center;
        }
        if (threadIdx.y == 0) {
            sharedData[sharedY - 1][sharedX] = (j > 0) ? d_input[idx - width] : center;
        }
        if (threadIdx.y == blockDim.y - 1) {
            sharedData[sharedY + 1][sharedX] = (j < height - 1) ? d_input[idx + width] : center;
        }

        sharedData[sharedY][sharedX] = center;
        __syncthreads();

        // Calculate updated value using shared memory
        float left = sharedData[sharedY][sharedX - 1];
        float right = sharedData[sharedY][sharedX + 1];
        float top = sharedData[sharedY - 1][sharedX];
        float bottom = sharedData[sharedY + 1][sharedX];

        d_output[idx] = center + alpha * (left + right + top + bottom - 4 * center);
    }
}

bool isAddingSection = false;  // Flag to indicate if the user is adding a section
int startX, startY;  // Starting coordinates of the selection
int endX, endY;  // Ending coordinates of the selection
// CUDA kernel to update d_input with a section when requested
__global__ void
addSection(float* d_input, int width, int height, int startX, int startY, int endX, int endY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        int idx = j * width + i;
        if (i >= startX && i <= endX && j >= startY && j <= endY) {
            d_input[idx] = 10.0f;
        }
    }
}
// Mouse callback function to add sections with 1.0f when clicking
void
mouseCallback(int button, int state, int x, int y)
{
    printf("Mouse callback: %d %d %d %d\n", button, state, x, y);
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            isAddingSection = true;
            startX = static_cast<int>(static_cast<float>(x) / WIDTH * WIDTH);
            startY = static_cast<int>((1.0f - static_cast<float>(y) / HEIGHT) * HEIGHT);
        } else if (state == GLUT_UP && isAddingSection) {
            isAddingSection = false;
            endX = static_cast<int>(static_cast<float>(x) / WIDTH * WIDTH);
            endY = static_cast<int>((1.0f - static_cast<float>(y) / HEIGHT) * HEIGHT);

            // Ensure that the starting and ending coordinates are in the correct
            // order
            if (startX > endX) {
                std::swap(startX, endX);
            }
            if (startY > endY) {
                std::swap(startY, endY);
            }

            // Launch the CUDA kernel to update d_input with the selected rectangle
            addSection<<<dim3(NUM_BLOCKS_X, NUM_BLOCKS_Y), dim3(NUM_THREADS_X, NUM_THREADS_Y)>>>(
                    d_input,
                    WIDTH,
                    HEIGHT,
                    startX,
                    startY,
                    endX,
                    endY);

            // Synchronize to ensure the kernel has finished
            cudaDeviceSynchronize();
        }
    }
}

int currentStep = 0;
float h_output[WIDTH * HEIGHT];

// OpenGL display function
void
display()
{
    currentStep++;
    // Call the kernel to update the simulation
    heatSolver<<<dim3(NUM_BLOCKS_X, NUM_BLOCKS_Y), dim3(NUM_THREADS_X, NUM_THREADS_Y)>>>(
            d_input,
            d_output,
            WIDTH,
            HEIGHT,
            ALPHA);
    // Synchronize to ensure the kernel has finished
    cudaDeviceSynchronize();

    if (currentStep % 1000 != 0) {
        // Swap d_input and d_output
        float* temp = d_input;
        d_input = d_output;
        d_output = temp;
    } else {
        // Map the CUDA buffer to OpenGL
        float* d_mapped_buffer = nullptr;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cudaGLResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_mapped_buffer, &num_bytes, cudaGLResource);

        // Copy data from d_output to the OpenGL PBO
        cudaMemcpy(d_mapped_buffer, d_output, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToDevice);

        // Unmap the CUDA buffer from OpenGL
        cudaGraphicsUnmapResources(1, &cudaGLResource, 0);

        glClear(GL_COLOR_BUFFER_BIT);

        // If a non-zero named buffer object is bound to the
        // GL_PIXEL_UNPACK_BUFFER target (see main function) while a block of
        // pixels is specified, data is treated as a byte offset into the buffer
        // object's data store.
        glDrawPixels(WIDTH, HEIGHT, GL_RED, GL_FLOAT, 0);  // Note the use of nullptr

        glutSwapBuffers();
    }

    // Request the display function to be called
    glutPostRedisplay();
}

int
main(int argc, char** argv)
{
    // Initialize OpenGL and create a window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Heat Equation Simulation");

    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW initialization error: " << glewGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Set up OpenGL context and callbacks
    glutDisplayFunc(display);
    glutMouseFunc(mouseCallback);  // Register the mouse callback

    // Initialize CUDA device memory
    cudaMalloc((void**)&d_input, sizeof(float) * WIDTH * HEIGHT);
    cudaMalloc((void**)&d_output, sizeof(float) * WIDTH * HEIGHT);

    // Initialize input data with initial conditions
    float* h_input = new float[WIDTH * HEIGHT];

    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < HEIGHT; ++j) {
            int idx = j * WIDTH + i;
            h_input[idx] = 0.0f;
        }
    }

    cudaMemcpy(d_input, h_input, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_input;

    // Initialize CUDA-OpenGL interoperability
    GLuint cudaGLBuffer;
    glGenBuffers(1, &cudaGLBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, cudaGLBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaGLResource, cudaGLBuffer, cudaGraphicsMapFlagsNone);

    // Start the OpenGL main loop
    glutMainLoop();

    // Clean up CUDA resources
    cudaFree(d_input);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}
