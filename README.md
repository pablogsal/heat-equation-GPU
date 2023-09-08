# Heat Equation Simulation

This project is an implementation of a heat equation simulation using CUDA and
OpenGL. The simulation solves the heat equation over a 2D grid and provides a
visual representation of the temperature distribution. Users can interact with
the simulation by adding heat sources to the grid through mouse input.

## Prerequisites

Before running this project, you'll need the following dependencies:

- OpenGL (GLUT)
- GLEW
- CUDA

## Compilation

To compile the project, you can use the following commands:

```shell
# Compile CUDA code
make build
cd build
cmake ..
make

# Run the program
./heat_equation
```

## Usage

- Left-click on the simulation window to add a heat source to the grid. Click and drag to create a rectangular heat source.
- The simulation will update and display the temperature distribution in real-time.
- The program will continue running until you close the window.

## License

This project is provided under the [MIT License](LICENSE).
