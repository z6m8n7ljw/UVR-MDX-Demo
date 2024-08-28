# Audio Separation: Browser-Based Audio Separation with WebGPU and ONNX Runtime Web

This repository contains an example of running a model inference for audio separation, in a browser using [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) with WebGPU.

## Model Overview

Audio Separation creates masks for an audio using an encoder. By using WebGPU, we can speed up the encoder, making it feasible to run it inside the browser, even on an integrated GPU.

## Getting Started

### Prerequisites

Ensure that you have [Node.js](https://nodejs.org/) installed on your machine.

### Installation

1. Install the required dependencies:

```sh
npm install
```

### Building the Project

1. Bundle the code using webpack:

```sh
npm run build
```

This command generates the bundle file `./dist/index.js`.

### The ONNX Model

The model used in this project is hosted on [Hugging Face](https://huggingface.co/seanghay/uvr_models).

### Running the Project

Start a web server to serve the current folder at http://localhost:8080/. To start the server, run:

```sh
npm run dev
```
