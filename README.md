# Kaldi ASR Integration With Triton

This repository provides a Kaldi ASR custom backend for the NVIDIA Triton (former TensorRT Inference Server). It can be used to demonstrate high-performance online inference on Kaldi ASR models. This includes handling the gRPC communication between the Triton and clients, and the dynamic batching of inference requests. This repository is tested and maintained by NVIDIA.

## Table Of Contents

- [Kaldi ASR Integration With Triton](#kaldi-asr-integration-with-triton)
  - [Table Of Contents](#table-of-contents)
  - [Solution overview](#solution-overview)
    - [Reference model](#reference-model)
    - [Default configuration](#default-configuration)
  - [Setup](#setup)
    - [Requirements](#requirements)
  - [Performance](#performance)
    - [Metrics](#metrics)
    - [Results](#results)

## Solution overview

This repository provides a wrapper around the online GPU-accelerated ASR pipeline from the paper [GPU-Accelerated Viterbi Exact Lattice Decoder for Batched Online and Offline Speech Recognition](https://arxiv.org/abs/1910.10032). That work includes a high-performance implementation of a GPU HMM Decoder, a low-latency Neural Net driver, fast Feature Extraction for preprocessing, and new ASR pipelines tailored for GPUs. These different modules have been integrated into the Kaldi ASR framework.

This repository contains a Triton custom backend for the Kaldi ASR framework. This custom backend calls the high-performance online GPU pipeline from the Kaldi ASR framework. This Triton integration provides ease-of-use to Kaldi ASR inference: gRPC streaming server, dynamic sequence batching, and multi-instances support. A client connects to the gRPC server, streams audio by sending chunks to the server, and gets back the inferred text as an answer. More information about the Triton can be found [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/).

This Triton integration is meant to be used with the LibriSpeech model for demonstration purposes. We include a pre-trained version of this model to allow you to easily test this work. Both the Triton integration and the underlying Kaldi ASR online GPU pipeline are a work in progress and will support more functionalities in the future.

### Reference model

A reference model is used by all test scripts and benchmarks presented in this repository to illustrate this solution. We are using the Kaldi ASR `LibriSpeech` recipe, available [here](https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5). It was trained by NVIDIA and is delivered as a pre-trained model.

### Default configuration

Details about parameters:

- `model path`: Configured to use the pretrained LibriSpeech model.
- `use_tensor_cores`: 1
- `main_q_capacity`: 30000
- `aux_q_capacity`: 400000
- `beam`: 10
- `num_channels`: 4000
- `lattice_beam`: 7
- `max_active`: 10,000
- `frame_subsampling_factor`: 3
- `acoustic_scale`: 1.0
- `num_worker_threads`: 40
- `max_batch_size`: 400
- `instance_group.count`: 1

## Setup

### Requirements

This repository contains Dockerfiles and Kubernetes yamls which extend the Kaldi and Triton NVIDIA GPU Cloud (NGC) containers and encapsulates some dependencies. Aside from these dependencies, ensure you have:

- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
- [Kubernetes Tanzu](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-vsphere-tanzu.html)
- [NVIDIA GPU Operator](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-gpu-operator.html)
- [Supported NVIDIA GPUs](https://docs.nvidia.com/ai-enterprise/1.1/product-support-matrix/index.html#abstract)
- [Local Docker Registry](https://github.com/tuttlebr/deepops/blob/deep-learning-examples/playbooks/k8s-cluster/container-registry.yml)

1. Create Tanzu Cluster

   ```bash
   kubectl -f cluster-configuration.yaml create
   ```

2. Deploy NVAIE GPU Operator

   [NVIDIA GPU Operator](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-gpu-operator.html)

3. Deploy Helm Chart
   
   ```bash
   helm install triton triton-speech-recognition \
      --set imageCredentials.password=<NGC_REGISTRY_API_KEY> \
      --set imageCredentials.email=<NGC_REGISTRY_EMAIL>
   ```

## Performance

### Metrics

Throughput is measured using the RTFX metric. It is defined such as : `RTFX = (number of seconds of audio inferred) / (compute time in seconds)`. It is the inverse of the RTF (Real Time Factor) metric, such as `RTFX = 1/RTF`.

Latency is defined as the delay between the availability of the last chunk of audio and the reception of the inferred text. More precisely, it is defined such as :

1. _Client:_ Last audio chunk available
2. **\*t0** <- Current time\*
3. _Client:_ Send last audio chunk
4. _Server:_ Compute inference of last chunk
5. _Server:_ Generate the raw lattice for the full utterance
6. _Server:_ Determinize the raw lattice
7. _Client:_ Receive lattice output
8. _Client:_ Call callback with output
9. **\*t1** <- Current time\*

The latency is defined such as `latency = t1 - t0`.

### Results

| GPU  | Realtime I/O | Number of parallel audio channels | Latency (s) |      |      |      |
| ---- | ------------ | --------------------------------- | ----------- | ---- | ---- | ---- |
|      |              |                                   | 90%         | 95%  | 99%  | Avg  |
| A100 | Yes          | 2000                              | 0.11        | 0.12 | 0.14 | 0.09 |
| V100 | Yes          | 2000                              | 0.42        | 0.50 | 0.61 | 0.23 |
| V100 | Yes          | 1000                              | 0.09        | 0.09 | 0.11 | 0.07 |
| T4   | Yes          | 600                               | 0.17        | 0.18 | 0.22 | 0.14 |
| T4   | Yes          | 400                               | 0.12        | 0.13 | 0.15 | 0.10 |
