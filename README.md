# EdgeML: Comprehensive EdgeAI Metrics & Hardware Deployment Framework

An end-to-end repository and experimental framework dedicated to implementing, benchmarking, and deploying optimized Machine Learning models within severely resource-constrained edge hardware environments. This project bridges the gap between high-level model design and low-level bare-metal execution, covering the entire pipeline from training hardware-aware Convolutional Neural Networks (CNNs) to evaluating micro-architectural inference performance. 

By prioritizing Size, Weight, and Power (SWaP) constraints alongside traditional accuracy metrics, EdgeML provides a robust methodology for translating theoretical AI models into practical, real-world IoT and embedded sensor applications.

## ✨ Key Features & Capabilities

* **Hardware-Aware Model Design:** Custom training pipelines optimized for minimal parameter counts and reduced computational complexity.
* **Bare-Metal C++ Inference:** Highly optimized, low-overhead C++ execution scripts designed to bypass bloated operating systems and run directly on microcontrollers.
* **Empirical Benchmarking:** Rigorous profiling of deployed models to capture real-world execution metrics rather than simulated estimates.
* **Cross-Platform Adaptability:** A modular architecture designed to evaluate performance across a heterogeneous mix of development boards and edge processors.

---

## 📂 Repository Structure & Architecture

The `EDGE ML/` directory acts as the central hub for core research, algorithmic implementation, and hardware documentation. The repository is logically divided into execution environments, analytical tracking, and foundational research:

### 🧠 Core Implementation
* **`CNN.ipynb`**: A comprehensive Jupyter Notebook containing the model's architectural definition, custom training loops, and preprocessing pipelines. This notebook incorporates edge-specific optimization techniques (such as weight pruning and quantization paradigms) to prepare the CNN for deployment without severe accuracy degradation.
* **`main.cpp`**: The primary C++ source code engineered for low-latency, deterministic inference. This file contains the memory-safe execution logic required to parse input vectors, compute neural network layers, benchmark execution time at the clock-cycle level, and interface seamlessly with target hardware peripherals.

### 📚 Documentation & Research Frameworks
* **`EDGEAI Metrics Framework canva.pdf` & `Famework Doc.docx`**: Exhaustive architectural guides detailing the exact evaluation methodologies, statistical formulas, and benchmarking criteria (latency, dynamic memory footprint, static storage, and power efficiency) utilized to validate edge models.
* **`EdgeGlove.pdf`**: Specialized documentation relating to specific hardware interfaces and applied project verticals—specifically focusing on wearable edge sensors and localized gesture-recognition pipelines.
* **`EDGE ML First Report.pdf` & `questions.pdf`**: Foundational project milestones, theoretical research questions, experimental hypotheses, and structured initial reporting regarding the feasibility of specific embedded AI deployments.

### 📈 Data Repositories & Empirical Tracking
* **`Metrics Measurement.xlsx`**: A master dataset aggregating empirical profiling metrics collected directly from hardware, allowing for complex multi-variable analysis of model performance.
* **`Boards my edit.xlsx`**: A detailed hardware matrix tracking the specifications, memory constraints, clock speeds, and architectural nuances (e.g., ARM Cortex-M vs. RISC-V) of the target development boards.

---

## 📊 Comprehensive Edge AI Metrics Framework

Standard ML evaluation relies heavily on validation accuracy, precision, and recall. However, operationalizing models on microcontrollers requires a rigorous, multi-dimensional assessment. This project systematically tracks, benchmarks, and optimizes the following key performance indicators (KPIs):

1. **Inference Latency (Throughput):** Measuring the absolute execution time per sample in milliseconds (ms) or frames-per-second (FPS). This includes assessing the latency introduced by memory access bottlenecks and arithmetic logic unit (ALU) computations.
2. **Dynamic Memory Footprint (SRAM):** Tracking the peak RAM utilization during inference. Since microcontrollers frequently possess less than 256KB of SRAM, minimizing the memory required for storing intermediate tensor activations is critical.
3. **Static Storage Overhead (Flash):** Calculating the compiled binary size and the read-only memory (ROM) required to store the neural network weights, biases, and the underlying C++ inference engine.
4. **Energy Efficiency & Power Draw:** (Where applicable) Estimating the energy consumed per inference (measured in microjoules, µJ) to ensure viability in battery-powered or energy-harvesting environments.
5. **Hardware Adaptability:** Conducting comparative analysis to determine how architectural constraints impact the same model across different deployment ecosystems.

---

## 🛠️ Getting Started & Deployment Guide

Follow these instructions to replicate the training environment and deploy the compiled inference engine.

### 1. Python Environment (Model Training, Optimization, & Simulation)
To run the Jupyter training notebooks, preprocess the datasets, and export the optimized model weights, ensure you have Python 3.8+ installed on your host machine.

```bash
# 1. Clone the repository to your local machine
git clone https://github.com/ArshithaRajkumar/EdgeML.git

# 2. Navigate into the repository directory
cd EdgeML

# 3. Initialize an isolated virtual environment to prevent dependency conflicts
python -m venv venv

# 4. Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate  
# On Windows Command Prompt / PowerShell:
venv\Scripts\activate

# 5. Launch the Jupyter interactive development environment
jupyter notebook
```

### 2. C++ Edge Inference Deployment (Local Testing & Compilation)
The `main.cpp` file contains the firmware logic required for hardware execution. Before flashing to a microcontroller, it is highly recommended to compile and benchmark the inference engine locally on your host machine to verify logical correctness.

```bash
# 1. Navigate to the specific working directory containing the C++ source
cd "EDGE ML"

# 2. Compile the source code. 
# The -O3 flag applies maximum compiler optimizations for execution speed.
g++ -O3 main.cpp -o edge_inference

# 3. Execute the compiled binary to run the local inference benchmark
./edge_inference
```

*(Note: For actual hardware deployment, this C++ code will need to be cross-compiled using the specific toolchain for your target board, such as `arm-none-eabi-gcc` for STM32 devices, or integrated into an IDE like Arduino IDE or STM32CubeIDE).*

---

## 🚀 Future Roadmap
* Integration of INT8 post-training quantization to further reduce memory footprint.
* Expansion of `main.cpp` to support hardware-accelerated vector instructions (e.g., ARM CMSIS-NN).
* Automated CI/CD pipelines for testing model performance across virtualized hardware environments.

## 📜 License
This software is provided under the open-source **GPL-3.0 License**. This guarantees the freedom to study, share, and modify the software, ensuring that all derivative works remain free and open. Please consult the `LICENSE` file in the root directory for full legal details and terms of use.
