#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stddef.h>

// Mean Squared Error (scalar output)
double mse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double error = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_pred[i] - y_true[i];
        error += diff * diff;
    }
    return error / y_true.size();
}

// Derivative of MSE w.r.t y_pred
std::vector<double> mse_prime(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    std::vector<double> gradient(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        gradient[i] = 2.0 * (y_pred[i] - y_true[i]) / y_true.size();
    }
    return gradient;
}

class TanhActivation {
private:
    std::vector<double> input_cache;
    std::vector<double> output_cache;

public:
    std::vector<double> forward(const std::vector<double>& input) {
        input_cache = input;
        output_cache.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output_cache[i] = std::tanh(input[i]);
        }
        return output_cache;
    }

    std::vector<double> backward(const std::vector<double>& output_gradient) {
        std::vector<double> input_gradient(output_cache.size());
        for (size_t i = 0; i < output_cache.size(); ++i) {
            input_gradient[i] = output_gradient[i] * (1 - output_cache[i] * output_cache[i]);
        }
        return input_gradient;
    }
};

// Initialize random double between -1 and 1
double rand_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// Dense layer class
class Dense {
private:
    int input_size, output_size;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> input;

public:
    Dense(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
        weights.resize(output_size, std::vector<double>(input_size));
        biases.resize(output_size);

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights[i][j] = rand_weight();
            }
            biases[i] = rand_weight();
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        this->input = input;
        std::vector<double> output(output_size, 0.0);
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                output[i] += weights[i][j] * input[j];
            }
            output[i] += biases[i];
        }
        return output;
    }

    std::vector<double> backward(const std::vector<double>& output_gradient, double learning_rate) {
        std::vector<std::vector<double>> weight_gradients(output_size, std::vector<double>(input_size, 0.0));
        std::vector<double> input_gradient(input_size, 0.0);

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weight_gradients[i][j] = output_gradient[i] * input[j];
                input_gradient[j] += output_gradient[i] * weights[i][j];
            }
            biases[i] -= learning_rate * output_gradient[i];
        }

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights[i][j] -= learning_rate * weight_gradients[i][j];
            }
        }

        return input_gradient;
    }
};

int main() {
    std::srand(std::time(0));

    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<std::vector<double>> outputs = {
        {0.0}, {1.0}, {1.0}, {0.0}
    };

    Dense layer1(2, 3);
    TanhActivation activation1;
    Dense layer2(3, 1);
    TanhActivation activation2;

    int epochs = 1000;
    double learning_rate = 0.1;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> hidden_output = layer1.forward(inputs[i]);
            hidden_output = activation1.forward(hidden_output);
            std::vector<double> output = layer2.forward(hidden_output);
            output = activation2.forward(output);

            total_error += mse(outputs[i], output);

            std::vector<double> grad = mse_prime(outputs[i], output);
            grad = activation2.backward(grad);
            std::vector<double> hidden_grad = layer2.backward(grad, learning_rate);
            hidden_grad = activation1.backward(hidden_grad);
            layer1.backward(hidden_grad, learning_rate);
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Error: " << total_error / inputs.size() << "\n";
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::vector<double> hidden_output = layer1.forward(inputs[i]);
                hidden_output = activation1.forward(hidden_output);
                std::vector<double> output = layer2.forward(hidden_output);
                output = activation2.forward(output);
                std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "], Output: " << output[0] << ", True: " << outputs[i][0] << "\n";
            }
            std::cout << "-----------------------------------\n";
        }
    }

    std::cout << "\nTesting the model:\n";
    int correct_predictions = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> hidden_output = layer1.forward(inputs[i]);
        hidden_output = activation1.forward(hidden_output);
        std::vector<double> output = layer2.forward(hidden_output);
        output = activation2.forward(output);
        double predicted = output[0] >= 0.5 ? 1.0 : 0.0;
        if (predicted == outputs[i][0]) correct_predictions++;
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "], Predicted: " << predicted << ", True: " << outputs[i][0] << "\n";
    }

    double accuracy = static_cast<double>(correct_predictions) / inputs.size() * 100.0;
    std::cout << "\nAccuracy: " << accuracy << "%\n";

    return 0;
}