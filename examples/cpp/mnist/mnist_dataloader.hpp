#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/tensor.hpp"

namespace fs = std::filesystem;

#ifndef NEURAL_NETWORK_PROJECT_ROOT
#define NEURAL_NETWORK_PROJECT_ROOT "."
#endif

class MnistDatasetLoader {
  public:
    struct RawDataset {
        std::vector<float> images;
        std::vector<int> labels;
        std::size_t rows = 0;
        std::size_t cols = 0;
    };

    struct BatchTensors {
        std::vector<std::shared_ptr<Tensor>> images;
        std::vector<std::shared_ptr<Tensor>> labels;
    };

    explicit MnistDatasetLoader(fs::path dataset_dir = fs::path(NEURAL_NETWORK_PROJECT_ROOT) / "examples" / "data" / "mnist")
        : dataset_dir_(std::move(dataset_dir)) {
        if (!dataset_dir_.is_absolute()) {
            dataset_dir_ = fs::weakly_canonical(fs::absolute(dataset_dir_));
        }
    }

    void load_training_set() {
        training_ = load_dataset({"train-images-idx3-ubyte", "train-images.idx3-ubyte"}, {"train-labels-idx1-ubyte", "train-labels.idx1-ubyte"});
        training_images_ = to_tensor(training_);
        training_labels_ = one_hot_labels(training_);
    }

    void load_test_set() {
        test_ = load_dataset({"t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"}, {"t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"});
        test_images_ = to_tensor(test_);
        test_labels_ = one_hot_labels(test_);
    }

    const std::shared_ptr<Tensor> &training_images() const { return require_tensor(training_images_, "training images"); }
    const std::shared_ptr<Tensor> &training_labels() const { return require_tensor(training_labels_, "training labels"); }
    const std::shared_ptr<Tensor> &test_images() const { return require_tensor(test_images_, "test images"); }
    const std::shared_ptr<Tensor> &test_labels() const { return require_tensor(test_labels_, "test labels"); }

    std::shared_ptr<Tensor> flatten(bool training = true) {
        auto &tensor = training ? training_images_ : test_images_;
        if (!tensor) {
            throw std::logic_error("Load the dataset before calling flatten");
        }
        tensor = flatten(tensor);
        return tensor;
    }

    std::shared_ptr<Tensor> unflatten(std::size_t height, std::size_t width, bool training = true) {
        auto &tensor = training ? training_images_ : test_images_;
        if (!tensor) {
            throw std::logic_error("Load the dataset before calling unflatten");
        }
        tensor = unflatten(tensor, height, width);
        return tensor;
    }

    BatchTensors make_batches(std::size_t batch_size, bool training = true) const {
        const auto &images = training ? training_images_ : test_images_;
        const auto &labels = training ? training_labels_ : test_labels_;
        return {make_batches(images, batch_size), make_batches(labels, batch_size)};
    }

    std::shared_ptr<Tensor> to_tensor(const RawDataset &dataset) const {
        const std::size_t sample_count = dataset.labels.size();
        const std::size_t channels = 1;
        const std::size_t height = dataset.rows;
        const std::size_t width = dataset.cols;
        const std::size_t pixels_per_image = height * width;

        auto tensor = std::make_shared<Tensor>(std::vector<size_t>{sample_count, channels, height, width});
        for (std::size_t sample = 0; sample < sample_count; ++sample) {
            const std::size_t sample_offset = sample * pixels_per_image;
            for (std::size_t pixel = 0; pixel < pixels_per_image; ++pixel) {
                tensor->set_data(sample * channels * height * width + pixel, dataset.images[sample_offset + pixel] / 255.0f);
            }
        }
        return tensor;
    }

    std::shared_ptr<Tensor> flatten(const std::shared_ptr<Tensor> &tensor) const {
        if (!tensor) {
            throw std::invalid_argument("flatten called with null tensor");
        }
        if (tensor->shape().size() != 4) {
            throw std::invalid_argument("flatten expects a tensor shaped [N, C, H, W]");
        }
        const auto shape = tensor->shape();
        if (shape[1] != 1) {
            throw std::invalid_argument("flatten currently supports single-channel tensors only");
        }
        Tensor flattened = tensor->clone();
        flattened.reshape(std::vector<size_t>{shape[0], shape[1] * shape[2] * shape[3]});
        return std::make_shared<Tensor>(flattened);
    }

    std::shared_ptr<Tensor> unflatten(const std::shared_ptr<Tensor> &tensor, std::size_t height, std::size_t width) const {
        if (!tensor) {
            throw std::invalid_argument("unflatten called with null tensor");
        }
        if (tensor->shape().size() != 2) {
            throw std::invalid_argument("unflatten expects a tensor shaped [N, H*W]");
        }
        const auto shape = tensor->shape();
        if (shape[1] != height * width) {
            throw std::invalid_argument("flattened size does not match requested height * width");
        }
        Tensor restored = tensor->clone();
        restored.reshape(std::vector<size_t>{shape[0], 1, height, width});
        return std::make_shared<Tensor>(restored);
    }

    std::shared_ptr<Tensor> one_hot_labels(const RawDataset &dataset, std::size_t num_classes = 10) const {
        auto labels = std::make_shared<Tensor>(std::vector<size_t>{dataset.labels.size(), num_classes});
        for (std::size_t i = 0; i < dataset.labels.size(); ++i) {
            const int label = dataset.labels[i];
            if (label < 0 || static_cast<std::size_t>(label) >= num_classes) {
                throw std::out_of_range("MNIST label out of range for one-hot encoding");
            }
            for (std::size_t cls = 0; cls < num_classes; ++cls) {
                labels->set_data(i * num_classes + cls, cls == static_cast<std::size_t>(label) ? 1.0f : 0.0f);
            }
        }
        return labels;
    }

    std::vector<std::shared_ptr<Tensor>> make_batches(const std::shared_ptr<Tensor> &tensor, std::size_t batch_size) const {
        if (!tensor) {
            throw std::invalid_argument("make_batches called with null tensor");
        }
        if (batch_size == 0) {
            throw std::invalid_argument("batch_size must be greater than zero");
        }

        const auto shape = tensor->shape();
        const std::size_t n = shape[0];
        const std::size_t num_batches = (n + batch_size - 1) / batch_size;
        if (shape.size() != 2 && shape.size() != 3 && shape.size() != 4) {
            throw std::invalid_argument("make_batches expects a tensor shaped [N, C, H, W], [N, C, H*W], or [N, classes]");
        }

        std::vector<size_t> batch_shape(shape.begin(), shape.end());
        std::vector<std::shared_ptr<Tensor>> batches;
        batches.reserve(num_batches);
        const std::size_t sample_size = tensor->size() / n;

        for (std::size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            const std::size_t start = batch_idx * batch_size;
            const std::size_t end = std::min(start + batch_size, n);
            batch_shape[0] = end - start;
            auto batch = std::make_shared<Tensor>(batch_shape);
            for (std::size_t sample = start; sample < end; ++sample) {
                const std::size_t source_offset = sample * sample_size;
                const std::size_t destination_offset = (sample - start) * sample_size;
                for (std::size_t element = 0; element < sample_size; ++element) {
                    batch->set_data(destination_offset + element, tensor->data()[source_offset + element]);
                }
            }
            batches.push_back(batch);
        }
        return batches;
    }

  private:
    fs::path dataset_dir_;
    RawDataset training_;
    RawDataset test_;
    std::shared_ptr<Tensor> training_images_;
    std::shared_ptr<Tensor> training_labels_;
    std::shared_ptr<Tensor> test_images_;
    std::shared_ptr<Tensor> test_labels_;

    static const std::shared_ptr<Tensor> &require_tensor(const std::shared_ptr<Tensor> &tensor, const char *name) {
        if (!tensor) {
            throw std::logic_error(std::string("Load the dataset before requesting ") + name);
        }
        return tensor;
    }

    static std::uint32_t read_u32_be(std::ifstream &stream) {
        std::array<std::uint8_t, 4> bytes{};
        stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) {
            throw std::runtime_error("Unable to read MNIST header");
        }
        return (static_cast<std::uint32_t>(bytes[0]) << 24) | (static_cast<std::uint32_t>(bytes[1]) << 16) |
               (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
    }

    static std::uint8_t read_u8(std::ifstream &stream) {
        std::uint8_t value = 0;
        stream.read(reinterpret_cast<char *>(&value), 1);
        if (!stream) {
            throw std::runtime_error("Unable to read MNIST byte");
        }
        return value;
    }

    static std::string alternate_idx_name(const std::string &filename) {
        const std::string needle = "-idx";
        const std::size_t pos = filename.find(needle);
        if (pos == std::string::npos) {
            return filename;
        }
        return filename.substr(0, pos) + ".idx" + filename.substr(pos + needle.size());
    }

    RawDataset load_dataset(const std::vector<std::string> &image_candidates, const std::vector<std::string> &label_candidates) const {
        std::string image_filename;
        std::string label_filename;
        for (const auto &candidate : image_candidates) {
            if (fs::is_regular_file(dataset_dir_ / candidate)) {
                image_filename = candidate;
                break;
            }
        }
        for (const auto &candidate : label_candidates) {
            if (fs::is_regular_file(dataset_dir_ / candidate)) {
                label_filename = candidate;
                break;
            }
        }
        if (image_filename.empty() || label_filename.empty()) {
            const fs::path expected_dir = fs::weakly_canonical(fs::absolute(dataset_dir_));
            throw std::runtime_error("MNIST binary files not found in " + expected_dir.string());
        }

        std::ifstream image_stream(dataset_dir_ / image_filename, std::ios::binary);
        std::ifstream label_stream(dataset_dir_ / label_filename, std::ios::binary);
        if (!image_stream || !label_stream) {
            throw std::runtime_error("Unable to open MNIST dataset files");
        }

        const std::uint32_t magic_images = read_u32_be(image_stream);
        const std::uint32_t num_images = read_u32_be(image_stream);
        const std::uint32_t rows = read_u32_be(image_stream);
        const std::uint32_t cols = read_u32_be(image_stream);
        const std::uint32_t magic_labels = read_u32_be(label_stream);
        const std::uint32_t num_labels = read_u32_be(label_stream);
        if (magic_images != 0x00000803u || magic_labels != 0x00000801u) {
            throw std::runtime_error("Unexpected MNIST magic values");
        }
        if (num_images != num_labels) {
            throw std::runtime_error("Image and label counts do not match");
        }

        RawDataset dataset;
        dataset.rows = rows;
        dataset.cols = cols;
        dataset.images.resize(static_cast<std::size_t>(num_images) * rows * cols);
        dataset.labels.resize(num_labels);
        for (float &pixel : dataset.images) {
            pixel = static_cast<float>(read_u8(image_stream));
        }
        for (int &label : dataset.labels) {
            label = static_cast<int>(read_u8(label_stream));
        }
        return dataset;
    }
};