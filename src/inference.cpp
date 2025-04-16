// src/inference.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include "common.h"

constexpr int MAX_TOKENS_OUT = 128;

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " <path_to_model> \"<prompt_text>\"" << std::endl;
    return 1;
  }

  const char *model_path = argv[1];
  const std::string prompt = argv[2];

  auto start_time = std::chrono::high_resolution_clock::now();

  auto model = load_model(model_path);
  if (!model)
  {
    std::cerr << "Failed to load model." << std::endl;
    return 1;
  }

  auto ctx = create_context(model.get());
  if (!ctx)
  {
    std::cerr << "Failed to create context." << std::endl;
    return 1;
  }

  const llama_vocab *vocab = llama_model_get_vocab(model.get());
  const int vocab_size = llama_vocab_n_tokens(vocab);
  std::cout << "Model vocab size: " << vocab_size << std::endl;

  std::vector<std::string> prompts = {prompt};

  auto batch_result = process_batch(ctx.get(), model.get(), prompts, MAX_TOKENS_OUT);

  std::string output;
  PromptStats stats;

  if (!batch_result.outputs.empty())
  {
    output = batch_result.outputs[0];
    stats = batch_result.stats[0];
  }
  else
  {
    std::cerr << "Error: Failed to generate output" << std::endl;
    return 1;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  std::cout << output << std::endl;
  stats.print_single();
  std::cout << "Total time (including model loading): " << total_duration << " ms\n";
  std::cout << "\nDone.\n";

  return 0;
}