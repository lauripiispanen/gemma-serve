#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <numeric>
#include <queue>
#include "llama.h"
#include "common.h"

constexpr int BATCH_MAX_TOKENS_OUT = 512;
constexpr int DEFAULT_BATCH_SIZE = 4;

int main(int argc, char **argv)
{
  int batch_size = DEFAULT_BATCH_SIZE;

  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <path_to_model> <prompts_file> <output_file> [batch_size]" << std::endl;
    std::cerr << "  batch_size: Number of prompts to process in parallel (default: " << DEFAULT_BATCH_SIZE << ")" << std::endl;
    return 1;
  }

  const char *model_path = argv[1];
  const char *prompts_file_path = argv[2];
  const char *output_file_path = argv[3];

  if (argc > 4)
  {
    batch_size = std::stoi(argv[4]);
    if (batch_size <= 0)
    {
      std::cerr << "Invalid batch size. Using default: " << DEFAULT_BATCH_SIZE << std::endl;
      batch_size = DEFAULT_BATCH_SIZE;
    }
  }

  std::ifstream prompts_file(prompts_file_path);
  if (!prompts_file.is_open())
  {
    std::cerr << "Failed to open prompts file: " << prompts_file_path << std::endl;
    return 1;
  }

  std::ofstream output_file(output_file_path);
  if (!output_file.is_open())
  {
    std::cerr << "Failed to open output file: " << output_file_path << std::endl;
    return 1;
  }

  std::cout << "Loading model from " << model_path << std::endl;

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

  std::cout << "Starting batch inference with " << batch_size << " prompts per batch..." << std::endl;

  BatchStats batch_stats;
  batch_stats.batch_start = std::chrono::high_resolution_clock::now();

  std::string prompt;
  std::vector<std::string> prompt_batch;
  std::vector<int> prompt_ids;
  int next_prompt_id = 1;

  std::queue<std::pair<int, std::string>> all_prompts;
  while (std::getline(prompts_file, prompt))
  {
    if (!prompt.empty())
    {
      all_prompts.push({next_prompt_id++, prompt});
    }
  }

  while (!all_prompts.empty())
  {
    prompt_batch.clear();
    prompt_ids.clear();

    int batch_prompt_count = std::min(batch_size, static_cast<int>(all_prompts.size()));
    for (int i = 0; i < batch_prompt_count; i++)
    {
      auto [id, text] = all_prompts.front();
      all_prompts.pop();
      prompt_batch.push_back(text);
      prompt_ids.push_back(id);
    }

    std::cout << "Processing batch of " << batch_prompt_count << " prompts..." << std::endl;

    auto batch_result = process_batch(ctx.get(), model.get(), prompt_batch, BATCH_MAX_TOKENS_OUT);

    for (size_t i = 0; i < batch_result.outputs.size(); i++)
    {
      int prompt_id = prompt_ids[i];
      std::string output = batch_result.outputs[i];
      PromptStats stats = batch_result.stats[i];
      stats.prompt_id = prompt_id;

      output_file << "Prompt " << prompt_id << ":\n";
      output_file << "Input: " << prompt_batch[i] << "\n";
      output_file << "Output: " << output << "\n\n";

      stats.print_batch();
      batch_stats.add_prompt_stats(stats);
    }
  }

  batch_stats.batch_end = std::chrono::high_resolution_clock::now();
  batch_stats.print_summary();

  std::cout << "Batch processing complete. Results written to " << output_file_path << std::endl;
  return 0;
}