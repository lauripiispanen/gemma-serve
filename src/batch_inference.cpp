#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <numeric>
#include "llama.h"
#include "common.h"

constexpr int BATCH_MAX_TOKENS_OUT = 512;

PromptStats process_prompt(
    std::unique_ptr<llama_context, decltype(&llama_free)> &ctx,
    const llama_model *model,
    const std::string &prompt,
    std::ofstream &output_file,
    int prompt_id)
{
  PromptStats stats;
  stats.prompt_id = prompt_id;

  std::vector<llama_token> tokens;
  int n_tokens = 0;

  stats.t_tokenize_done = std::chrono::high_resolution_clock::now();

  if (process_prompt(ctx.get(), model, prompt, tokens, n_tokens) != 0)
  {
    return stats;
  }

  stats.input_tokens = n_tokens;

  std::string output = generate_text(
      ctx.get(),
      model,
      tokens,
      n_tokens,
      stats,
      BATCH_MAX_TOKENS_OUT);

  // Write prompt and output to file
  output_file << "Prompt " << prompt_id << ":\n";
  output_file << "Input: " << prompt << "\n";
  output_file << "Output: " << output << "\n\n";

  return stats;
}

int main(int argc, char **argv)
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <path_to_model> <prompts_file> <output_file>" << std::endl;
    return 1;
  }

  const char *model_path = argv[1];
  const char *prompts_file_path = argv[2];
  const char *output_file_path = argv[3];

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

  std::cout << "Starting batch inference..." << std::endl;

  BatchStats batch_stats;
  batch_stats.batch_start = std::chrono::high_resolution_clock::now();

  std::string prompt;
  int prompt_id = 0;

  while (std::getline(prompts_file, prompt))
  {
    if (prompt.empty())
    {
      continue;
    }

    prompt_id++;
    std::cout << "Processing prompt " << prompt_id << "..." << std::endl;

    PromptStats prompt_stats = process_prompt(ctx, model.get(), prompt, output_file, prompt_id);
    prompt_stats.print_batch();
    batch_stats.add_prompt_stats(prompt_stats);
  }

  batch_stats.batch_end = std::chrono::high_resolution_clock::now();
  batch_stats.print_summary();

  std::cout << "Batch processing complete. Results written to " << output_file_path << std::endl;
  return 0;
}