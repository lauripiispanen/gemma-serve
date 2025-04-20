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
#include "file_prompt_source.h"
constexpr int BATCH_MAX_TOKENS_OUT = 500;
constexpr int DEFAULT_BATCH_SIZE = 4;

int main(int argc, char **argv)
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <model_path> <prompts.txt> <output.txt>\n";
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string prompt_path = argv[2];
  const std::string output_path = argv[3];
  const int batch_size = argc > 4 ? std::stoi(argv[4]) : 4;

  auto model = load_model(model_path.c_str());
  auto ctx = create_context(model.get());

  FilePromptSource source(prompt_path);
  std::ofstream output_file(output_path);

  auto stats = process_continuous_batch(ctx.get(), model.get(), source, output_file, batch_size, BATCH_MAX_TOKENS_OUT);
  stats.print_summary();
  return 0;
}