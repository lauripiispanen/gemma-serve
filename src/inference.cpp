// src/inference.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include "common.h"
#include "single_prompt_source.h"

constexpr int MAX_TOKENS_OUT = 128;

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " <model_path> <prompt>\n";
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string prompt = argv[2];

  auto model = load_model(model_path.c_str());
  auto ctx = create_context(model.get());

  SinglePromptSource source(prompt);
  auto stats = process_continuous_batch(ctx.get(), model.get(), source, std::cout, 1, 512);
  stats.print_summary();
  return 0;
}