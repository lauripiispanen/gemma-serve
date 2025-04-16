// src/inference.cpp
#include <iostream>
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

  PromptStats stats;

  std::vector<llama_token> tokens;
  int n_tokens = 0;

  stats.t_tokenize_done = std::chrono::high_resolution_clock::now();

  if (process_prompt(ctx.get(), model.get(), prompt, tokens, n_tokens) != 0)
  {
    return 1;
  }

  stats.input_tokens = n_tokens;

  const llama_vocab *vocab = llama_model_get_vocab(model.get());
  const int vocab_size = llama_vocab_n_tokens(vocab);
  std::cout << "Model vocab size: " << vocab_size << std::endl;

  std::cout << generate_text(
      ctx.get(),
      model.get(),
      tokens,
      n_tokens,
      stats);

  std::cout << std::endl;
  stats.print_single();
  std::cout << "\nDone.\n";

  return 0;
}