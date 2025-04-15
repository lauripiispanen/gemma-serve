// src/inference.cpp
#include <iostream>
#include "llama.h"

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
    return 1;
  }

  llama_model_params model_params = llama_model_default_params();
  llama_context_params ctx_params = llama_context_default_params();

  // Adjust paths as needed
  const char *model_path = argv[1];
  std::cout << "Loading model from " << model_path << std::endl;

  llama_model *model = llama_model_load_from_file(model_path, model_params);
  if (!model)
  {
    std::cerr << "Failed to load model." << std::endl;
    return 1;
  }

  llama_context *ctx = llama_init_from_model(model, ctx_params);
  if (!ctx)
  {
    std::cerr << "Failed to create context." << std::endl;
    return 1;
  }

  std::cout << "Model loaded and context created successfully!" << std::endl;

  llama_free(ctx);
  llama_model_free(model);
  return 0;
}