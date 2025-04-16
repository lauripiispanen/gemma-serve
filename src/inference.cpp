// src/inference.cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "llama.h"

constexpr int MAX_TOKENS_OUT = 128;
constexpr int TOKEN_BUFFER_PADDING = 32;

struct Stats
{
  int input_tokens = 0;
  int output_tokens = 0;
  std::chrono::high_resolution_clock::time_point t_start;
  std::chrono::high_resolution_clock::time_point t_tokenize_done;
  std::chrono::high_resolution_clock::time_point t_prefill_done;
  std::chrono::high_resolution_clock::time_point t_first_token;
  std::chrono::high_resolution_clock::time_point t_last_token;

  void print() const
  {
    using namespace std::chrono;

    const auto input_time = duration_cast<milliseconds>(t_prefill_done - t_tokenize_done).count();
    const auto ttft = duration_cast<milliseconds>(t_first_token - t_prefill_done).count();
    const auto gen_time = duration_cast<milliseconds>(t_last_token - t_first_token).count();
    const auto total_time = duration_cast<milliseconds>(t_last_token - t_start).count();

    std::cout << "\n\n--- Inference Stats ---\n";
    std::cout << "Input tokens     : " << input_tokens << "\n";
    std::cout << "Output tokens    : " << output_tokens << "\n";
    std::cout << "Time to first token (TTFT): " << ttft << " ms\n";
    std::cout << "Input processing : " << input_time << " ms\n";
    std::cout << "Output generation: " << gen_time << " ms\n";
    std::cout << "Total time       : " << total_time << " ms\n";
    if (gen_time > 0)
    {
      const double tps = static_cast<double>(output_tokens) / (gen_time / 1000.0);
      std::cout << "Tokens/sec       : " << tps << "\n";
    }
    std::cout << "------------------------\n";
  }
};

class ScopedBatch
{
private:
  llama_batch batch;
  bool initialized = false;

public:
  // Constructor for a single token batch
  ScopedBatch(llama_token token, llama_pos pos)
  {
    batch = llama_batch_init(1, 0, 1);
    if (!batch.token || !batch.pos || !batch.logits || !batch.n_seq_id || !batch.seq_id)
    {
      std::cerr << "Failed to initialize batch\n";
      return;
    }

    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = pos;
    batch.logits[0] = 1; // Request logits for this token
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;

    initialized = true;
  }

  ScopedBatch(const ScopedBatch &) = delete;
  ScopedBatch &operator=(const ScopedBatch &) = delete;
  ScopedBatch(ScopedBatch &&) = delete;
  ScopedBatch &operator=(ScopedBatch &&) = delete;

  ~ScopedBatch()
  {
    if (initialized)
    {
      llama_batch_free(batch);
    }
  }

  llama_batch *get() { return initialized ? &batch : nullptr; }

  bool is_valid() const { return initialized; }
};

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " <path_to_model> \"<prompt_text>\"" << std::endl;
    return 1;
  }

  const char *model_path = argv[1];
  std::cout << "Loading model from " << model_path << std::endl;
  const std::string prompt = argv[2];

  llama_model_params model_params = llama_model_default_params();
  llama_context_params ctx_params = llama_context_default_params();

  std::unique_ptr<llama_model, decltype(&llama_model_free)>
      model(llama_model_load_from_file(model_path, model_params), llama_model_free);

  if (!model)
  {
    std::cerr << "Failed to load model." << std::endl;
    return 1;
  }

  std::unique_ptr<llama_context, decltype(&llama_free)>
      ctx(llama_init_from_model(model.get(), ctx_params), llama_free);

  if (!ctx)
  {
    std::cerr << "Failed to create context." << std::endl;
    return 1;
  }

  const llama_vocab *vocab = llama_model_get_vocab(model.get());

  std::vector<llama_token> tokens(prompt.size() + TOKEN_BUFFER_PADDING);

  Stats stats;
  stats.t_start = std::chrono::high_resolution_clock::now();

  const int n_tokens = llama_tokenize(
      vocab,
      prompt.c_str(),
      static_cast<int>(prompt.size()),
      tokens.data(),
      static_cast<int>(tokens.size()),
      true, // Add BOS/EOS if model needs it
      true  // Parse special tokens
  );

  if (n_tokens < 0)
  {
    std::cerr << "Tokenization failed or buffer too small" << std::endl;
    return 1;
  }

  stats.t_tokenize_done = std::chrono::high_resolution_clock::now();
  stats.input_tokens = n_tokens;

  tokens.resize(n_tokens);

  llama_batch batch = llama_batch_init(n_tokens, 0, 1);
  if (!batch.token || !batch.logits || !batch.n_seq_id || !batch.seq_id)
  {
    std::cerr << "Failed to initialize batch" << std::endl;
    return 1;
  }

  batch.n_tokens = n_tokens;

  for (int i = 0; i < n_tokens; i++)
  {
    batch.token[i] = tokens[i];
  }

  batch.pos = nullptr;

  for (int i = 0; i < n_tokens; i++)
  {
    batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
  }

  for (int i = 0; i < n_tokens; i++)
  {
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
  }

  const int prefill_result = llama_decode(ctx.get(), batch);
  if (prefill_result != 0)
  {
    std::cerr << "Failed to evaluate prompt, error code: " << prefill_result << std::endl;
    llama_batch_free(batch);
    return 1;
  }

  stats.t_prefill_done = std::chrono::high_resolution_clock::now();
  llama_batch_free(batch);

  auto sparams = llama_sampler_chain_default_params();
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>
      sampler(llama_sampler_chain_init(sparams), llama_sampler_free);

  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(64));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.95, 1));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(1.0));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(1234));

  const int vocab_size = llama_vocab_n_tokens(vocab);
  std::cout << "Model vocab size: " << vocab_size << std::endl;

  // Generation loop
  for (int i = 0; i < MAX_TOKENS_OUT; ++i)
  {
    const llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);

    if (i == 0)
    {
      stats.t_first_token = std::chrono::high_resolution_clock::now();
    }

    char buffer[128];
    const int len = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, false);
    if (len < 0)
    {
      break;
    }

    std::cout << std::string(buffer, len) << std::flush;
    stats.output_tokens++;

    ScopedBatch next_batch(token, static_cast<llama_pos>(tokens.size() + i));
    if (!next_batch.is_valid())
    {
      std::cerr << "Failed to create batch for next token" << std::endl;
      break;
    }

    const int decode_result = llama_decode(ctx.get(), *next_batch.get());
    if (decode_result != 0)
    {
      std::cerr << "\nFailed to evaluate next token, error code: " << decode_result << std::endl;
      break;
    }

    llama_sampler_accept(sampler.get(), token);

    if (token == llama_vocab_eos(vocab))
    {
      std::cout << "\n[Reached <eos> token]\n";
      break;
    }
  }

  stats.t_last_token = std::chrono::high_resolution_clock::now();
  stats.print();

  std::cout << "\nDone.\n";
  return 0;
}