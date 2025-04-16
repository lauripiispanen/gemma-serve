#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "llama.h"

constexpr int MAX_TOKENS_OUT_DEFAULT = 128;
constexpr int TOKEN_BUFFER_PADDING = 32;

/**
 * RAII wrapper for llama_batch to ensure proper cleanup
 */
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

struct PromptStats
{
  int prompt_id = 0;
  int input_tokens = 0;
  int output_tokens = 0;

  std::chrono::high_resolution_clock::time_point t_start;
  std::chrono::high_resolution_clock::time_point t_tokenize_done;
  std::chrono::high_resolution_clock::time_point t_prefill_done;
  std::chrono::high_resolution_clock::time_point t_first_token;
  std::chrono::high_resolution_clock::time_point t_last_token;

  std::chrono::milliseconds tokenize_time{0};
  std::chrono::milliseconds prefill_time{0};
  std::chrono::milliseconds ttft{0};
  std::chrono::milliseconds generation_time{0};
  std::chrono::milliseconds total_time{0};

  PromptStats()
  {
    t_start = std::chrono::high_resolution_clock::now();
  }

  void finalize()
  {
    using namespace std::chrono;

    if (tokenize_time.count() == 0 && t_tokenize_done.time_since_epoch().count() > 0)
    {
      tokenize_time = duration_cast<milliseconds>(t_tokenize_done - t_start);
    }

    if (prefill_time.count() == 0 && t_prefill_done.time_since_epoch().count() > 0)
    {
      prefill_time = duration_cast<milliseconds>(t_prefill_done - t_tokenize_done);
    }

    if (ttft.count() == 0 && t_first_token.time_since_epoch().count() > 0)
    {
      ttft = duration_cast<milliseconds>(t_first_token - t_prefill_done);
    }

    if (generation_time.count() == 0 && t_last_token.time_since_epoch().count() > 0 && t_first_token.time_since_epoch().count() > 0)
    {
      generation_time = duration_cast<milliseconds>(t_last_token - t_first_token);
    }

    if (total_time.count() == 0 && t_last_token.time_since_epoch().count() > 0)
    {
      total_time = duration_cast<milliseconds>(t_last_token - t_start);
    }
  }

  void print_single() const
  {
    using namespace std::chrono;

    std::cout << "\n\n--- Inference Stats ---\n";
    std::cout << "Input tokens     : " << input_tokens << "\n";
    std::cout << "Output tokens    : " << output_tokens << "\n";
    std::cout << "Time to first token (TTFT): " << ttft.count() << " ms\n";
    std::cout << "Input processing : " << prefill_time.count() << " ms\n";
    std::cout << "Output generation: " << generation_time.count() << " ms\n";
    std::cout << "Total time       : " << total_time.count() << " ms\n";

    if (generation_time.count() > 0)
    {
      const double tps = static_cast<double>(output_tokens) / (generation_time.count() / 1000.0);
      std::cout << "Tokens/sec       : " << tps << "\n";
    }
    std::cout << "------------------------\n";
  }

  void print_batch() const
  {
    std::cout << "Prompt " << prompt_id << ": "
              << input_tokens << " in, "
              << output_tokens << " out, "
              << "TTFT: " << ttft.count() << " ms, "
              << "Gen: " << generation_time.count() << " ms, "
              << "Total: " << total_time.count() << " ms";

    if (generation_time.count() > 0)
    {
      const double tps = static_cast<double>(output_tokens) / (generation_time.count() / 1000.0);
      std::cout << ", " << tps << " tokens/sec";
    }
    std::cout << std::endl;
  }
};

/**
 * Batch stats for aggregating multiple prompts
 */
struct BatchStats
{
  std::vector<PromptStats> prompt_stats;
  std::chrono::high_resolution_clock::time_point batch_start;
  std::chrono::high_resolution_clock::time_point batch_end;

  void add_prompt_stats(const PromptStats &stats)
  {
    prompt_stats.push_back(stats);
  }

  void print_summary() const; // Implementation in common.cpp
};

/**
 * Helper functions for model operations
 */

std::unique_ptr<llama_model, decltype(&llama_model_free)> load_model(const char *model_path);

std::unique_ptr<llama_context, decltype(&llama_free)> create_context(llama_model *model);

std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> create_sampler();

/**
 * Result structure for batch processing
 */
struct BatchProcessingResult
{
  std::vector<std::string> outputs;
  std::vector<PromptStats> stats;
};

/**
 * Process multiple prompts in a single batch
 *
 * @param ctx LLAMA context
 * @param model LLAMA model
 * @param prompts Vector of input texts to process
 * @param max_tokens_out Maximum number of tokens to generate per prompt
 * @return BatchProcessingResult containing outputs and stats for each prompt
 */
BatchProcessingResult process_batch(
    llama_context *ctx,
    const llama_model *model,
    const std::vector<std::string> &prompts,
    int max_tokens_out);