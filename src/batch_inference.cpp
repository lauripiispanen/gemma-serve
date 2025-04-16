#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <numeric>
#include "llama.h"

constexpr int MAX_TOKENS_OUT = 512;
constexpr int TOKEN_BUFFER_PADDING = 32;

struct PromptStats
{
  int prompt_id = 0;
  int input_tokens = 0;
  int output_tokens = 0;
  std::chrono::milliseconds tokenize_time{0};
  std::chrono::milliseconds prefill_time{0};
  std::chrono::milliseconds ttft{0}; // Time to first token
  std::chrono::milliseconds generation_time{0};
  std::chrono::milliseconds total_time{0};

  void print() const
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

struct BatchStats
{
  std::vector<PromptStats> prompt_stats;
  std::chrono::high_resolution_clock::time_point batch_start;
  std::chrono::high_resolution_clock::time_point batch_end;

  void add_prompt_stats(const PromptStats &stats)
  {
    prompt_stats.push_back(stats);
  }

  void print_summary() const
  {
    using namespace std::chrono;

    if (prompt_stats.empty())
    {
      std::cout << "No prompts processed.\n";
      return;
    }

    int total_input_tokens = 0;
    int total_output_tokens = 0;
    std::chrono::milliseconds total_tokenize_time{0};
    std::chrono::milliseconds total_prefill_time{0};
    std::chrono::milliseconds total_generation_time{0};
    std::chrono::milliseconds total_processing_time{0};

    for (const auto &stats : prompt_stats)
    {
      total_input_tokens += stats.input_tokens;
      total_output_tokens += stats.output_tokens;
      total_tokenize_time += stats.tokenize_time;
      total_prefill_time += stats.prefill_time;
      total_generation_time += stats.generation_time;
      total_processing_time += stats.total_time;
    }

    const auto batch_time = duration_cast<milliseconds>(batch_end - batch_start).count();
    const double avg_input_tokens = static_cast<double>(total_input_tokens) / prompt_stats.size();
    const double avg_output_tokens = static_cast<double>(total_output_tokens) / prompt_stats.size();
    const double avg_ttft = std::accumulate(prompt_stats.begin(), prompt_stats.end(), 0.0,
                                            [](double sum, const PromptStats &stats)
                                            {
                                              return sum + stats.ttft.count();
                                            }) /
                            prompt_stats.size();
    const double avg_gen_time = std::accumulate(prompt_stats.begin(), prompt_stats.end(), 0.0,
                                                [](double sum, const PromptStats &stats)
                                                {
                                                  return sum + stats.generation_time.count();
                                                }) /
                                prompt_stats.size();
    const double avg_total_time = std::accumulate(prompt_stats.begin(), prompt_stats.end(), 0.0,
                                                  [](double sum, const PromptStats &stats)
                                                  {
                                                    return sum + stats.total_time.count();
                                                  }) /
                                  prompt_stats.size();

    std::cout << "\n\n--- Batch Inference Stats ---\n";
    std::cout << "Total prompts processed: " << prompt_stats.size() << "\n";
    std::cout << "Total input tokens     : " << total_input_tokens << "\n";
    std::cout << "Total output tokens    : " << total_output_tokens << "\n";
    std::cout << "Avg input tokens/prompt: " << avg_input_tokens << "\n";
    std::cout << "Avg output tokens/prompt: " << avg_output_tokens << "\n";
    std::cout << "Avg TTFT               : " << avg_ttft << " ms\n";
    std::cout << "Avg generation time    : " << avg_gen_time << " ms\n";
    std::cout << "Avg prompt total time  : " << avg_total_time << " ms\n";
    std::cout << "Total batch time       : " << batch_time << " ms\n";

    if (total_generation_time.count() > 0)
    {
      const double overall_tps = static_cast<double>(total_output_tokens) / (total_generation_time.count() / 1000.0);
      std::cout << "Overall tokens/sec    : " << overall_tps << "\n";
    }

    std::cout << "Throughput            : " << (prompt_stats.size() * 1000.0 / batch_time) << " prompts/sec\n";
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

PromptStats process_prompt(
    std::unique_ptr<llama_context, decltype(&llama_free)> &ctx,
    const llama_model *model,
    const std::string &prompt,
    std::ofstream &output_file,
    int prompt_id)
{
  PromptStats stats;
  stats.prompt_id = prompt_id;

  const llama_vocab *vocab = llama_model_get_vocab(model);
  std::vector<llama_token> tokens(prompt.size() + TOKEN_BUFFER_PADDING);

  auto t_start = std::chrono::high_resolution_clock::now();

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
    std::cerr << "Tokenization failed for prompt " << prompt_id << std::endl;
    return stats;
  }

  auto t_tokenize_done = std::chrono::high_resolution_clock::now();
  stats.input_tokens = n_tokens;
  stats.tokenize_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_tokenize_done - t_start);

  tokens.resize(n_tokens);

  llama_batch batch = llama_batch_init(n_tokens, 0, 1);
  if (!batch.token || !batch.logits || !batch.n_seq_id || !batch.seq_id)
  {
    std::cerr << "Failed to initialize batch for prompt " << prompt_id << std::endl;
    return stats;
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
    std::cerr << "Failed to evaluate prompt " << prompt_id << ", error code: " << prefill_result << std::endl;
    llama_batch_free(batch);
    return stats;
  }

  auto t_prefill_done = std::chrono::high_resolution_clock::now();
  stats.prefill_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_done - t_tokenize_done);

  llama_batch_free(batch);

  auto sparams = llama_sampler_chain_default_params();
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>
      sampler(llama_sampler_chain_init(sparams), llama_sampler_free);

  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(64));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.95, 1));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(1.0));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(1234));

  std::string output;
  auto t_first_token = t_prefill_done;
  bool first_token = true;

  // Generation loop
  for (int i = 0; i < MAX_TOKENS_OUT; ++i)
  {
    const llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);

    if (first_token)
    {
      t_first_token = std::chrono::high_resolution_clock::now();
      stats.ttft = std::chrono::duration_cast<std::chrono::milliseconds>(t_first_token - t_prefill_done);
      first_token = false;
    }

    char buffer[128];
    const int len = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, false);
    if (len < 0)
    {
      break;
    }

    std::string token_text(buffer, len);
    output += token_text;
    stats.output_tokens++;

    ScopedBatch next_batch(token, static_cast<llama_pos>(tokens.size() + i));
    if (!next_batch.is_valid())
    {
      std::cerr << "Failed to create batch for next token in prompt " << prompt_id << std::endl;
      break;
    }

    const int decode_result = llama_decode(ctx.get(), *next_batch.get());
    if (decode_result != 0)
    {
      std::cerr << "Failed to evaluate next token for prompt " << prompt_id << ", error code: " << decode_result << std::endl;
      break;
    }

    llama_sampler_accept(sampler.get(), token);

    if (token == llama_vocab_eos(vocab))
    {
      break;
    }
  }

  auto t_last_token = std::chrono::high_resolution_clock::now();
  stats.generation_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_last_token - t_first_token);
  stats.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_last_token - t_start);

  // Write prompt and output to file
  output_file << "Prompt " << prompt_id << ":\n";
  output_file << "Input: " << prompt << "\n";
  output_file << "Output: " << output << "\n\n";

  llama_kv_self_clear(ctx.get()); // Reset context for next prompt
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

  // Open input and output files
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
    prompt_stats.print();
    batch_stats.add_prompt_stats(prompt_stats);
  }

  batch_stats.batch_end = std::chrono::high_resolution_clock::now();
  batch_stats.print_summary();

  std::cout << "Batch processing complete. Results written to " << output_file_path << std::endl;
  return 0;
}