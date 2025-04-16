#include "common.h"
#include <numeric>

void BatchStats::print_summary() const
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

std::unique_ptr<llama_model, decltype(&llama_model_free)> load_model(const char *model_path)
{
  std::cout << "Loading model from " << model_path << std::endl;
  llama_model_params model_params = llama_model_default_params();

  return std::unique_ptr<llama_model, decltype(&llama_model_free)>(
      llama_model_load_from_file(model_path, model_params),
      llama_model_free);
}

std::unique_ptr<llama_context, decltype(&llama_free)> create_context(llama_model *model)
{
  if (!model)
  {
    std::cerr << "Cannot create context: model is null" << std::endl;
    return std::unique_ptr<llama_context, decltype(&llama_free)>(nullptr, llama_free);
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  return std::unique_ptr<llama_context, decltype(&llama_free)>(
      llama_init_from_model(model, ctx_params),
      llama_free);
}

std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> create_sampler()
{
  auto sparams = llama_sampler_chain_default_params();
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>
      sampler(llama_sampler_chain_init(sparams), llama_sampler_free);

  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(64));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.95, 1));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(1.0));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(1234));

  return sampler;
}

int process_prompt(
    llama_context *ctx,
    const llama_model *model,
    const std::string &prompt,
    std::vector<llama_token> &tokens,
    int &n_tokens)
{
  llama_kv_self_clear(ctx);

  const llama_vocab *vocab = llama_model_get_vocab(model);

  if (tokens.size() < prompt.size() + TOKEN_BUFFER_PADDING)
  {
    tokens.resize(prompt.size() + TOKEN_BUFFER_PADDING);
  }

  n_tokens = llama_tokenize(
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
    std::cerr << "Tokenization failed" << std::endl;
    return 1;
  }

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

  const int prefill_result = llama_decode(ctx, batch);
  llama_batch_free(batch);

  if (prefill_result != 0)
  {
    std::cerr << "Failed to evaluate prompt, error code: " << prefill_result << std::endl;
    return 1;
  }

  return 0;
}

std::string generate_text(
    llama_context *ctx,
    const llama_model *model,
    const std::vector<llama_token> &tokens,
    int n_tokens,
    PromptStats &stats,
    int max_tokens)
{
  const llama_vocab *vocab = llama_model_get_vocab(model);
  std::string output;

  auto sampler = create_sampler();

  bool first_token = true;
  auto t_prefill_done = std::chrono::high_resolution_clock::now();
  stats.t_prefill_done = t_prefill_done;

  for (int i = 0; i < max_tokens; ++i)
  {
    const llama_token token = llama_sampler_sample(sampler.get(), ctx, -1);

    if (first_token)
    {
      stats.t_first_token = std::chrono::high_resolution_clock::now();
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
      std::cerr << "Failed to create batch for next token" << std::endl;
      break;
    }

    const int decode_result = llama_decode(ctx, *next_batch.get());
    if (decode_result != 0)
    {
      std::cerr << "Failed to evaluate next token, error code: " << decode_result << std::endl;
      break;
    }

    llama_sampler_accept(sampler.get(), token);

    if (token == llama_vocab_eos(vocab))
    {
      break;
    }
  }

  stats.t_last_token = std::chrono::high_resolution_clock::now();
  stats.finalize();

  return output;
}