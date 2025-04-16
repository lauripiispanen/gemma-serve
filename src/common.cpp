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

BatchProcessingResult process_batch(
    llama_context *ctx,
    const llama_model *model,
    const std::vector<std::string> &prompts,
    int max_tokens_out)
{
  const auto batch_start = std::chrono::high_resolution_clock::now();
  BatchProcessingResult result;
  result.outputs.resize(prompts.size());
  result.stats.resize(prompts.size());

  for (size_t i = 0; i < prompts.size(); i++)
  {
    result.stats[i].prompt_id = i;
    result.stats[i].t_start = batch_start;
  }

  // Step 1: Tokenize all prompts
  std::vector<std::vector<llama_token>> all_tokens(prompts.size());
  std::vector<int> n_tokens(prompts.size());

  for (size_t i = 0; i < prompts.size(); i++)
  {
    all_tokens[i].resize(prompts[i].size() + TOKEN_BUFFER_PADDING);

    n_tokens[i] = llama_tokenize(
        llama_model_get_vocab(model),
        prompts[i].c_str(),
        static_cast<int>(prompts[i].size()),
        all_tokens[i].data(),
        static_cast<int>(all_tokens[i].size()),
        true, // Add BOS/EOS if model needs it
        true  // Parse special tokens
    );

    if (n_tokens[i] < 0)
    {
      std::cerr << "Tokenization failed for prompt " << i << std::endl;
      continue;
    }

    all_tokens[i].resize(n_tokens[i]);
    result.stats[i].input_tokens = n_tokens[i];
    result.stats[i].t_tokenize_done = std::chrono::high_resolution_clock::now();
  }

  // Step 2: Create and process initial batch with all prompts
  llama_kv_self_clear(ctx);

  // Count total tokens across all prompts
  int total_tokens = 0;
  for (int n : n_tokens)
  {
    if (n > 0)
      total_tokens += n;
  }

  if (total_tokens == 0)
  {
    return result;
  }

  // Create batch for all prompts
  llama_batch batch = llama_batch_init(total_tokens, 0, prompts.size());

  // Fill the batch with tokens from all prompts
  int token_idx = 0;
  for (size_t i = 0; i < prompts.size(); i++)
  {
    if (n_tokens[i] <= 0)
      continue;

    for (int j = 0; j < n_tokens[i]; j++)
    {
      batch.token[token_idx] = all_tokens[i][j];
      batch.pos[token_idx] = j; // Set position information
      batch.n_seq_id[token_idx] = 1;
      batch.seq_id[token_idx][0] = i; // Use prompt index as sequence ID

      // Only compute logits for the last token of each prompt
      batch.logits[token_idx] = (j == n_tokens[i] - 1) ? 1 : 0;

      token_idx++;
    }
  }

  batch.n_tokens = token_idx;

  // Process the batch
  if (llama_decode(ctx, batch) != 0)
  {
    std::cerr << "Failed to decode initial batch" << std::endl;
    llama_batch_free(batch);
    return result;
  }

  llama_batch_free(batch);

  // Step 3: Generation phase - create samplers for each sequence
  std::vector<std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>> samplers;
  std::vector<bool> completed(prompts.size(), false);
  std::vector<bool> first_token(prompts.size(), true);
  std::vector<int> token_counts(prompts.size(), 0);

  // Initialize samplers for each sequence
  for (size_t i = 0; i < prompts.size(); i++)
  {
    if (n_tokens[i] <= 0)
    {
      completed[i] = true;
      continue;
    }

    auto sampler = create_sampler();
    samplers.push_back(std::move(sampler));
    result.stats[i].t_prefill_done = std::chrono::high_resolution_clock::now();
  }

  // Generation loop
  for (int gen_idx = 0; gen_idx < max_tokens_out; gen_idx++)
  {
    // Check if all sequences are completed
    bool all_completed = true;
    for (size_t i = 0; i < prompts.size(); i++)
    {
      if (!completed[i])
      {
        all_completed = false;
        break;
      }
    }

    if (all_completed)
      break;

    // Count active sequences first to know how many tokens we need
    int active_sequences = 0;
    for (size_t i = 0; i < prompts.size(); i++)
    {
      if (!completed[i])
      {
        active_sequences++;
      }
    }

    if (active_sequences == 0)
    {
      break;
    }

    // Initialize batch with the exact size we need
    llama_batch next_batch = llama_batch_init(active_sequences, 0, prompts.size());
    if (!next_batch.token || !next_batch.pos || !next_batch.n_seq_id || !next_batch.seq_id || !next_batch.logits)
    {
      std::cerr << "Failed to initialize generation batch" << std::endl;
      break;
    }
    next_batch.n_tokens = 0;

    // First, get the logits for each sequence
    std::vector<llama_token> next_tokens(prompts.size());
    std::vector<bool> token_generated(prompts.size(), false);
    for (size_t i = 0; i < prompts.size(); i++)
    {
      if (completed[i])
        continue;

      // Safety check - ensure we have valid samplers
      if (i >= samplers.size() || !samplers[i])
      {
        std::cerr << "Warning: Invalid sampler for sequence " << i << std::endl;
        completed[i] = true;
        continue;
      }

      try
      {
        const llama_token token = llama_sampler_sample(samplers[i].get(), ctx, -1);
        next_tokens[i] = token;
        token_generated[i] = true;
      }
      catch (...)
      {
        std::cerr << "Error sampling token for sequence " << i << std::endl;
        completed[i] = true;
        continue;
      }

      if (first_token[i])
      {
        result.stats[i].t_first_token = std::chrono::high_resolution_clock::now();
        first_token[i] = false;
      }

      // Convert token to text and append to output
      char buffer[128];
      const int len = llama_token_to_piece(llama_model_get_vocab(model), next_tokens[i], buffer, sizeof(buffer), 0, false);

      if (len >= 0)
      {
        result.outputs[i] += std::string(buffer, len);
      }

      result.stats[i].output_tokens++;
      token_counts[i]++;

      // Check for EOS
      if (next_tokens[i] == llama_vocab_eos(llama_model_get_vocab(model)))
      {
        completed[i] = true;
      }
    }

    // Now build batch with the sampled tokens
    for (size_t i = 0; i < prompts.size(); i++)
    {
      if (completed[i] || !token_generated[i])
        continue;

      const int pos = n_tokens[i] + token_counts[i] - 1;
      const int idx = next_batch.n_tokens;

      // Safety check to avoid buffer overrun
      if (idx >= active_sequences)
      {
        std::cerr << "Warning: Batch index out of bounds" << std::endl;
        continue;
      }

      next_batch.token[idx] = next_tokens[i];
      next_batch.pos[idx] = pos;
      next_batch.n_seq_id[idx] = 1;
      next_batch.seq_id[idx][0] = i;
      next_batch.logits[idx] = 1; // Always compute logits for generated tokens

      next_batch.n_tokens++;

      // Accept the token
      llama_sampler_accept(samplers[i].get(), next_tokens[i]);
    }

    // Process batch
    if (next_batch.n_tokens > 0)
    {
      if (llama_decode(ctx, next_batch) != 0)
      {
        std::cerr << "Failed to decode generation batch" << std::endl;
        llama_batch_free(next_batch);
        break;
      }
    }

    llama_batch_free(next_batch);
  }

  // Finalize stats
  for (size_t i = 0; i < prompts.size(); i++)
  {
    if (n_tokens[i] > 0)
    {
      result.stats[i].t_last_token = std::chrono::high_resolution_clock::now();
      result.stats[i].finalize();
    }
  }

  return result;
}