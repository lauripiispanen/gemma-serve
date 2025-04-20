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
  const double avg_prefill_time = std::accumulate(prompt_stats.begin(), prompt_stats.end(), 0.0,
                                                  [](double sum, const PromptStats &stats)
                                                  {
                                                    return sum + stats.prefill_time.count();
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
  std::cout << "Avg prefill time       : " << avg_prefill_time << " ms\n";
  std::cout << "Avg prompt total time  : " << avg_total_time << " ms\n";
  std::cout << "Total batch time       : " << batch_time << " ms\n";

  if (total_generation_time.count() > 0)
  {
    const double overall_tps = static_cast<double>(total_output_tokens) / (batch_time / 1000.0);
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

struct ActivePrompt
{
  int id;
  std::string input;
  std::string output;
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler;
  PromptStats stats;
  std::vector<llama_token> tokens;
  int pos = 0;
  int gen_count = 0;
  bool done = false;
  bool first_token = true;
  int seq_id = 0;
  bool seq_id_set = false;
  llama_token last_token = 0;
  ActivePrompt(int pid, std::string text, std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> s)
      : id(pid), input(std::move(text)), sampler(std::move(s)), seq_id(0), seq_id_set(false)
  {
    stats.prompt_id = id;
    stats.t_start = std::chrono::high_resolution_clock::now();
  }
};

BatchStats process_continuous_batch(
    llama_context *ctx,
    llama_model *model,
    PromptSource &source,
    std::ostream &output,
    int batch_size,
    int max_tokens_out)
{
  llama_kv_self_clear(ctx);
  BatchStats batch_stats;
  batch_stats.batch_start = std::chrono::high_resolution_clock::now();

  std::vector<std::unique_ptr<ActivePrompt>> active;
  // Add a sequence counter that increments for each new prompt
  int next_seq_id = 0;
  // Track logit indices for each active prompt
  std::vector<int> logit_indices;

  // Fill initial prompts
  while (active.size() < static_cast<size_t>(batch_size) && source.has_next())
  {
    auto next = source.next_prompt();
    if (!next.valid)
      break;

    int id = next.id;
    std::string text = next.text;
    auto ap = std::make_unique<ActivePrompt>(id, text, create_sampler());
    // Assign a unique sequence ID
    ap->seq_id = next_seq_id++;
    ap->seq_id_set = true;

    std::vector<llama_token> tokens(text.size() + TOKEN_BUFFER_PADDING);
    int n = llama_tokenize(
        llama_model_get_vocab(model), text.c_str(), text.size(),
        tokens.data(), tokens.size(), true, true);
    tokens.resize(n);

    ap->tokens = tokens;
    ap->stats.input_tokens = n;
    ap->stats.t_tokenize_done = std::chrono::high_resolution_clock::now();
    ap->stats.t_prefill_done = ap->stats.t_tokenize_done;

    active.push_back(std::move(ap));
    // Add entry for this prompt in logit indices
    logit_indices.push_back(-1);
  }

  // Prefill
  int total_tokens = 0;
  for (const auto &ap : active)
    total_tokens += ap->tokens.size();
  if (total_tokens == 0)
    return batch_stats;

  llama_batch prefill = llama_batch_init(total_tokens, 0, active.size());
  int idx = 0;
  for (size_t i = 0; i < active.size(); ++i)
  {
    for (int j = 0; j < active[i]->tokens.size(); ++j)
    {
      prefill.token[idx] = active[i]->tokens[j];
      prefill.pos[idx] = j;
      prefill.n_seq_id[idx] = 1;
      prefill.seq_id[idx][0] = active[i]->seq_id;
      // Only compute logits for the last token of each prompt
      prefill.logits[idx] = (j == active[i]->tokens.size() - 1) ? 1 : 0;

      // Store the batch index of the last token for each prompt
      if (j == active[i]->tokens.size() - 1)
      {
        logit_indices[i] = idx;
      }

      idx++;
    }
    active[i]->pos = static_cast<int>(active[i]->tokens.size());
    active[i]->stats.t_prefill_done = std::chrono::high_resolution_clock::now();
  }
  prefill.n_tokens = idx;

  if (llama_decode(ctx, prefill) != 0)
  {
    std::cerr << "Failed to decode prefill batch after clearing KV cache" << std::endl;
    llama_batch_free(prefill);
    return batch_stats;
  }

  llama_batch_free(prefill);

  std::vector<int> free_seq_ids;

  // Generation loop
  while (!active.empty())
  {
    // Add a flag to track which prompts need prefill
    std::vector<bool> needs_prefill(active.size(), false);

    // Process existing prompts first
    for (size_t i = 0; i < active.size(); ++i)
    {
      auto &ap = active[i];
      if (ap->done)
        continue;

      // Make sure we have a valid logit index for this prompt
      if (i >= logit_indices.size() || logit_indices[i] < 0)
      {
        std::cerr << "Warning: Invalid logit index for prompt " << i << std::endl;
        ap->done = true;
        continue;
      }

      llama_token tok = llama_sampler_sample(ap->sampler.get(), ctx, logit_indices[i]);

      if (ap->first_token)
      {
        ap->stats.t_first_token = std::chrono::high_resolution_clock::now();
        ap->first_token = false;
      }

      char buf[128];
      int len = llama_token_to_piece(
          llama_model_get_vocab(model), tok, buf, sizeof(buf), 0, false);
      if (len > 0)
        ap->output += std::string(buf, len);

      llama_sampler_accept(ap->sampler.get(), tok);
      ap->last_token = tok; // Store the token for later use
      ap->gen_count++;
      ap->stats.output_tokens++;

      // Check stop condition
      if (tok == llama_vocab_eos(llama_model_get_vocab(model)) ||
          ap->gen_count >= max_tokens_out)
      {
        ap->done = true;
        ap->stats.t_last_token = std::chrono::high_resolution_clock::now();
        ap->stats.finalize();

        output << "Prompt " << ap->id << ":\n";
        output << "Input: " << ap->input << "\n";
        output << "Output: " << ap->output << "\n\n";

        ap->stats.print_batch();
        batch_stats.add_prompt_stats(ap->stats);
        if (!llama_kv_self_seq_rm(ctx, ap->seq_id, -1, -1))
        {
          std::cerr << "Failed to remove sequence " << ap->seq_id << std::endl;
        }
        free_seq_ids.push_back(ap->seq_id);

        // Attempt to refill
        if (source.has_next())
        {
          auto next = source.next_prompt();
          if (next.valid)
          {
            int id = next.id;
            std::string text = next.text;
            auto new_ap = std::make_unique<ActivePrompt>(id, text, create_sampler());
            // Assign a unique sequence ID for the new prompt
            new_ap->seq_id = free_seq_ids.back();
            free_seq_ids.pop_back();
            new_ap->seq_id_set = true;

            std::vector<llama_token> tokens(text.size() + TOKEN_BUFFER_PADDING);
            int n = llama_tokenize(
                llama_model_get_vocab(model), text.c_str(), text.size(),
                tokens.data(), tokens.size(), true, true);
            tokens.resize(n);

            new_ap->tokens = tokens;
            new_ap->stats.input_tokens = n;
            new_ap->stats.t_tokenize_done = std::chrono::high_resolution_clock::now();

            // Mark this prompt as needing prefill instead of doing decode here
            needs_prefill[i] = true;

            new_ap->pos = n;
            ap = std::move(new_ap);
            continue;
          }
        }

        // No refill possible, remove from batch
        active.erase(active.begin() + i);
        logit_indices.erase(logit_indices.begin() + i);
        needs_prefill.erase(needs_prefill.begin() + i);
        --i;
        continue;
      }
    }

    // Prefill any new prompts
    bool has_prefills = false;
    int prefill_total_tokens = 0;

    // First, count total tokens needed for prefill
    for (size_t i = 0; i < active.size(); ++i)
    {
      if (needs_prefill[i])
      {
        prefill_total_tokens += active[i]->tokens.size();
        has_prefills = true;
      }
    }

    // Process all prefills in one batch
    if (has_prefills)
    {
      llama_batch prefill_batch = llama_batch_init(prefill_total_tokens, 0, active.size());
      int pidx = 0;

      for (size_t i = 0; i < active.size(); ++i)
      {
        if (!needs_prefill[i])
          continue;

        auto &ap = active[i];
        for (int j = 0; j < ap->tokens.size(); ++j)
        {
          prefill_batch.token[pidx] = ap->tokens[j];
          prefill_batch.pos[pidx] = j;
          prefill_batch.n_seq_id[pidx] = 1;
          prefill_batch.seq_id[pidx][0] = ap->seq_id;
          prefill_batch.logits[pidx] = (j == ap->tokens.size() - 1) ? 1 : 0;

          // Store the batch index of the last token
          if (j == ap->tokens.size() - 1)
          {
            logit_indices[i] = pidx;
          }

          pidx++;
        }

        ap->stats.t_prefill_done = std::chrono::high_resolution_clock::now();
      }

      prefill_batch.n_tokens = pidx;

      if (llama_decode(ctx, prefill_batch) != 0)
      {
        std::cerr << "Failed to decode prefill batch for new prompts" << std::endl;
      }

      llama_batch_free(prefill_batch);
    }

    // Now set up the generation batch for all active prompts
    llama_batch batch = llama_batch_init(active.size(), 0, active.size());
    batch.n_tokens = 0;

    for (size_t i = 0; i < active.size(); ++i)
    {
      auto &ap = active[i];

      int bidx = batch.n_tokens++;
      batch.token[bidx] = ap->last_token; // Use the stored last token
      batch.pos[bidx] = ap->pos++;
      batch.n_seq_id[bidx] = 1;
      batch.seq_id[bidx][0] = ap->seq_id;
      batch.logits[bidx] = 1;
      logit_indices[i] = bidx;
    }

    if (batch.n_tokens > 0)
    {
      int ret = llama_decode(ctx, batch);
      if (ret != 0)
      {
        std::cerr << "Failed to decode generation batch" << std::endl;
      }
    }

    llama_batch_free(batch);
  }

  batch_stats.batch_end = std::chrono::high_resolution_clock::now();
  return batch_stats;
}