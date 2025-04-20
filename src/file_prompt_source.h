#pragma once
#include "prompt_source.h"
#include <fstream>

class FilePromptSource : public PromptSource
{
public:
  explicit FilePromptSource(const std::string &path) : file_(path), next_id_(1) {}

  PromptResult next_prompt() override
  {
    std::string line;
    while (std::getline(file_, line))
    {
      if (!line.empty())
      {
        PromptResult result;
        result.valid = true;
        result.id = next_id_++;
        result.text = line;
        return result;
      }
    }
    return {false, 0, ""};
  }

  bool has_next() const override
  {
    return file_.good();
  }

private:
  std::ifstream file_;
  int next_id_;
};