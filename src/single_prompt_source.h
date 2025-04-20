#pragma once
#include "prompt_source.h"

class SinglePromptSource : public PromptSource
{
public:
  explicit SinglePromptSource(std::string prompt)
      : prompt_(std::move(prompt)), consumed_(false) {}

  PromptResult next_prompt() override
  {
    if (consumed_)
      return {false, 0, ""};
    consumed_ = true;
    PromptResult result;
    result.valid = true;
    result.id = 1;
    result.text = prompt_;
    return result;
  }

  bool has_next() const override
  {
    return !consumed_;
  }

private:
  std::string prompt_;
  bool consumed_;
};