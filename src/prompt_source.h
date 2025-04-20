#pragma once
#include <string>

// Forward declare the class
class PromptSource
{
public:
  virtual ~PromptSource() = default;

  // Simple alternative return type
  struct PromptResult
  {
    bool valid;
    int id;
    std::string text;
  };

  virtual PromptResult next_prompt() = 0;
  virtual bool has_next() const = 0;
};