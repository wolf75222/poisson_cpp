#pragma once

#include <fftw3.h>

namespace poisson::spectral {

/// RAII wrapper around `fftw_plan`. Non-copyable, movable.
class FFTWPlan {
 public:
  FFTWPlan() noexcept = default;

  explicit FFTWPlan(::fftw_plan plan) noexcept : plan_(plan) {}

  FFTWPlan(const FFTWPlan&) = delete;
  FFTWPlan& operator=(const FFTWPlan&) = delete;

  FFTWPlan(FFTWPlan&& other) noexcept : plan_(other.plan_) {
    other.plan_ = nullptr;
  }
  FFTWPlan& operator=(FFTWPlan&& other) noexcept {
    if (this != &other) {
      destroy();
      plan_ = other.plan_;
      other.plan_ = nullptr;
    }
    return *this;
  }

  ~FFTWPlan() { destroy(); }

  ::fftw_plan get() const noexcept { return plan_; }
  explicit operator bool() const noexcept { return plan_ != nullptr; }

  void execute() const noexcept { ::fftw_execute(plan_); }

 private:
  void destroy() noexcept {
    if (plan_) {
      ::fftw_destroy_plan(plan_);
      plan_ = nullptr;
    }
  }

  ::fftw_plan plan_ = nullptr;
};

}  // namespace poisson::spectral
