#include <iostream>

#include "optuna.h"

int main() {
  optuna::Study study("sqlite:///example.db", "test_study", optuna::MINIMIZE, true);

  optuna::SearchSpace search_space;
  search_space.add_categorical<std::string>("c", {"a", "b"});
  search_space.add_float("x", -10, 10);
  search_space.add_float("y", -10, 10);

  for (int i = 0; i < 10; ++i) {
    auto trial = study.ask(search_space);
    auto c = trial.param<std::string>("c");
    auto x = trial.param<double>("x");
    auto y = trial.param<double>("y");
    if (c == "a") {
      study.tell(trial, x * x + y * y);
    } else {
      study.tell(trial, x * x + y * y + 1);
    }
  }

  for (const auto &trial : study.trials()) {
    std::cout << trial.number << " "
              << trial.state << " "
              << trial.param<std::string>("c") << " "
              << trial.param<double>("x") << " "
              << trial.param<double>("y") << " "
              << trial.value << std::endl;
  }
}
