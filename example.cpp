#include <iostream>

#include "optuna.h"

int main() {
  optuna::Study study("sqlite:///example.db", "test_study", optuna::MINIMIZE, true);

  optuna::SearchSpace search_space;
  search_space.add_categorical<std::string>("c", {"a", "b"});
  search_space.add_float("x", -10, 10);
  search_space.add_int("y", -10, 10);

  for (int i = 0; i < 10; ++i) {
    optuna::Trial trial = study.ask(search_space);
    std::string c = trial.param<std::string>("c");
    double x = trial.param<double>("x");
    int y = trial.param<int>("y");
    if (c == "a") {
      study.tell(trial, x * x + y * y);
    } else {
      study.tell(trial, x * x + y * y + 1);
    }
  }

  for (const optuna::FrozenTrial &trial : study.trials()) {
    std::cout << trial.number << " "
              << trial.state << " "
              << trial.param<std::string>("c") << " "
              << trial.param<double>("x") << " "
              << trial.param<int>("y") << " "
              << trial.value << std::endl;
  }

  const optuna::FrozenTrial best_trial = study.best_trial();
  std::cout << best_trial.number << " " << best_trial.value << std::endl;
}
