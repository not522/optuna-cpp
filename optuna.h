/*
MIT License

Copyright (c) 2021 Naoto Mizuno

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace optuna {
std::string run_command(const std::string &command) {
  auto fp = popen(command.c_str(), "r");
  std::string ret;
  constexpr int PATH_MAX = 65536;
  char path[PATH_MAX];
  while (fgets(path, PATH_MAX, fp) != NULL) {
    ret += path;
  }
  pclose(fp);
  return ret;
}

enum StudyDirection {
  MINIMIZE,
  MAXIMIZE,
};

class Trial {
public:
  const int number;
  const json params;

  Trial(const json &trial)
      : number(trial["number"].get<int>()), params(trial["params"]) {}

  template <typename T> T param(const std::string &name) const {
    return this->params[name].get<T>();
  }
};

class FrozenTrial : public Trial {
public:
  const std::string state;
  const double value;

  FrozenTrial(const json &trial)
      : Trial(trial), state(trial["state"]),
        value(trial["state"] == "COMPLETE"
                  ? trial["value"].get<double>()
                  : std::numeric_limits<double>::quiet_NaN()) {}

  template <typename T> T param(const std::string &name) const {
    return this->params[name].get<T>();
  }
};

class SearchSpace {
  json search_space;

public:
  void add_float(const std::string &name, const double &low, const double &high,
                 const double &step = 0, const bool &log = false) {
    if (step == 0) {
      if (!log) {
        this->search_space[name]["name"] = "UniformDistribution";
        this->search_space[name]["attributes"]["low"] = low;
        this->search_space[name]["attributes"]["high"] = high;
      } else {
        this->search_space[name]["name"] = "LogUniformDistribution";
        this->search_space[name]["attributes"]["low"] = low;
        this->search_space[name]["attributes"]["high"] = high;
      }
    } else {
      if (!log) {
        this->search_space[name]["name"] = "DiscreteUniformDistribution";
        this->search_space[name]["attributes"]["low"] = low;
        this->search_space[name]["attributes"]["high"] = high;
        this->search_space[name]["attributes"]["q"] = step;
      } else {
        assert(step == 0 || !log);
      }
    }
  }

  void add_int(const std::string &name, const int &low, const int &high,
               const int &step = 1, const int &log = false) {
    if (!log) {
      this->search_space[name]["name"] = "IntUniformDistribution";
      this->search_space[name]["attributes"]["low"] = low;
      this->search_space[name]["attributes"]["high"] = high;
      this->search_space[name]["attributes"]["step"] = step;
    } else {
      this->search_space[name]["name"] = "IntLogUniformDistribution";
      this->search_space[name]["attributes"]["low"] = low;
      this->search_space[name]["attributes"]["high"] = high;
      this->search_space[name]["attributes"]["step"] = step;
    }
  }

  template <typename T>
  void add_categorical(const std::string &name, const std::vector<T> &choices) {
    this->search_space[name]["name"] = "CategoricalDistribution";
    this->search_space[name]["attributes"]["choices"] = choices;
  }

  json to_json() const { return this->search_space; }
};

class Study {
  const std::string storage;
  const std::string study_name;
  const StudyDirection direction;

  std::string base_command(std::string subcommand) const {
    auto command = "optuna " + subcommand;
    command += " --storage " + this->storage;
    command += " --study-name " + this->study_name;
    return command;
  }

public:
  Study(const std::string &storage, const std::string &study_name,
        const StudyDirection &direction = MINIMIZE,
        const bool &skip_if_exists = false)
      : storage(storage), study_name(study_name), direction(direction) {
    auto command = base_command("create-study");
    if (this->direction == MINIMIZE) {
      command += " --direction minimize";
    } else {
      command += " --direction maximize";
    }
    if (skip_if_exists) {
      command += " --skip-if-exists";
    }
    run_command(command);
  }

  Trial ask(const SearchSpace &search_space) const {
    auto command = base_command("ask");
    if (this->direction == MINIMIZE) {
      command += " --direction minimize";
    } else {
      command += " --direction maximize";
    }
    command += " --search-space '" + search_space.to_json().dump() + "'";
    return Trial(json::parse(run_command(command)));
  }

  void tell(const Trial &trial, const double &value) const {
    auto command = base_command("tell");
    command += " --trial-number " + std::to_string(trial.number);
    command += " --values " + std::to_string(value);
    run_command(command);
  }

  FrozenTrial best_trial() const {
    auto command = base_command("best-trial") + " -f json";
    return FrozenTrial(json::parse(run_command(command)));
  }

  std::vector<FrozenTrial> trials() const {
    auto command = base_command("trials") + " -f json";
    json trials_json = json::parse(run_command(command));
    return std::vector<FrozenTrial>(trials_json.begin(), trials_json.end());
  }
};
} // namespace optuna
