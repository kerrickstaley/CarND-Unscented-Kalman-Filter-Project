#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  if (!estimations.size()) {
    // no idea if throwing a string is a good practice, oh well
    throw std::string("empty vector passed to CalculateRMSE");
  }

  if (estimations.size() != ground_truth.size()) {
    throw std::string("`estimations` was a different size than `ground_truth`");
  }

  VectorXd sqDiffSum = VectorXd::Zero(estimations[0].size());

  for (int i = 0; i < estimations.size(); i++) {
    if (estimations[i].size() != sqDiffSum.size()) {
      throw std::string("inconsistently sized VectorXd passed in `estimations`");
    }
    if (ground_truth[i].size() != sqDiffSum.size()) {
      throw std::string("inconsistently sized VectorXd passed in `ground_truth`");
    }

    VectorXd diff = estimations[i] - ground_truth[i];
    diff = diff.array() * diff.array();
    sqDiffSum += diff;
  }

  VectorXd rmse = (sqDiffSum / estimations.size()).array().sqrt();
  return rmse;
}
