#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  std_a_ = 0.5;

  std_yawdd_ = 0.1;

  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  weights_ =  VectorXd(2 * n_aug_ + 1);
  weights_.fill(1 / (2.0 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // initialize H_laser_ and R_laser_
  H_laser_ = MatrixXd(2, n_x_);
  H_laser_.setZero();
  H_laser_(0, 0) = 1;
  H_laser_(1, 1) = 1;

  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;

  R_radar_  = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;

    // set initial state
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
      double px = meas_package.raw_measurements_(0),
             py = meas_package.raw_measurements_(1);

      x_ << px, py, 0, 0, 0;
    } else {
      double rho = meas_package.raw_measurements_(0),
             gamma = meas_package.raw_measurements_(1),
             rho_dot = meas_package.raw_measurements_(2);

      x_ << rho * cos(gamma),  // pos_x
            rho * sin(gamma),  // pos_y
            0,                 // v_abs (curious what happens when we use fabs(rho_dot))
            0,                 // yaw_angle (curious what happens when we use gamma + (rho_dot < 0 ? pi : 0))
            0;                 // yaw_rate
    }

    P_ << MatrixXd::Identity(5, 5);

    is_initialized_ = true;

    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1.0e6;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
    if (use_laser_) {
      UpdateLidar(meas_package);
    }
  } else {
    if (use_radar_) {
      UpdateRadar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  if (debug_x_p_) {
    cerr << "enter Prediction()" << endl;
  }

  // calculate augmented state mean vector
  VectorXd x_aug(n_aug_);
  x_aug.setZero();
  x_aug.head(n_x_) = x_;

  // calculate augmented state covariance matrix
  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.setZero();
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // calculate sigma points
  MatrixXd Xsig(n_aug_, 2 * n_aug_ + 1);
  MatrixXd P_aug_sqrt = P_aug.llt().matrixL();
  MatrixXd spread = sqrt(lambda_ + n_aug_) * P_aug_sqrt;

  Xsig.col(0) = x_aug;

  for (int i = 0; i < n_aug_; i++) {
    Xsig.col(1 + i) = x_aug + spread.col(i);
    Xsig.col(1 + n_aug_ + i) = x_aug - spread.col(i);
  }

  // run sigma points through prediction function
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig(0, i);
    double py = Xsig(1, i);
    double v = Xsig(2, i);
    double yaw = Xsig(3, i);
    double yaw_dot = Xsig(4, i);
    double nu_a = Xsig(5, i);
    double nu_yaw = Xsig(6, i);

    // calculate updated px and py
    // check for zero yaw_dot case
    if (fabs(yaw_dot) < 1e-6) {
      Xsig_pred_(0, i) = px + v * cos(yaw) * delta_t;
      Xsig_pred_(1, i) = py + v * sin(yaw) * delta_t;
    } else {
      Xsig_pred_(0, i) = px + v / yaw_dot
                         * (sin(yaw + yaw_dot * delta_t) - sin(yaw));
      Xsig_pred_(1, i) = py + v / yaw_dot
                         * (-cos(yaw + yaw_dot * delta_t) + cos(yaw));
    }
    Xsig_pred_(0, i) += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
    Xsig_pred_(1, i) += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;

    // calculate updated v
    Xsig_pred_(2, i) = v + delta_t * nu_a;

    // calculate updated yaw
    Xsig_pred_(3, i) = tools_.NormalizeAngle(
      yaw + yaw_dot * delta_t + 0.5 * delta_t * delta_t * nu_yaw);

    // calculate updated yaw_dot
    Xsig_pred_(4, i) = yaw_dot + delta_t * nu_yaw;
  }

  // calculate updated state vector
  x_.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // calculate updated state covariance matrix
  P_.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd diff = Xsig_pred_.col(i) - x_;
    diff(3) = tools_.NormalizeAngle(diff(3));
    P_ += weights_(i) * diff * diff.transpose();
  }

  if (debug_x_p_) {
    cerr << "x_ is now: " << endl << x_ << endl;
    cerr << "P_ is now:" << endl << P_ << endl;
    cerr << "exit Prediction()" << endl;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  if (debug_x_p_) {
    cerr << "enter UpdateLidar()" << endl;
  }

  // lidar measurement is linear, so we just use the standard KF equations instead of UKF
  MatrixXd& H = H_laser_;
  MatrixXd& R = R_laser_;

  VectorXd y = meas_package.raw_measurements_ - H * x_;
  MatrixXd S = H * P_ * H.transpose() + R;

  MatrixXd K = P_ * H.transpose() * S.inverse();

  x_ += K * y;
  x_(3) = tools_.NormalizeAngle(x_(3));
  P_ -= K * H * P_;

  if (debug_x_p_) {
    cerr << "x_ is now:" << endl << x_ << endl;
    cerr << "P_ is now:" << endl << P_ << endl;
    cerr << "exit UpdateLidar()" << endl;
  }
  /**
  TODO:

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  if (debug_x_p_) {
    cerr << "enter UpdateRadar()" << endl;
  }

  // calculate sigma points
  MatrixXd Zsig(3, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    // rho
    Zsig(0, i) = sqrt(px * px + py * py);
    // gamma
    Zsig(1, i) = atan2(py, px);
    // rho_dot
    Zsig(2, i) = (px * cos(yaw) * v + py * sin(yaw) * v) / Zsig(0, i);
  }

  // calculate z_pred
  VectorXd z_pred(3);
  z_pred.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate S
  MatrixXd S(3, 3);
  S.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    diff(1) = tools_.NormalizeAngle(diff(1));
    S += weights_(i) * diff * diff.transpose();
  }
  S += R_radar_;

  // compute cross correlation matrix T
  MatrixXd T(n_x_, 3);
  T.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    T += weights_(i) * (Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  }

  // compute Kalman gain
  MatrixXd K = T * S.inverse();

  // calculate innovation
  VectorXd y = meas_package.raw_measurements_ - z_pred;

  // update state vector
  x_ += K * y;

  // update state covariance matrix
  P_ -= K * S * K.transpose();

  if (debug_x_p_) {
    cerr << "x_ is now:" << endl << x_ << endl;
    cerr << "P_ is now:" << endl << P_ << endl;
    cerr << "exit UpdateRadar()" << endl;
  }
  /**
  TODO:

  You'll also need to calculate the radar NIS.
  */
}
