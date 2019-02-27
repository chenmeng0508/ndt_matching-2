#include "ndt_matching.h"

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>



#include "common.h"

NDTMatching::NDTMatching()
{
  pthread_mutex_init(&mutex_, NULL);
  // Publishers
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  predict_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/predict_pose", 10);
  predict_pose_imu_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/predict_pose_imu", 10);
  predict_pose_odom_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/predict_pose_odom", 10);
  predict_pose_imu_odom_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/predict_pose_imu_odom", 10);
  ndt_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/ndt_pose", 10);
  // current_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 10);
  localizer_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/localizer_pose", 10);

  // Subscribers
  ros::Subscriber points_sub = nh.subscribe("cloud", 1, &NDTMatching::points_callback, this);
  ros::Subscriber map_sub = nh.subscribe("points_map", 1, &NDTMatching::map_callback, this);
  //ros::Subscriber odom_sub = nh.subscribe("/vehicle/odom", _queue_size * 10, odom_callback);
  //ros::Subscriber imu_sub = nh.subscribe(_imu_topic.c_str(), _queue_size * 10, imu_callback);

  //pthread_t thread;
  //pthread_create(&thread, NULL, thread_func, NULL);

  tf_btol_ = Eigen::Matrix4f::Identity();
  ros::spin();
}

void NDTMatching::map_callback(const sensor_msgs::PointCloud2::ConstPtr &input)
{
  std::cout << "Update points_map." << std::endl;

  // Convert the data type(from sensor_msgs to pcl).
  pcl::PointCloud<pcl::PointXYZ> map;

  pcl::fromROSMsg(*input, map);

  pcl::PointCloud<pcl::PointXYZ>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZ>(map));
  map_ptr_ = map_ptr;
  // Setting point cloud to be aligned to.
  if (method_type_ == MethodType::PCL_GENERIC)
  {
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> new_ndt;
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    new_ndt.setResolution(ndt_res_);
    new_ndt.setInputTarget(map_ptr_);
    new_ndt.setMaximumIterations(max_iter_);
    new_ndt.setStepSize(step_size_);
    new_ndt.setTransformationEpsilon(trans_eps_);

    new_ndt.align(*output_cloud, Eigen::Matrix4f::Identity());
    pthread_mutex_lock(&mutex_);
    ndt_ = new_ndt;
    pthread_mutex_unlock(&mutex_);
  }
  else if (method_type_ == MethodType::PCL_ANH)
  {
    cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> new_anh_ndt;
    new_anh_ndt.setResolution(ndt_res_);
    new_anh_ndt.setInputTarget(map_ptr_);
    new_anh_ndt.setMaximumIterations(max_iter_);
    new_anh_ndt.setStepSize(step_size_);
    new_anh_ndt.setTransformationEpsilon(trans_eps_);

    pcl::PointCloud<pcl::PointXYZ>::Ptr dummy_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointXYZ dummy_point;
    dummy_scan_ptr->push_back(dummy_point);
    new_anh_ndt.setInputSource(dummy_scan_ptr);

    new_anh_ndt.align(Eigen::Matrix4f::Identity());

    pthread_mutex_lock(&mutex_);
    anh_ndt_ = new_anh_ndt;
    pthread_mutex_unlock(&mutex_);
  }
#ifdef CUDA_FOUND
  else if (method_type_ == MethodType::PCL_ANH_GPU)
  {
    std::shared_ptr<gpu::GNormalDistributionsTransform> new_anh_gpu_ndt_ptr =
        std::make_shared<gpu::GNormalDistributionsTransform>();
    new_anh_gpu_ndt_ptr->setResolution(ndt_res_);
    new_anh_gpu_ndt_ptr->setInputTarget(map_ptr_);
    new_anh_gpu_ndt_ptr->setMaximumIterations(max_iter_);
    new_anh_gpu_ndt_ptr->setStepSize(step_size_);
    new_anh_gpu_ndt_ptr->setTransformationEpsilon(trans_eps_);

    pcl::PointCloud<pcl::PointXYZ>::Ptr dummy_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointXYZ dummy_point;
    dummy_scan_ptr->push_back(dummy_point);
    new_anh_gpu_ndt_ptr->setInputSource(dummy_scan_ptr);

    new_anh_gpu_ndt_ptr->align(Eigen::Matrix4f::Identity());

    pthread_mutex_lock(&mutex_);
    anh_gpu_ndt_ptr_ = new_anh_gpu_ndt_ptr;
    pthread_mutex_unlock(&mutex_);
  }
#endif
#ifdef USE_PCL_OPENMP
  else if (method_type_ == MethodType::PCL_OPENMP)
  {
    pcl_omp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> new_omp_ndt;
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    new_omp_ndt.setResolution(ndt_res_);
    new_omp_ndt.setInputTarget(map_ptr_);
    new_omp_ndt.setMaximumIterations(max_iter_);
    new_omp_ndt.setStepSize(step_size_);
    new_omp_ndt.setTransformationEpsilon(trans_eps_);

    new_omp_ndt.align(*output_cloud, Eigen::Matrix4f::Identity());

    pthread_mutex_lock(&mutex_);
    omp_ndt_ = new_omp_ndt;
    pthread_mutex_unlock(&mutex_);
  }
#endif
  map_loaded_ = 1;
}

void NDTMatching::points_callback(const sensor_msgs::PointCloud2::ConstPtr &input)
{

  int iteration = 0;
  double fitness_score = 0.0;
  double trans_probability = 0.0;

  std::cout << "get points" << std::endl;
  if (!map_loaded_)
    return;

  std::chrono::time_point<std::chrono::system_clock> matching_start, matching_end;

  matching_start = std::chrono::system_clock::now();

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion predict_q, ndt_q, current_q, localizer_q;

  pcl::PointXYZ p;
  pcl::PointCloud<pcl::PointXYZ> filtered_scan;

  ros::Time current_scan_time = input->header.stamp;
  static ros::Time previous_scan_time = current_scan_time;

  pcl::fromROSMsg(*input, filtered_scan);
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>(filtered_scan));
  int scan_points_num = filtered_scan_ptr->size();

  Eigen::Matrix4f t(Eigen::Matrix4f::Identity());  // base_link
  Eigen::Matrix4f t2(Eigen::Matrix4f::Identity()); // localizer

  std::chrono::time_point<std::chrono::system_clock> align_start, align_end, getFitnessScore_start, getFitnessScore_end;
  static double align_time, getFitnessScore_time = 0.0;

  pthread_mutex_lock(&mutex_);

  if (method_type_ == MethodType::PCL_GENERIC)
    ndt_.setInputSource(filtered_scan_ptr);
  else if (method_type_ == MethodType::PCL_ANH)
    anh_ndt_.setInputSource(filtered_scan_ptr);
#ifdef CUDA_FOUND
  else if (method_type_ == MethodType::PCL_ANH_GPU)
    anh_gpu_ndt_ptr_->setInputSource(filtered_scan_ptr);
#endif
#ifdef USE_PCL_OPENMP
  else if (method_type_ == MethodType::PCL_OPENMP)
    omp_ndt_.setInputSource(filtered_scan_ptr);
#endif

  // Guess the initial gross estimation of the transformation
  double diff_time = (current_scan_time - previous_scan_time).toSec();

  double offset_x, offset_y, offset_z, offset_yaw; // current_pos - previous_pose_

  if (offset_ == "linear")
  {
    offset_x = current_velocity_x_ * diff_time;
    offset_y = current_velocity_y_ * diff_time;
    offset_z = current_velocity_z_ * diff_time;
    offset_yaw = angular_velocity_ * diff_time;
  }
  else if (offset_ == "quadratic")
  {
    offset_x = (current_velocity_x_ + current_accel_x_ * diff_time) * diff_time;
    offset_y = (current_velocity_y_ + current_accel_y_ * diff_time) * diff_time;
    offset_z = current_velocity_z_ * diff_time;
    offset_yaw = angular_velocity_ * diff_time;
  }
  else
  {
    offset_x = 0.0;
    offset_y = 0.0;
    offset_z = 0.0;
    offset_yaw = 0.0;
  }

  if (diff_time == 0)
  {
    previous_pose_.x = 0;
    previous_pose_.y = 0;
    previous_pose_.z = 0;
    previous_pose_.roll = 0;
    previous_pose_.pitch = 0;
    previous_pose_.yaw = 0;
  }

  predict_pose.x = previous_pose_.x + offset_x;
  predict_pose.y = previous_pose_.y + offset_y;
  predict_pose.z = previous_pose_.z + offset_z;
  predict_pose.roll = previous_pose_.roll;
  predict_pose.pitch = previous_pose_.pitch;
  predict_pose.yaw = previous_pose_.yaw + offset_yaw;

  if (use_imu_ == true && use_odom_ == true)
    imu_odom_calc(current_scan_time);
  if (use_imu_ == true && use_odom_ == false)
    imu_calc(current_scan_time);
  if (use_imu_ == false && use_odom_ == true)
    odom_calc(current_scan_time);

  pose predict_pose_for_ndt;
  if (use_imu_ == true && use_odom_ == true)
    predict_pose_for_ndt = predict_pose_imu_odom_;
  else if (use_imu_ == true && use_odom_ == false)
    predict_pose_for_ndt = predict_pose_imu_;
  else if (use_imu_ == false && use_odom_ == true)
    predict_pose_for_ndt = predict_pose_odom_;
  else
    predict_pose_for_ndt = predict_pose;

  Eigen::Translation3f init_translation(predict_pose_for_ndt.x, predict_pose_for_ndt.y, predict_pose_for_ndt.z);
  Eigen::AngleAxisf init_rotation_x(predict_pose_for_ndt.roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y(predict_pose_for_ndt.pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z(predict_pose_for_ndt.yaw, Eigen::Vector3f::UnitZ());
  Eigen::Matrix4f init_guess = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x) * tf_btol_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  bool has_converged;
  if (method_type_ == MethodType::PCL_GENERIC)
  {
    align_start = std::chrono::system_clock::now();
    ndt_.align(*output_cloud, init_guess);
    align_end = std::chrono::system_clock::now();

    has_converged = ndt_.hasConverged();

    t = ndt_.getFinalTransformation();
    iteration = ndt_.getFinalNumIteration();

    getFitnessScore_start = std::chrono::system_clock::now();
    //fitness_score = ndt_.getFitnessScore();
    getFitnessScore_end = std::chrono::system_clock::now();

    trans_probability = ndt_.getTransformationProbability();
  }
  else if (method_type_ == MethodType::PCL_ANH)
  {
    align_start = std::chrono::system_clock::now();
    anh_ndt_.align(init_guess);
    align_end = std::chrono::system_clock::now();

    has_converged = anh_ndt_.hasConverged();

    t = anh_ndt_.getFinalTransformation();
    iteration = anh_ndt_.getFinalNumIteration();

    getFitnessScore_start = std::chrono::system_clock::now();
    fitness_score = anh_ndt_.getFitnessScore();
    getFitnessScore_end = std::chrono::system_clock::now();

    trans_probability = anh_ndt_.getTransformationProbability();
  }
#ifdef CUDA_FOUND
  else if (method_type_ == MethodType::PCL_ANH_GPU)
  {
    align_start = std::chrono::system_clock::now();
    anh_gpu_ndt_ptr->align(init_guess);
    align_end = std::chrono::system_clock::now();

    has_converged = anh_gpu_ndt_ptr->hasConverged();

    t = anh_gpu_ndt_ptr->getFinalTransformation();
    iteration = anh_gpu_ndt_ptr->getFinalNumIteration();

    getFitnessScore_start = std::chrono::system_clock::now();
    fitness_score = anh_gpu_ndt_ptr->getFitnessScore();
    getFitnessScore_end = std::chrono::system_clock::now();

    trans_probability = anh_gpu_ndt_ptr->getTransformationProbability();
  }
#endif
#ifdef USE_PCL_OPENMP
  else if (method_type_ == MethodType::PCL_OPENMP)
  {
    align_start = std::chrono::system_clock::now();
    omp_ndt.align(*output_cloud, init_guess);
    align_end = std::chrono::system_clock::now();

    has_converged = omp_ndt.hasConverged();

    t = omp_ndt.getFinalTransformation();
    iteration = omp_ndt.getFinalNumIteration();

    getFitnessScore_start = std::chrono::system_clock::now();
    fitness_score = omp_ndt.getFitnessScore();
    getFitnessScore_end = std::chrono::system_clock::now();

    trans_probability = omp_ndt.getTransformationProbability();
  }
#endif
  align_time = std::chrono::duration_cast<std::chrono::microseconds>(align_end - align_start).count() / 1000.0;

  t2 = t * tf_btol_.inverse();
  getFitnessScore_time =
      std::chrono::duration_cast<std::chrono::microseconds>(getFitnessScore_end - getFitnessScore_start).count() /
      1000.0;

  pthread_mutex_unlock(&mutex_);

  tf::Matrix3x3 mat_l; // localizer
  mat_l.setValue(static_cast<double>(t(0, 0)), static_cast<double>(t(0, 1)), static_cast<double>(t(0, 2)),
                 static_cast<double>(t(1, 0)), static_cast<double>(t(1, 1)), static_cast<double>(t(1, 2)),
                 static_cast<double>(t(2, 0)), static_cast<double>(t(2, 1)), static_cast<double>(t(2, 2)));
  pose localizer_pose;
  // Update localizer_pose
  localizer_pose.x = t(0, 3);
  localizer_pose.y = t(1, 3);
  localizer_pose.z = t(2, 3);
  mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw, 1);

  tf::Matrix3x3 mat_b; // base_link
  mat_b.setValue(static_cast<double>(t2(0, 0)), static_cast<double>(t2(0, 1)), static_cast<double>(t2(0, 2)),
                 static_cast<double>(t2(1, 0)), static_cast<double>(t2(1, 1)), static_cast<double>(t2(1, 2)),
                 static_cast<double>(t2(2, 0)), static_cast<double>(t2(2, 1)), static_cast<double>(t2(2, 2)));

  pose ndt_pose;
  // Update ndt_pose
  ndt_pose.x = t2(0, 3);
  ndt_pose.y = t2(1, 3);
  ndt_pose.z = t2(2, 3);
  mat_b.getRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw, 1);

  // Calculate the difference between ndt_pose and predict_pose
  double predict_pose_error = sqrt((ndt_pose.x - predict_pose_for_ndt.x) * (ndt_pose.x - predict_pose_for_ndt.x) +
                                   (ndt_pose.y - predict_pose_for_ndt.y) * (ndt_pose.y - predict_pose_for_ndt.y) +
                                   (ndt_pose.z - predict_pose_for_ndt.z) * (ndt_pose.z - predict_pose_for_ndt.z));

  int use_predict_pose = 0;

  if (predict_pose_error <= PREDICT_POSE_THRESHOLD)
  {
    use_predict_pose = 0;
  }
  else
  {
    use_predict_pose = 1;
  }
  use_predict_pose = 0;

  if (use_predict_pose == 0)
  {
    current_pose_.x = ndt_pose.x;
    current_pose_.y = ndt_pose.y;
    current_pose_.z = ndt_pose.z;
    current_pose_.roll = ndt_pose.roll;
    current_pose_.pitch = ndt_pose.pitch;
    current_pose_.yaw = ndt_pose.yaw;
  }
  else
  {
    current_pose_.x = predict_pose_for_ndt.x;
    current_pose_.y = predict_pose_for_ndt.y;
    current_pose_.z = predict_pose_for_ndt.z;
    current_pose_.roll = predict_pose_for_ndt.roll;
    current_pose_.pitch = predict_pose_for_ndt.pitch;
    current_pose_.yaw = predict_pose_for_ndt.yaw;
  }

  // Compute the velocity and acceleration
  double diff = 0.0;
  double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_yaw;

  diff_x = current_pose_.x - previous_pose_.x;
  diff_y = current_pose_.y - previous_pose_.y;
  diff_z = current_pose_.z - previous_pose_.z;
  diff_yaw = calcDiffForRadian(current_pose_.yaw, previous_pose_.yaw);
  diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

  current_velocity_ = (diff_time > 0) ? (diff / diff_time) : 0;
  current_velocity_x_ = (diff_time > 0) ? (diff_x / diff_time) : 0;
  current_velocity_y_ = (diff_time > 0) ? (diff_y / diff_time) : 0;
  current_velocity_z_ = (diff_time > 0) ? (diff_z / diff_time) : 0;
  angular_velocity_ = (diff_time > 0) ? (diff_yaw / diff_time) : 0;

  current_pose_imu_.x = current_pose_.x;
  current_pose_imu_.y = current_pose_.y;
  current_pose_imu_.z = current_pose_.z;
  current_pose_imu_.roll = current_pose_.roll;
  current_pose_imu_.pitch = current_pose_.pitch;
  current_pose_imu_.yaw = current_pose_.yaw;

  current_velocity_imu_x_ = current_velocity_x_;
  current_velocity_imu_y_ = current_velocity_y_;
  current_velocity_imu_z_ = current_velocity_z_;

  current_pose_odom_.x = current_pose_.x;
  current_pose_odom_.y = current_pose_.y;
  current_pose_odom_.z = current_pose_.z;
  current_pose_odom_.roll = current_pose_.roll;
  current_pose_odom_.pitch = current_pose_.pitch;
  current_pose_odom_.yaw = current_pose_.yaw;

  current_pose_imu_odom_.x = current_pose_.x;
  current_pose_imu_odom_.y = current_pose_.y;
  current_pose_imu_odom_.z = current_pose_.z;
  current_pose_imu_odom_.roll = current_pose_.roll;
  current_pose_imu_odom_.pitch = current_pose_.pitch;
  current_pose_imu_odom_.yaw = current_pose_.yaw;

  double current_velocity_smooth = (current_velocity_ + previous_velocity_ + previous_previous_velocity_) / 3.0;
  if (current_velocity_smooth < 0.2)
  {
    current_velocity_smooth = 0.0;
  }
  current_accel_ = (diff_time > 0) ? ((current_velocity_ - previous_velocity_) / diff_time) : 0;
  current_accel_x_ = (diff_time > 0) ? ((current_velocity_x_ - previous_velocity_x_) / diff_time) : 0;
  current_accel_y_ = (diff_time > 0) ? ((current_velocity_y_ - previous_velocity_y_) / diff_time) : 0;
  current_accel_z_ = (diff_time > 0) ? ((current_velocity_z_ - previous_velocity_z_) / diff_time) : 0;

  // Set values for publishing pose
  predict_q.setRPY(predict_pose.roll, predict_pose.pitch, predict_pose.yaw);
  predict_pose_msg_.header.frame_id = "/map";
  predict_pose_msg_.header.stamp = current_scan_time;
  predict_pose_msg_.pose.position.x = predict_pose.x;
  predict_pose_msg_.pose.position.y = predict_pose.y;
  predict_pose_msg_.pose.position.z = predict_pose.z;
  predict_pose_msg_.pose.orientation.x = predict_q.x();
  predict_pose_msg_.pose.orientation.y = predict_q.y();
  predict_pose_msg_.pose.orientation.z = predict_q.z();
  predict_pose_msg_.pose.orientation.w = predict_q.w();

  tf::Quaternion predict_q_imu;
  predict_q_imu.setRPY(predict_pose_imu_.roll, predict_pose_imu_.pitch, predict_pose_imu_.yaw);
  predict_pose_imu_msg_.header.frame_id = "map";
  predict_pose_imu_msg_.header.stamp = input->header.stamp;
  predict_pose_imu_msg_.pose.position.x = predict_pose_imu_.x;
  predict_pose_imu_msg_.pose.position.y = predict_pose_imu_.y;
  predict_pose_imu_msg_.pose.position.z = predict_pose_imu_.z;
  predict_pose_imu_msg_.pose.orientation.x = predict_q_imu.x();
  predict_pose_imu_msg_.pose.orientation.y = predict_q_imu.y();
  predict_pose_imu_msg_.pose.orientation.z = predict_q_imu.z();
  predict_pose_imu_msg_.pose.orientation.w = predict_q_imu.w();
  predict_pose_imu_pub_.publish(predict_pose_imu_msg_);

  tf::Quaternion predict_q_odom;
  predict_q_odom.setRPY(predict_pose_odom_.roll, predict_pose_odom_.pitch, predict_pose_odom_.yaw);
  predict_pose_odom_msg_.header.frame_id = "map";
  predict_pose_odom_msg_.header.stamp = input->header.stamp;
  predict_pose_odom_msg_.pose.position.x = predict_pose_odom_.x;
  predict_pose_odom_msg_.pose.position.y = predict_pose_odom_.y;
  predict_pose_odom_msg_.pose.position.z = predict_pose_odom_.z;
  predict_pose_odom_msg_.pose.orientation.x = predict_q_odom.x();
  predict_pose_odom_msg_.pose.orientation.y = predict_q_odom.y();
  predict_pose_odom_msg_.pose.orientation.z = predict_q_odom.z();
  predict_pose_odom_msg_.pose.orientation.w = predict_q_odom.w();
  predict_pose_odom_pub_.publish(predict_pose_odom_msg_);

  tf::Quaternion predict_q_imu_odom;
  predict_q_imu_odom.setRPY(predict_pose_imu_odom_.roll, predict_pose_imu_odom_.pitch, predict_pose_imu_odom_.yaw);
  predict_pose_imu_odom_msg_.header.frame_id = "map";
  predict_pose_imu_odom_msg_.header.stamp = input->header.stamp;
  predict_pose_imu_odom_msg_.pose.position.x = predict_pose_imu_odom_.x;
  predict_pose_imu_odom_msg_.pose.position.y = predict_pose_imu_odom_.y;
  predict_pose_imu_odom_msg_.pose.position.z = predict_pose_imu_odom_.z;
  predict_pose_imu_odom_msg_.pose.orientation.x = predict_q_imu_odom.x();
  predict_pose_imu_odom_msg_.pose.orientation.y = predict_q_imu_odom.y();
  predict_pose_imu_odom_msg_.pose.orientation.z = predict_q_imu_odom.z();
  predict_pose_imu_odom_msg_.pose.orientation.w = predict_q_imu_odom.w();
  predict_pose_imu_odom_pub_.publish(predict_pose_imu_odom_msg_);

  ndt_q.setRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw);

  ndt_pose_msg_.header.frame_id = "/map";
  ndt_pose_msg_.header.stamp = current_scan_time;
  ndt_pose_msg_.pose.position.x = ndt_pose.x;
  ndt_pose_msg_.pose.position.y = ndt_pose.y;
  ndt_pose_msg_.pose.position.z = ndt_pose.z;
  ndt_pose_msg_.pose.orientation.x = ndt_q.x();
  ndt_pose_msg_.pose.orientation.y = ndt_q.y();
  ndt_pose_msg_.pose.orientation.z = ndt_q.z();
  ndt_pose_msg_.pose.orientation.w = ndt_q.w();

  current_q.setRPY(current_pose_.roll, current_pose_.pitch, current_pose_.yaw);
  // current_pose_ is published by vel_pose_mux
  /*
    current_pose_msg.header.frame_id = "/map";
    current_pose_msg.header.stamp = current_scan_time;
    current_pose_msg.pose.position.x = current_pose_.x;
    current_pose_msg.pose.position.y = current_pose_.y;
    current_pose_msg.pose.position.z = current_pose_.z;
    current_pose_msg.pose.orientation.x = current_q.x();
    current_pose_msg.pose.orientation.y = current_q.y();
    current_pose_msg.pose.orientation.z = current_q.z();
    current_pose_msg.pose.orientation.w = current_q.w();
    */

  localizer_q.setRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw);

  localizer_pose_msg_.header.frame_id = "/map";
  localizer_pose_msg_.header.stamp = current_scan_time;
  localizer_pose_msg_.pose.position.x = localizer_pose.x;
  localizer_pose_msg_.pose.position.y = localizer_pose.y;
  localizer_pose_msg_.pose.position.z = localizer_pose.z;
  localizer_pose_msg_.pose.orientation.x = localizer_q.x();
  localizer_pose_msg_.pose.orientation.y = localizer_q.y();
  localizer_pose_msg_.pose.orientation.z = localizer_q.z();
  localizer_pose_msg_.pose.orientation.w = localizer_q.w();

  predict_pose_pub_.publish(predict_pose_msg_);
  ndt_pose_pub_.publish(ndt_pose_msg_);
  // current_pose_ is published by vel_pose_mux
  //    current_pose_pub.publish(current_pose_msg);
  localizer_pose_pub_.publish(localizer_pose_msg_);

  // Send TF "/base_link" to "/map"
  transform.setOrigin(tf::Vector3(current_pose_.x, current_pose_.y, current_pose_.z));
  transform.setRotation(current_q);
  //    br.sendTransform(tf::StampedTransform(transform, current_scan_time, "/map", "/base_link"));
  br.sendTransform(tf::StampedTransform(transform, current_scan_time, "/map", "/base_link"));

  matching_end = std::chrono::system_clock::now();
  double exe_time = std::chrono::duration_cast<std::chrono::microseconds>(matching_end - matching_start).count() / 1000.0;

  /* Compute NDT_Reliability */
  double ndt_reliability = Wa * (exe_time / 100.0) * 100.0 + Wb * (iteration / 10.0) * 100.0 +
                           Wc * ((2.0 - trans_probability) / 2.0) * 100.0;

  // Write log
  if (output_log_data_)
  {
    if (!ofs_)
    {
      std::cerr << "Could not open " << filename_ << "." << std::endl;
    }
    else
    {
      ofs_ << input->header.seq << "," << scan_points_num << "," << step_size_ << "," << trans_eps_ << "," << std::fixed
           << std::setprecision(5) << current_pose_.x << "," << std::fixed << std::setprecision(5) << current_pose_.y << ","
           << std::fixed << std::setprecision(5) << current_pose_.z << "," << current_pose_.roll << "," << current_pose_.pitch
           << "," << current_pose_.yaw << "," << predict_pose.x << "," << predict_pose.y << "," << predict_pose.z << ","
           << predict_pose.roll << "," << predict_pose.pitch << "," << predict_pose.yaw << ","
           << current_pose_.x - predict_pose.x << "," << current_pose_.y - predict_pose.y << ","
           << current_pose_.z - predict_pose.z << "," << current_pose_.roll - predict_pose.roll << ","
           << current_pose_.pitch - predict_pose.pitch << "," << current_pose_.yaw - predict_pose.yaw << ","
           << predict_pose_error << "," << iteration << "," << fitness_score << "," << trans_probability << ","
           << ndt_reliability << "," << current_velocity_ << "," << current_velocity_smooth << "," << current_accel_
           << "," << angular_velocity_ << "," << exe_time << "," << align_time << "," << getFitnessScore_time
           << std::endl;
    }
  }

  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "Sequence: " << input->header.seq << std::endl;
  std::cout << "Timestamp: " << input->header.stamp << std::endl;
  std::cout << "Frame ID: " << input->header.frame_id << std::endl;
  //		std::cout << "Number of Scan Points: " << scan_ptr->size() << " points." << std::endl;
  std::cout << "Number of Filtered Scan Points: " << scan_points_num << " points." << std::endl;
  std::cout << "NDT has converged: " << has_converged << std::endl;
  std::cout << "Fitness Score: " << fitness_score << std::endl;
  std::cout << "Transformation Probability: " << trans_probability << std::endl;
  std::cout << "Execution Time: " << exe_time << " ms." << std::endl;
  std::cout << "Number of Iterations: " << iteration << std::endl;
  std::cout << "NDT Reliability: " << ndt_reliability << std::endl;
  std::cout << "(x,y,z,roll,pitch,yaw): " << std::endl;
  std::cout << "(" << current_pose_.x << ", " << current_pose_.y << ", " << current_pose_.z << ", " << current_pose_.roll
            << ", " << current_pose_.pitch << ", " << current_pose_.yaw << ")" << std::endl;
  std::cout << "Transformation Matrix: " << std::endl;
  std::cout << t << std::endl;
  std::cout << "Align time: " << align_time << std::endl;
  std::cout << "Get fitness score time: " << getFitnessScore_time << std::endl;
  std::cout << "-----------------------------------------------------------------" << std::endl;

  offset_imu_x_ = 0.0;
  offset_imu_y_ = 0.0;
  offset_imu_z_ = 0.0;
  offset_imu_roll_ = 0.0;
  offset_imu_pitch_ = 0.0;
  offset_imu_yaw_ = 0.0;

  offset_odom_x_ = 0.0;
  offset_odom_y_ = 0.0;
  offset_odom_z_ = 0.0;
  offset_odom_roll_ = 0.0;
  offset_odom_pitch_ = 0.0;
  offset_odom_yaw_ = 0.0;

  offset_imu_odom_x_ = 0.0;
  offset_imu_odom_y_ = 0.0;
  offset_imu_odom_z_ = 0.0;
  offset_imu_odom_roll_ = 0.0;
  offset_imu_odom_pitch_ = 0.0;
  offset_imu_odom_yaw_ = 0.0;

  // Update previous_***
  previous_pose_.x = current_pose_.x;
  previous_pose_.y = current_pose_.y;
  previous_pose_.z = current_pose_.z;
  previous_pose_.roll = current_pose_.roll;
  previous_pose_.pitch = current_pose_.pitch;
  previous_pose_.yaw = current_pose_.yaw;

  previous_scan_time = current_scan_time;

  previous_previous_velocity_ = previous_velocity_;
  previous_velocity_ = current_velocity_;
  previous_velocity_x_ = current_velocity_x_;
  previous_velocity_y_ = current_velocity_y_;
  previous_velocity_z_ = current_velocity_z_;
  previous_accel_ = current_accel_;
}

void NDTMatching::imu_odom_calc(ros::Time current_time)
{
  static ros::Time previous_time = current_time;
  double diff_time = (current_time - previous_time).toSec();

  double diff_imu_roll = imu_.angular_velocity.x * diff_time;
  double diff_imu_pitch = imu_.angular_velocity.y * diff_time;
  double diff_imu_yaw = imu_.angular_velocity.z * diff_time;

  current_pose_imu_odom_.roll += diff_imu_roll;
  current_pose_imu_odom_.pitch += diff_imu_pitch;
  current_pose_imu_odom_.yaw += diff_imu_yaw;

  double diff_distance = odom_.twist.twist.linear.x * diff_time;
  offset_imu_odom_x_ += diff_distance * cos(-current_pose_imu_odom_.pitch) * cos(current_pose_imu_odom_.yaw);
  offset_imu_odom_y_ += diff_distance * cos(-current_pose_imu_odom_.pitch) * sin(current_pose_imu_odom_.yaw);
  offset_imu_odom_z_ += diff_distance * sin(-current_pose_imu_odom_.pitch);

  offset_imu_odom_roll_ += diff_imu_roll;
  offset_imu_odom_pitch_ += diff_imu_pitch;
  offset_imu_odom_yaw_ += diff_imu_yaw;

  predict_pose_imu_odom_.x = previous_pose_.x + offset_imu_odom_x_;
  predict_pose_imu_odom_.y = previous_pose_.y + offset_imu_odom_y_;
  predict_pose_imu_odom_.z = previous_pose_.z + offset_imu_odom_z_;
  predict_pose_imu_odom_.roll = previous_pose_.roll + offset_imu_odom_roll_;
  predict_pose_imu_odom_.pitch = previous_pose_.pitch + offset_imu_odom_pitch_;
  predict_pose_imu_odom_.yaw = previous_pose_.yaw + offset_imu_odom_yaw_;

  previous_time = current_time;
}

void NDTMatching::odom_calc(ros::Time current_time)
{
  static ros::Time previous_time = current_time;
  double diff_time = (current_time - previous_time).toSec();

  double diff_odom_roll = odom_.twist.twist.angular.x * diff_time;
  double diff_odom_pitch = odom_.twist.twist.angular.y * diff_time;
  double diff_odom_yaw = odom_.twist.twist.angular.z * diff_time;

  current_pose_odom_.roll += diff_odom_roll;
  current_pose_odom_.pitch += diff_odom_pitch;
  current_pose_odom_.yaw += diff_odom_yaw;

  double diff_distance = odom_.twist.twist.linear.x * diff_time;
  offset_odom_x_ += diff_distance * cos(-current_pose_odom_.pitch) * cos(current_pose_odom_.yaw);
  offset_odom_y_ += diff_distance * cos(-current_pose_odom_.pitch) * sin(current_pose_odom_.yaw);
  offset_odom_z_ += diff_distance * sin(-current_pose_odom_.pitch);

  offset_odom_roll_ += diff_odom_roll;
  offset_odom_pitch_ += diff_odom_pitch;
  offset_odom_yaw_ += diff_odom_yaw;

  predict_pose_odom_.x = previous_pose_.x + offset_odom_x_;
  predict_pose_odom_.y = previous_pose_.y + offset_odom_y_;
  predict_pose_odom_.z = previous_pose_.z + offset_odom_z_;
  predict_pose_odom_.roll = previous_pose_.roll + offset_odom_roll_;
  predict_pose_odom_.pitch = previous_pose_.pitch + offset_odom_pitch_;
  predict_pose_odom_.yaw = previous_pose_.yaw + offset_odom_yaw_;

  previous_time = current_time;
}

void NDTMatching::imu_calc(ros::Time current_time)
{
  static ros::Time previous_time = current_time;
  double diff_time = (current_time - previous_time).toSec();

  double diff_imu_roll = imu_.angular_velocity.x * diff_time;
  double diff_imu_pitch = imu_.angular_velocity.y * diff_time;
  double diff_imu_yaw = imu_.angular_velocity.z * diff_time;

  current_pose_imu_.roll += diff_imu_roll;
  current_pose_imu_.pitch += diff_imu_pitch;
  current_pose_imu_.yaw += diff_imu_yaw;

  double accX1 = imu_.linear_acceleration.x;
  double accY1 = std::cos(current_pose_imu_.roll) * imu_.linear_acceleration.y -
                 std::sin(current_pose_imu_.roll) * imu_.linear_acceleration.z;
  double accZ1 = std::sin(current_pose_imu_.roll) * imu_.linear_acceleration.y +
                 std::cos(current_pose_imu_.roll) * imu_.linear_acceleration.z;

  double accX2 = std::sin(current_pose_imu_.pitch) * accZ1 + std::cos(current_pose_imu_.pitch) * accX1;
  double accY2 = accY1;
  double accZ2 = std::cos(current_pose_imu_.pitch) * accZ1 - std::sin(current_pose_imu_.pitch) * accX1;

  double accX = std::cos(current_pose_imu_.yaw) * accX2 - std::sin(current_pose_imu_.yaw) * accY2;
  double accY = std::sin(current_pose_imu_.yaw) * accX2 + std::cos(current_pose_imu_.yaw) * accY2;
  double accZ = accZ2;

  offset_imu_x_ += current_velocity_imu_x_ * diff_time + accX * diff_time * diff_time / 2.0;
  offset_imu_y_ += current_velocity_imu_y_ * diff_time + accY * diff_time * diff_time / 2.0;
  offset_imu_z_ += current_velocity_imu_z_ * diff_time + accZ * diff_time * diff_time / 2.0;

  current_velocity_imu_x_ += accX * diff_time;
  current_velocity_imu_y_ += accY * diff_time;
  current_velocity_imu_z_ += accZ * diff_time;

  offset_imu_roll_ += diff_imu_roll;
  offset_imu_pitch_ += diff_imu_pitch;
  offset_imu_yaw_ += diff_imu_yaw;

  predict_pose_imu_.x = previous_pose_.x + offset_imu_x_;
  predict_pose_imu_.y = previous_pose_.y + offset_imu_y_;
  predict_pose_imu_.z = previous_pose_.z + offset_imu_z_;
  predict_pose_imu_.roll = previous_pose_.roll + offset_imu_roll_;
  predict_pose_imu_.pitch = previous_pose_.pitch + offset_imu_pitch_;
  predict_pose_imu_.yaw = previous_pose_.yaw + offset_imu_yaw_;

  previous_time = current_time;
}
