#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/exe/feature.h"
#include "colmap/feature/sift.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/base_controller.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <csignal>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tqdm.h"
#include <H5Cpp.h>
#include <glog/logging.h>

using namespace colmap;

// Include necessary headers for database and SFM operations

namespace fs = std::filesystem;

std::atomic<int> gSignalThatStoppedMe = -1;
void ImportImages(const std::string& database_path,
                  const std::string& image_path,
                  const CameraMode camera_mode,
                  const std::vector<std::string>& image_list,
                  const ImageReaderOptions& options_) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_DIR_EXISTS(image_path);

  ImageReaderOptions options(options_);
  options.database_path = database_path;
  options.image_path = image_path;
  options.image_list = image_list;
  UpdateImageReaderOptionsFromCameraMode(options, camera_mode);

  Database database(options.database_path);
  DatabaseTransaction database_transaction(&database);
  ImageReader image_reader(options, &database);
  LOG(INFO) << "Importing images into the database...\n";

  for ([[maybe_unused]] unsigned long a :
       tq::trange(image_reader.NumImages())) {
    if (image_reader.NextIndex() >= image_reader.NumImages()) {
      break;
    }
    if (gSignalThatStoppedMe.load() != -1) {  // gracefully stop
      LOG(INFO) << "Stopping at " << image_reader.NextIndex() << "\n";
      break;
    }
    Camera camera;
    Image image;
    Bitmap bitmap;
    if (image_reader.Next(&camera, &image, &bitmap, nullptr) !=
        ImageReader::Status::SUCCESS) {
      continue;
    }
    if (image.ImageId() == kInvalidImageId) {
      image.SetImageId(database.WriteImage(image));
    }
  }
}

std::unordered_map<std::string, int> GetImageIDs(colmap::Database& db) {
  std::vector<Image> images = db.ReadAllImages();
  std::unordered_map<std::string, int> image_ids;

  for (const auto& item : images) {
    // LOG(INFO) << item.Name() << item.ImageId() << "\n";
    image_ids.insert({item.Name(), item.ImageId()});
  }
  return image_ids;
}

FeatureKeypointsBlob get_keypoints(const std::string& path,
                                   const std::string& name,
                                   bool return_uncertainty = false) {
  assert(return_uncertainty == false);  // not implemented

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      keypoints;
  double uncertainty = 0.0;
  //    LOG(INFO) << name << "\n";
  try {
    // Open the HDF5 file
    H5::H5File file(path, H5F_ACC_RDONLY);

    // Access the group and the dataset
    H5::Group group = file.openGroup(name);
    /*// Get the number of objects in the group
    hsize_t num_objs = group.getNumObjs();

    // Iterate over the objects
    for (hsize_t i = 0; i < num_objs; ++i) {
      // Get the name of the object
      std::string obj_name = group.getObjnameByIdx(i);

      // Check if the object is a dataset
      H5O_info_t obj_info;
      group.getObjinfo(obj_name.c_str(), obj_info);

      if (obj_info.type == H5O_TYPE_DATASET) {
        LOG(INFO) << (obj_name) << "\n";
      }
    }*/
    H5::DataSet dataset = group.openDataSet("keypoints");

    // Get the dataspace and allocate memory for the data
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims, NULL);

    keypoints.resize(dims[0], dims[1]);

    // Read the dataset into the array
    // TODO check HALF
    dataset.read(keypoints.data(), H5::PredType::NATIVE_DOUBLE);
    //      LOG(INFO) << keypoints << "\n";

    // Check for the uncertainty attribute
    if (return_uncertainty && dataset.attrExists("uncertainty")) {
      H5::Attribute attr = dataset.openAttribute("uncertainty");
      attr.read(H5::PredType::NATIVE_DOUBLE, &uncertainty);
    }

    file.close();
  } catch (H5::FileIException& error) {
    error.printErrorStack();
  } catch (H5::GroupIException& error) {
    error.printErrorStack();
  } catch (H5::DataSetIException& error) {
    error.printErrorStack();
  } catch (H5::DataSpaceIException& error) {
    error.printErrorStack();
  } catch (H5::AttributeIException& error) {
    error.printErrorStack();
  }

  return keypoints.cast<float>();
}

void ImportFeatures(std::unordered_map<std::string, int>& image_ids,
                    const std::string& database_path,
                    const std::string& features_path) {
  LOG(INFO) << "Importing features into the database...\n";
  Database db(database_path);
  DatabaseTransaction database_transaction(&db);
  for (const auto& item : tq::tqdm(image_ids)) {
    if (gSignalThatStoppedMe.load() != -1) {  // gracefully stop
      LOG(INFO) << "Stopping at " << item.first << " " << item.second << "\n";
      break;
    }
    auto& image_name = item.first;
    auto& image_id = item.second;
    FeatureKeypointsBlob keypoints = get_keypoints(features_path, image_name);
    keypoints.array() += 0.5;  // COLMAP origin
    db.WriteKeypoints(image_id, keypoints);
  }
}

// Function to replace all occurrences of a character in a string
std::string replace_all(const std::string& str, char old_char, char new_char) {
  std::string new_str = str;
  std::replace(new_str.begin(), new_str.end(), old_char, new_char);
  return new_str;
}

// Function to create a pair name from two strings
std::string names_to_pair(const std::string& name0,
                          const std::string& name1,
                          const std::string& separator = "/") {
  std::string replaced_name0 = replace_all(name0, '/', '-');
  std::string replaced_name1 = replace_all(name1, '/', '-');
  return replaced_name0 + separator + replaced_name1;
}

// Function to create a pair name from two strings using an older format
std::string names_to_pair_old(const std::string& name0,
                              const std::string& name1) {
  return names_to_pair(name0, name1, "-");
}

// Function to find the appropriate pair
std::pair<std::string, bool> find_pair(const H5::H5File& hfile,
                                       const std::string& name0,
                                       const std::string& name1) {
  std::string pair = names_to_pair(name0, name1);
  if (H5Lexists(hfile.getId(), pair.c_str(), H5P_DEFAULT)) {
    return {pair, false};
  }
  pair = names_to_pair(name1, name0);
  if (H5Lexists(hfile.getId(), pair.c_str(), H5P_DEFAULT)) {
    return {pair, true};
  }
  pair = names_to_pair_old(name0, name1);
  if (H5Lexists(hfile.getId(), pair.c_str(), H5P_DEFAULT)) {
    return {pair, false};
  }
  pair = names_to_pair_old(name1, name0);
  if (H5Lexists(hfile.getId(), pair.c_str(), H5P_DEFAULT)) {
    return {pair, true};
  }
  throw std::runtime_error(
      "Could not find pair (" + name0 + ", " + name1 +
      ")... Maybe you matched with a different list of pairs?");
}

// Define a utility function to create an Eigen::Map from
// std::vector<std::pair<int, int>>
Eigen::Map<Eigen::Matrix<uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>>
vector_to_eigen_map(std::vector<std::pair<int, int>>& vec) {
  // Ensure that std::pair<int, int> is laid out contiguously as expected
  static_assert(sizeof(std::pair<int, int>) == 2 * sizeof(int),
                "std::pair<int, int> is not laid out as expected");

  // Create an Eigen::Map that treats the vector data as an Eigen matrix
  return Eigen::Map<
      Eigen::Matrix<uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>>(
      reinterpret_cast<uint32_t*>(vec.data()), vec.size(), 2);
}

std::tuple<FeatureMatchesBlob, std::vector<double>> get_matches(
    const std::string& path,
    const std::string& name0,
    const std::string& name1) {
  std::vector<std::pair<int, int>> matches;
  std::vector<double> scores;

  try {
    // Open the HDF5 file
    H5::H5File file(path, H5F_ACC_RDONLY);

    // Find the pair and check if reverse is needed
    auto [pair, reverse] = find_pair(file, name0, name1);

    // Open the datasets
    H5::DataSet matches_dataset = file.openDataSet(pair + "/matches0");
    H5::DataSet scores_dataset = file.openDataSet(pair + "/matching_scores0");

    // Read the matches dataset
    H5::DataSpace matches_dataspace = matches_dataset.getSpace();
    hsize_t matches_dims[1];
    matches_dataspace.getSimpleExtentDims(matches_dims, NULL);
    std::vector<int> matches_data(matches_dims[0]);
    matches_dataset.read(matches_data.data(), H5::PredType::NATIVE_INT);

    // Read the scores dataset
    H5::DataSpace scores_dataspace = scores_dataset.getSpace();
    hsize_t scores_dims[1];
    scores_dataspace.getSimpleExtentDims(scores_dims, NULL);
    scores.resize(scores_dims[0]);
    scores_dataset.read(scores.data(), H5::PredType::NATIVE_DOUBLE);

    // Process the matches
    for (size_t i = 0; i < matches_data.size(); ++i) {
      if (matches_data[i] != -1) {
        matches.emplace_back(i, matches_data[i]);
      }
    }

    // Filter scores based on valid matches
    std::vector<double> filtered_scores;
    for (const auto& match : matches) {
      filtered_scores.push_back(scores[match.first]);
    }
    scores = std::move(filtered_scores);

    // Reverse the matches if needed
    if (reverse) {
      for (auto& match : matches) {
        std::swap(match.first, match.second);
      }
    }
  } catch (H5::FileIException& error) {
    error.printErrorStack();
  } catch (H5::GroupIException& error) {
    error.printErrorStack();
  } catch (H5::DataSetIException& error) {
    error.printErrorStack();
  } catch (H5::DataSpaceIException& error) {
    error.printErrorStack();
  } catch (std::runtime_error& error) {
    std::cerr << error.what() << '\n';
  }

  auto matches_blob = vector_to_eigen_map(matches);
  //  LOG(INFO) << matches_blob << "\n";

  return std::make_tuple(matches_blob, scores);
}

void ImportMatches(const std::unordered_map<std::string, int>& image_ids,
                   const std::string& database_path,
                   const std::string& pairs_path,
                   const std::string& matches_path,
                   const std::optional<double>& min_match_score = std::nullopt,
                   bool skip_geometric_verification = false,
                   unsigned long start_index = 0) {
  LOG(INFO) << ("Importing matches into the database...\n");

  // Read pairs from the file
  std::ifstream pairs_file(pairs_path);
  std::vector<std::pair<std::string, std::string>> pairs;
  std::string line;
  while (std::getline(pairs_file, line)) {
    std::istringstream iss(line);
    std::string name0, name1;
    if (!(iss >> name0 >> name1)) {
      break;
    }
    pairs.emplace_back(name0, name1);
  }
  pairs_file.close();

  // Connect to the database
  Database db(database_path);
  std::unique_ptr<DatabaseTransaction> database_transaction;

  // std::set<std::pair<int, int>> matched;

  auto matches = db.ReadAllMatches();
  /*for (const auto& [image_pair_t, FeatureMatches] : matches) {
    auto pair = Database::PairIdToImagePair(image_pair_t);
    matched.insert(pair);
    matched.insert({pair.second, pair.first});
  }*/
  for (auto idx : tq::trange(start_index, pairs.size())) {
    if (gSignalThatStoppedMe.load() != -1) {  // gracefully stop
        LOG(INFO) << "Stopping at " << idx << "\n";
        break;
    }
    if (idx % 10000 == 0) {
      LOG(INFO) << "Creating database transaction\n";

      database_transaction = std::make_unique<DatabaseTransaction>(&db);
    }

    const auto& [name0, name1] = pairs[idx];
    int id0 = image_ids.at(name0);
    int id1 = image_ids.at(name1);

    /*if (matched.count({id0, id1}) > 0 || matched.count({id1, id0}) > 0) {
      continue;
    }*/

    auto [matches, scores] = get_matches(matches_path, name0, name1);
    /*
        if (min_match_score) {
          FeatureMatchesBlob filtered_matches;
          for (size_t j = 0; j < scores.size(); ++j) {
            if (scores[j] > min_match_score.value()) {
              filtered_matches.push_back(matches[j]);
            }
          }
          matches = std::move(filtered_matches);
        }*/

    try {
        db.WriteMatches(id0, id1, matches);
    } catch(std::runtime_error err){
      LOG(ERROR) << "Stopping at " << idx << "\n";
    }
//    matched.insert({id0, id1});
//    matched.insert({id1, id0});
  }
}

void signalHandler(int signum) {
  LOG(INFO) << "Interrupt signal (" << signum << ") received.\n";
  gSignalThatStoppedMe.store(signum);
}

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);

  LOG(INFO) << "\nSetting signal handlers\n";

  std::signal(SIGTERM, signalHandler);
  std::signal(SIGSEGV, signalHandler);
  std::signal(SIGINT, signalHandler);
  std::signal(SIGABRT, signalHandler);
  std::signal(SIGHUP, signalHandler);

  OptionManager options;
  std::string database_path, pairs_path, features_path, matches_path,
      image_path;
  bool import_images = true;
  bool import_features = true;
  bool import_matches = true;
  unsigned long matches_start = 0;
  options.AddRequiredOption("database", &database_path);
  options.AddRequiredOption("pairs", &pairs_path);
  options.AddRequiredOption("features", &features_path);
  options.AddRequiredOption("matches", &matches_path);
  options.AddRequiredOption("image_dir", &image_path);
  options.AddDefaultOption("import_images", &import_images);
  options.AddDefaultOption("import_features", &import_features);
  options.AddDefaultOption("import_matches", &import_matches);
  options.AddDefaultOption("matches_start", &matches_start);
  options.Parse(argc, argv);
  //  std::string database_path =
  //  "/mnt/hdd/3d_recon/theater_hloc/features/database.db"; std::string
  //  pairs_path = "/mnt/hdd/3d_recon/theater_hloc/features/pairs.txt";
  //  std::string features_path =
  //  "/mnt/hdd/3d_recon/theater_hloc/features/feats-disk.h5"; std::string
  //  matches_path =
  //  "/mnt/hdd/3d_recon/theater_hloc/features/feats-disk_matches-disk-lightglue_pairs.h5";
  //  std::string image_path =
  //  "/mnt/hdd/datasets/own/budapest/nepszinhaz_medium/images";

  if (import_images && import_features && import_matches &&
      ExistsFile(database_path)) {
    LOG(ERROR) << "File " << (database_path) << " exists.";
    return EXIT_FAILURE;
  }

  colmap::Database db(database_path);
  if (import_images) {
    ImportImages(database_path,
                 image_path,
                 CameraMode::AUTO,
                 std::vector<std::string>(),
                 {});
  }
  auto image_ids = GetImageIDs(db);

  if (import_features) {
    ImportFeatures(image_ids, database_path, features_path);
  }
  if (import_matches) {
    ImportMatches(image_ids,
                  database_path,
                  pairs_path,
                  matches_path,
                  std::nullopt,
                  false,
                  matches_start);
  }

  return 0;
}