#pragma once

#include <openpgl/config.h>
#include <openpgl/cpp/OpenPGL.h>
#include <openpgl/cpp/Region.h>
#include <openpgl/cpp/SurfaceSamplingDistribution.h>
#include <openpgl/pathsegmentstorage.h>

#include <mutex>

#include "yocto/yocto_math.h"
#include "yocto/yocto_sampling.h"

namespace yocto {
namespace details {
inline auto get_default_field_arguments() {
  openpgl::cpp::FieldArguments field_arguments{};
  pglFieldArgumentsSetDefaults(field_arguments,
      PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
      PGL_DIRECTIONAL_DISTRIBUTION_TYPE::
          PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);
  // reinterpret_cast<PGLKDTreeArguments*>(
  //     field_arguments.spatialSturctureArguments)
  //     ->maxDepth = 16;
  return field_arguments;
}
}  // namespace details

inline pgl_vec3f to_pgl(const vec3f& v) { return {v.x, v.y, v.z}; }
inline pgl_vec2f to_pgl(const vec2f& v) { return {v.x, v.y}; }
inline vec3f     from_pgl(const pgl_vec3f& v) { return {v.x, v.y, v.z}; }
inline vec2f     from_pgl(const pgl_vec2f& v) { return {v.x, v.y}; }

constexpr auto g_max_training_iterations = 128;
constexpr auto g_min_training_iterations = g_max_training_iterations / 16;
constexpr auto g_max_training_samples    = 1024;

struct guiding_field {
  guiding_field(openpgl::cpp::Device& device)
      : m_field{&device, details::get_default_field_arguments()} {}

  bool update(openpgl::cpp::SampleStorage& sample_storage) {
    std::lock_guard<std::mutex> lock{m_update_lock};
    if (!m_train) {
      return false;
    }

    const auto num_samples = sample_storage.GetSizeSurface() +
                             sample_storage.GetSizeVolume();
    if (num_samples >= g_max_training_samples) {
      m_field.Update(sample_storage);
      sample_storage.Clear();

      if (m_field.GetIteration() >= g_max_training_iterations) {
        m_train = false;
      }

      return true;
    }

    return false;
  }

  template <typename T>
  auto create_sample_distribution() {
    return T{&m_field};
  }

  template <typename T>
  bool init_distrib(T& distribution, vec3f position, float random_float) {
    std::lock_guard<std::mutex> guard{m_update_lock};

    auto result = distribution.Init(&m_field, to_pgl(position), random_float);
    return result;
  }

  auto iterations() { return m_field.GetIteration(); }

  bool should_train() { return m_train; }

 private:
  openpgl::cpp::Field m_field;
  bool                m_train = true;
  std::mutex          m_update_lock;
  std::mutex          m_init_lock;
};

enum class guiding {
  InUse,
  Tentative,
  Unused,
};

constexpr float g_path_guiding_prob = 0.50f;

template <typename T>
inline float adjust_pdf_for_guiding(float pdf, vec3f incoming,
    T& guiding_distribution, rng_state& rng, guiding guiding_info) {
  auto direction = pgl_vec3f{incoming.x, incoming.y, incoming.z};
  switch (guiding_info) {
    case guiding::InUse: {
      // We won the lottery and we used guiding for sampling the new
      // direction.

      // std::printf("Using path guiding\n");
      const auto guided_sample_pdf = guiding_distribution.SamplePDF(
          {rand1f(rng), rand1f(rng)}, direction);
      // Prob BSDF * PDF + Prob Guiding * GPDF
      pdf *= (1.0f - g_path_guiding_prob);
      pdf += g_path_guiding_prob * guided_sample_pdf;

      return pdf;
    }
    case guiding::Tentative: {
      // We lost the lottery and we didn't use guiding while it _was
      // possible_ to do so. In this case we adjust the total PDF to
      // account for the missed chance.

      const auto guiding_pdf = guiding_distribution.PDF(direction);
      pdf *= (1.0f - g_path_guiding_prob);
      pdf += g_path_guiding_prob * guiding_pdf;

      return pdf;
    }

    default: return pdf;
  }
}

}  // namespace yocto
