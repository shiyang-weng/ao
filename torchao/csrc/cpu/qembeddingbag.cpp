#include <torch/all.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <c10/util/Unroll.h>
#include <c10/util/Float8_e4m3fn.h>

namespace torchao {

namespace {

static inline int32_t _scale_int32(int32_t value, float scale) {
  auto v_simd = _mm_setzero_ps();
  auto s_simd = _mm_set1_ps(scale);
  v_simd = _mm_cvt_si2ss(v_simd, value);
  v_simd = _mm_mul_round_ss(
      v_simd, s_simd, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  int32_t c = _mm_cvt_roundss_si32(
      v_simd, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  auto c_simd = _mm_set1_epi32(c);
  c_simd = _mm_cvtsepi32_epi8(c_simd);
  c = _mm_cvtsi128_si32(c_simd);
  return c;
}

#if defined(CPU_CAPABILITY_AVX512)
__m512 _mm512_load_e4m3_cvt_ps(const at::Float8_e4m3fn * weight, float* buf) {
  for (int i = 0; i < 16; i++) {
    buf[i] = static_cast<float>(weight[i]);
  }
  return _mm512_loadu_ps(buf);
}

void _mm512_ps_cvt_e4m3(at::Float8_e4m3fn* result, const __m512 x, float * buf) {
  _mm512_storeu_ps(buf, x);
  for (int i = 0; i < 16; i++) {
    result[i] = static_cast<at::Float8_e4m3fn>(buf[i]);
  }
}
#endif

template <typename index_t>
inline void qembeddingbag_kern(
    const int64_t bs_begin,
    const int64_t bs_end,
    const int64_t num_emb,
    const int64_t emb_dim,
    const index_t last_offset,
    const index_t* indices,
    const index_t* offsets,
    const at::Float8_e4m3fn* weight,
    const double scale,
    float* result) {
#if defined(CPU_CAPABILITY_AVX512)
  if (emb_dim % 128 == 0) {
      constexpr int64_t block_dim = 128;
      const int64_t num_blocks = emb_dim / block_dim;
      __m512 scale_v = _mm512_set1_ps(scale);
      float buf[16];
      for (int64_t b = bs_begin; b < bs_end; ++b) {
        __m512 x0, x1, x2, x3, x4, x5, x6, x7;
        int64_t start_idx = offsets[b];
        int64_t end_idx = ((b + 1) == bs_end && last_offset != -1)
            ? last_offset
            : offsets[b + 1];
        for (int64_t block_id = 0; block_id < num_blocks; block_id++) {
          // load first indices
          int64_t idx = indices[start_idx] * emb_dim + block_dim * block_id;
          float* block_result = result + block_dim * block_id;
          x0 = _mm512_load_e4m3_cvt_ps(&weight[idx], buf);
          x1 = _mm512_load_e4m3_cvt_ps(&weight[idx + 16], buf);
          x2 = _mm512_load_e4m3_cvt_ps(&weight[idx + 32], buf);
          x3 = _mm512_load_e4m3_cvt_ps(&weight[idx + 48], buf);
          x4 = _mm512_load_e4m3_cvt_ps(&weight[idx + 64], buf);
          x5 = _mm512_load_e4m3_cvt_ps(&weight[idx + 80], buf);
          x6 = _mm512_load_e4m3_cvt_ps(&weight[idx + 96], buf);
          x7 = _mm512_load_e4m3_cvt_ps(&weight[idx + 112], buf);
          for (int64_t j = start_idx + 1; j < end_idx; ++j) {
            // add following idx
            idx = indices[j] * emb_dim + block_dim * block_id;
            x0 = _mm512_add_ps(x0, _mm512_load_e4m3_cvt_ps(&weight[idx], buf));
            x1 = _mm512_add_ps(x1, _mm512_load_e4m3_cvt_ps(&weight[idx + 16], buf));
            x2 = _mm512_add_ps(x2, _mm512_load_e4m3_cvt_ps(&weight[idx + 32], buf));
            x3 = _mm512_add_ps(x3, _mm512_load_e4m3_cvt_ps(&weight[idx + 48], buf));
            x4 = _mm512_add_ps(x4, _mm512_load_e4m3_cvt_ps(&weight[idx + 64], buf));
            x5 = _mm512_add_ps(x5, _mm512_load_e4m3_cvt_ps(&weight[idx + 80], buf));
            x6 = _mm512_add_ps(x6, _mm512_load_e4m3_cvt_ps(&weight[idx + 96], buf));
            x7 = _mm512_add_ps(x7, _mm512_load_e4m3_cvt_ps(&weight[idx + 112], buf));
          }
          x0 = _mm512_mul_ps(x0, scale_v);
          x1 = _mm512_mul_ps(x1, scale_v);
          x2 = _mm512_mul_ps(x2, scale_v);
          x3 = _mm512_mul_ps(x3, scale_v);
          x4 = _mm512_mul_ps(x4, scale_v);
          x5 = _mm512_mul_ps(x5, scale_v);
          x6 = _mm512_mul_ps(x6, scale_v);
          x7 = _mm512_mul_ps(x7, scale_v);
          // store
          _mm512_store_ps(block_result, x0);
          _mm512_store_ps(block_result + 16, x1);
          _mm512_store_ps(block_result + 32, x2);
          _mm512_store_ps(block_result + 48, x3);
          _mm512_store_ps(block_result + 64, x4);
          _mm512_store_ps(block_result + 80, x5);
          _mm512_store_ps(block_result + 96, x6);
          _mm512_store_ps(block_result + 112, x7);
        }
        result += num_emb * emb_dim;
      }
    return;
  }
#endif
  for (int64_t b = bs_begin; b < bs_end; ++b) {
    int64_t start_idx = offsets[b];
    int64_t end_idx =
        ((b + 1) == bs_end && last_offset != -1) ? last_offset : offsets[b + 1];
    for (int64_t d = 0; d < emb_dim; d++) {
      int64_t idx = indices[start_idx] * emb_dim;
      float value = float(weight[idx + d]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        idx = indices[j] * emb_dim;
        value += float(weight[idx + d]);
      }
      value = value*scale;
      result[d] = value;
    }
    result += num_emb * emb_dim;
  }
}

template <typename index_t, typename data_t>
void qembeddingbagcat(
    float* o_ptr,
    data_t** w_ptr,
    index_t** indices_ptr,
    index_t** offsets_ptr,
    // int8_t* d_ptr,
    int64_t num_batch,
    int64_t num_emb,
    int64_t emb_dim,
    std::vector<int64_t> last_offsets,
    std::vector<double> w_scale,
    // double d_scale,
    double o_scale) {
  constexpr int64_t b_block = 512;
  const int64_t n_b_blocks = (num_batch - 1) / b_block + 1;
  // const double copy_scale = d_scale / o_scale;
  for (double& w_sca : w_scale) {
    w_sca = w_sca / o_scale;
  }
#pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < n_b_blocks; ++b) {
    for (int64_t n = 0; n < num_emb; ++n) {
      const int64_t bs_begin = b * b_block;
      const int64_t bs_end = std::min(num_batch, (b + 1) * b_block);
      float* r = &o_ptr[b * b_block * num_emb * emb_dim + n * emb_dim];
      const int64_t m = n;
      // avoid offsets not include last batch
      const index_t last_offset = bs_end == num_batch ? last_offsets[m] : -1;
      qembeddingbag_kern(
          bs_begin,
          bs_end,
          num_emb,
          emb_dim,
          last_offset,
          indices_ptr[m],
          offsets_ptr[m],
          w_ptr[m],
          w_scale[m],
          r);
    }
  }
}

at::Tensor qembeddingbag_impl(
    const at::TensorList& qweights,
    const at::TensorList& indices,
    const at::TensorList& offsets,
    const at::Tensor& w_scales,
    double o_scale,
    int64_t batch_size) {
  int64_t num_emb = qweights.size();
  TORCH_CHECK(num_emb > 0);
  TORCH_CHECK(num_emb == indices.size());
  TORCH_CHECK(num_emb == offsets.size());
  int64_t emb_dim = qweights[0].size(1);

  auto index_type = indices[0].scalar_type();
  auto qtype = qweights[0].scalar_type();

  std::vector<int64_t> last_offsets(num_emb, -1);
  std::vector<double> w_scale(num_emb, -1);
  float * w_scale_ptrs = w_scales.data_ptr<float>();

  for (int i = 0; i < num_emb; i++) {
    TORCH_CHECK(
        indices[i].is_contiguous() && indices[i].scalar_type() == index_type);
    TORCH_CHECK(
        offsets[i].is_contiguous() && offsets[i].scalar_type() == index_type);
    TORCH_CHECK(
        qweights[i].is_contiguous() && qweights[i].scalar_type() == qtype);
    TORCH_CHECK(
        qweights[i].dim() == 2 && qweights[i].size(1) == emb_dim);
    // handle last offsets
    last_offsets[i] = indices[i].numel();
    w_scale[i] = w_scale_ptrs[i];
  }
  at::Tensor output = at::empty({batch_size, num_emb * emb_dim}, qweights[0].options().dtype(at::kFloat));
  AT_DISPATCH_INDEX_TYPES(indices[0].scalar_type(), "embeddingbag_cat", [&] {
    at::Float8_e4m3fn* qweights_ptr[num_emb];
    index_t* indices_ptr[num_emb];
    index_t* offsets_ptr[num_emb];
    for (int i = 0; i < num_emb; i++) {
      qweights_ptr[i] = qweights[i].data_ptr<at::Float8_e4m3fn>();
      indices_ptr[i] = indices[i].data_ptr<index_t>();
      offsets_ptr[i] = offsets[i].data_ptr<index_t>();
    }
    float* output_ptr = output.data_ptr<float>();
    qembeddingbagcat<index_t, at::Float8_e4m3fn>(
        output_ptr,
        qweights_ptr,
        indices_ptr,
        offsets_ptr,
        batch_size,
        num_emb,
        emb_dim,
        last_offsets,
        w_scale,
        o_scale);
  });
  return output;
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::qembeddingbag", &qembeddingbag_impl);
}

} // namespace torchao